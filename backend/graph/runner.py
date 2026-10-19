"""Run driver: start, pause, resume and observe runs on a persistent checkpointer.

`Runner` ties the graph to the stores:

- the checkpointer (Postgres in production) holds the run state after every
  node, so `resume(run_id)` in a fresh process continues from the last
  checkpoint: completed nodes (and completed parallel branches, saved as
  pending writes) are not run again;
- the call ledger makes a step that is retried after a crash replay its LLM
  calls instead of paying for them again;
- the topic findings store lets researchers reuse fresh findings;
- the runs table carries status / outcome / cost, and the run events table
  gets one event per trace step (keyed by step id, so a replay never logs a
  step twice) for the SSE endpoint to poll.

Durability is "sync": a node's writes are committed before the next step
starts, so a killed process loses at most the steps that were running.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from typing import Any

from langgraph.types import Command

from backend.graph.budget import Budget
from backend.graph.builder import RECURSION_LIMIT, RunResult, build_graph, initial_state
from backend.graph.nodes import GraphDeps, _err
from backend.graph.plan_review import PlanApprovalRequest, PlanDecision
from backend.graph.state import RunFailure, RunOutcome
from backend.graph.trace import TraceStep
from backend.schemas import UserProfile
from backend.store.base import Stores
from backend.store.models import EventType, RunRecord

log = logging.getLogger(__name__)


def graph_config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


def awaiting_request(snapshot: Any) -> PlanApprovalRequest | None:
    """The pending plan-approval request of a paused run, if any."""
    for intr in getattr(snapshot, "interrupts", ()) or ():
        value = intr.value
        if isinstance(value, PlanApprovalRequest):
            return value
        if isinstance(value, dict) and "sub_questions" in value:
            return PlanApprovalRequest.model_validate(value)
    return None


def result_from_graph(graph: Any, run_id: str) -> RunResult | None:
    snapshot = graph.get_state(graph_config(run_id))
    if not snapshot.values:
        return None
    return RunResult.from_state(run_id, dict(snapshot.values), awaiting_request(snapshot))


def step_event(step: TraceStep) -> dict[str, Any]:
    """What the progress stream shows for one trace step (the full step is in the trace)."""
    return {
        "step_id": step.step_id,
        "node": step.node,
        "agent": step.agent,
        "sub_question_id": step.sub_question_id,
        "attempt": step.attempt,
        "status": step.status,
        "summary": step.output_summary,
        "decision": step.decision,
        "model": step.model,
        "tokens": step.input_tokens + step.output_tokens,
        "cost_usd": step.cost_usd,
        "latency_ms": step.latency_ms,
        "searches": sum(1 for c in step.tool_calls if c.tool.startswith("search:") and c.state != "refused"),
        "reused": any(c.tool == "findings_store" and c.state == "reused" for c in step.tool_calls),
        "replayed_calls": sum(1 for c in step.llm_calls if c.replayed),
    }


class Runner:
    def __init__(
        self,
        deps: GraphDeps,
        stores: Stores,
        *,
        reuse_findings: bool = True,
        durability: str = "sync",
    ):
        self.stores = stores
        self.deps = replace(
            deps,
            ledger=deps.ledger or stores.ledger,
            findings_store=(deps.findings_store or stores.findings) if reuse_findings else None,
        )
        self.durability = durability
        self.graph = build_graph(self.deps, checkpointer=stores.checkpointer)

    # --- public API ---------------------------------------------------------

    def start(
        self,
        question: str,
        profile: UserProfile | None = None,
        *,
        budget: Budget | None = None,
        run_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        require_plan_approval: bool = False,
    ) -> RunResult:
        run_id = run_id or uuid.uuid4().hex
        budget = budget or Budget()
        self.stores.runs.create(
            RunRecord(
                run_id=run_id, question=question, profile=profile, user_id=user_id,
                conversation_id=conversation_id, require_plan_approval=require_plan_approval,
            )
        )
        self._event(run_id, "run_started", {"question": question, "budget": budget.model_dump()}, key="run_started")
        state = initial_state(question, profile, budget, run_id, self.deps.clock(), require_plan_approval)
        return self._drive(run_id, state)

    def resume(self, run_id: str, decision: PlanDecision | dict[str, Any] | None = None) -> RunResult:
        """Continue a run from its last checkpoint.

        - Paused at the plan checkpoint: `decision` approves or edits the plan (without one,
          the pending request is returned unchanged). An invalid decision comes back as a
          new request with `error` set.
        - Interrupted by a crash or an error: re-runs only what had not finished.
        - Already finished: returns the stored result; nothing runs again.
        """
        snapshot = self.graph.get_state(graph_config(run_id))
        if not snapshot.values:
            raise KeyError(f"no checkpoint for run {run_id}")
        if not snapshot.next:
            return RunResult.from_state(run_id, dict(snapshot.values))
        pending = awaiting_request(snapshot)
        if pending is not None:
            if decision is None:
                return RunResult.from_state(run_id, dict(snapshot.values), pending)
            payload = decision.model_dump(mode="json") if isinstance(decision, PlanDecision) else decision
            self._event(run_id, "run_resumed", {"reason": "plan_decision", "action": payload.get("action")})
            return self._drive(run_id, Command(resume=payload))
        if decision is not None:
            raise ValueError(f"run {run_id} is not waiting for plan approval")
        self._event(run_id, "run_resumed", {"reason": "recovery", "next": list(snapshot.next)})
        return self._drive(run_id, None)

    def get(self, run_id: str) -> RunResult | None:
        return result_from_graph(self.graph, run_id)

    # --- internals ----------------------------------------------------------

    def _drive(self, run_id: str, graph_input: Any) -> RunResult:
        config = graph_config(run_id)
        self._update(run_id, status="running", error=None)
        try:
            for chunk in self.graph.stream(graph_input, config, stream_mode="updates", durability=self.durability):
                for node, update in chunk.items():
                    if node.startswith("__") or not isinstance(update, dict):
                        continue
                    for step in update.get("trace") or []:
                        self._event(run_id, "step", step_event(step), key=f"step:{step.step_id}")
        except Exception as e:  # e.g. a database outage: the run stays resumable from its checkpoint
            log.exception("run %s stopped with an error", run_id)
            message = _err(e)
            self._update(run_id, status="error", error=message)
            self._event(run_id, "run_error", {"error": message})
            try:
                state = dict(self.graph.get_state(config).values)
            except Exception:
                state = {}
            failure = RunFailure(reason="internal_error", node="graph", message=message)
            state["outcome"] = RunOutcome(
                status="failed", message=message, failure=failure, degradations=state.get("degradations") or []
            )
            return RunResult.from_state(run_id, state)

        snapshot = self.graph.get_state(config)
        pending = awaiting_request(snapshot)
        result = RunResult.from_state(run_id, dict(snapshot.values), pending)
        if pending is not None:
            ids = [getattr(i, "id", None) for i in snapshot.interrupts]
            self._update(run_id, status="awaiting_approval", cost_usd=result.spend.cost_usd)
            self._event(run_id, "plan_ready", pending.model_dump(mode="json"), key=f"plan_ready:{ids[0]}")
        else:
            assert result.outcome is not None
            outcome = result.outcome.model_dump(mode="json")
            self._update(
                run_id, status="completed", outcome_status=result.outcome.status, outcome=outcome,
                cost_usd=result.spend.cost_usd,
            )
            self._event(
                run_id, "run_finished",
                {"status": result.outcome.status, "message": result.outcome.message, "cost_usd": result.spend.cost_usd},
                key="run_finished",
            )
        return result

    def _event(self, run_id: str, type: EventType, data: dict[str, Any], *, key: str | None = None) -> None:
        try:
            self.stores.events.append(run_id, type, data, key=key)
        except Exception:  # progress events are best effort; the checkpoint is the source of truth
            log.warning("could not record %s event for run %s", type, run_id, exc_info=True)

    def _update(self, run_id: str, **fields: Any) -> None:
        try:
            self.stores.runs.update(run_id, **fields)
        except Exception:
            log.warning("could not update run %s", run_id, exc_info=True)

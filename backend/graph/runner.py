"""Run driver: create, advance, pause, resume, fork and observe runs on a persistent checkpointer.

`Runner` ties the graph to the stores:

- the checkpointer (Postgres in production) holds the run state after every
  node, so `resume(run_id)` / `advance(run_id)` in a fresh process continues
  from the last checkpoint: completed nodes (and completed parallel branches,
  saved as pending writes) are not run again;
- the call ledger makes a step that is retried after a crash replay its LLM
  calls instead of paying for them again;
- the topic findings store lets researchers reuse fresh findings;
- the runs table carries status / outcome / cost, and the run events table
  gets one event per trace step (keyed by step id, so a replay never logs a
  step twice) for the SSE endpoint to poll.

Durability is "sync": a node's writes are committed before the next step
starts, so a killed process loses at most the steps that were running.

Serverless (Phase 6): `advance` runs the graph in slices. It stops at the
first superstep boundary after `should_stop()` turns true (a checkpoint has
just been written, nothing is in flight), and the caller schedules the next
slice. `api/worker.py` wraps it in a lease.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from langgraph.types import Command

from backend.graph.budget import Budget, Spend
from backend.graph.builder import RECURSION_LIMIT, RunResult, build_graph, initial_state
from backend.graph.fork import ForkError, ForkInfo, fork_point, fork_values, scope_for
from backend.graph.nodes import GraphDeps, _err
from backend.graph.overrides import ScopedOverride, StepOverride
from backend.graph.plan_review import PlanApprovalRequest, PlanDecision
from backend.graph.progress import step_event
from backend.graph.state import RunFailure, RunOutcome
from backend.schemas import UserProfile
from backend.store.base import Stores
from backend.store.models import EventType, LedgerEntry, RunRecord

log = logging.getLogger(__name__)

__all__ = ["AdvanceOutcome", "Runner", "awaiting_request", "graph_config", "result_from_graph", "step_event"]


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
    pending = awaiting_request(snapshot)
    running = bool(snapshot.next) and pending is None
    return RunResult.from_state(run_id, dict(snapshot.values), pending, running=running)


AdvanceState = Literal["finished", "awaiting_approval", "paused", "error", "missing", "not_running"]


@dataclass
class AdvanceOutcome:
    """What one slice of work did. `paused` means more work remains: schedule another advance."""

    run_id: str
    state: AdvanceState
    result: RunResult | None = None
    steps: int = 0  # trace steps produced in this slice


def _totals(spend: Spend | None) -> dict[str, Any]:
    s = spend or Spend()
    return {"tokens": s.total_tokens, "cost_usd": round(s.cost_usd, 6), "llm_calls": s.llm_calls, "searches": s.searches}


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
        self._overrides: dict[str, ScopedOverride | None] = {}
        self._overrides_lock = threading.Lock()
        self.deps = replace(
            deps,
            ledger=deps.ledger or stores.ledger,
            findings_store=(deps.findings_store or stores.findings) if reuse_findings else None,
            overrides=deps.overrides or self._override_for,
        )
        self.durability = durability
        self.graph = build_graph(self.deps, checkpointer=stores.checkpointer)

    # --- public API ---------------------------------------------------------

    def create(
        self,
        question: str,
        profile: UserProfile | None = None,
        *,
        budget: Budget | None = None,
        run_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        require_plan_approval: bool = False,
    ) -> RunRecord:
        """Record a new run (status `running`); `advance` starts the graph."""
        run_id = run_id or uuid.uuid4().hex
        budget = budget or Budget()
        record = self.stores.runs.create(
            RunRecord(
                run_id=run_id, question=question, profile=profile, user_id=user_id,
                conversation_id=conversation_id, require_plan_approval=require_plan_approval,
                budget=budget.model_dump(mode="json"),
            )
        )
        self._event(run_id, "run_started", {"question": question, "budget": budget.model_dump()}, key="run_started")
        return record

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
        """Create a run and drive it until it finishes or pauses at the plan checkpoint."""
        record = self.create(
            question, profile, budget=budget, run_id=run_id, user_id=user_id,
            conversation_id=conversation_id, require_plan_approval=require_plan_approval,
        )
        out = self.advance(record.run_id)
        assert out.result is not None
        return out.result

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
            return self._drive(run_id, Command(resume=payload))[0]
        if decision is not None:
            raise ValueError(f"run {run_id} is not waiting for plan approval")
        self._event(run_id, "run_resumed", {"reason": "recovery", "next": list(snapshot.next)})
        return self._drive(run_id, None)[0]

    def advance(
        self,
        run_id: str,
        *,
        should_stop: Callable[[], bool] | None = None,
        heartbeat: Callable[[], bool] | None = None,
    ) -> AdvanceOutcome:
        """Do the next piece of work on a run, from whatever the record and checkpoint say.

        - no checkpoint yet: start the graph from the run record;
        - waiting at the plan checkpoint: resume with the record's `pending_decision`, if any;
        - mid-run (a previous slice paused, or a worker died): continue from the checkpoint;
        - finished: make sure the record says so (a crash may have hit between the two).

        `should_stop` is polled at superstep boundaries; `heartbeat` (e.g. renewing a lease) is
        called there too, and returning False stops at once (another worker owns the run now).
        """
        record = self.stores.runs.get(run_id)
        if record is None:
            return AdvanceOutcome(run_id, "missing")
        snapshot = self.graph.get_state(graph_config(run_id))
        decision_consumed = False
        if not snapshot.values:
            if record.status not in ("running", "error"):
                return AdvanceOutcome(run_id, "not_running")
            budget = Budget.model_validate(record.budget or {})
            graph_input: Any = initial_state(
                record.question, record.profile, budget, run_id, self.deps.clock(), record.require_plan_approval
            )
        elif not snapshot.next:
            result = RunResult.from_state(run_id, dict(snapshot.values))
            self._finish(run_id, result)
            return AdvanceOutcome(run_id, "finished", result)
        else:
            pending = awaiting_request(snapshot)
            if pending is not None:
                if record.pending_decision is None:
                    self._update(run_id, status="awaiting_approval")
                    return AdvanceOutcome(
                        run_id, "awaiting_approval", RunResult.from_state(run_id, dict(snapshot.values), pending)
                    )
                graph_input = Command(resume=record.pending_decision)
                decision_consumed = True
                self._event(
                    run_id, "run_resumed",
                    {"reason": "plan_decision", "action": record.pending_decision.get("action")},
                )
            else:
                graph_input = None
                if record.pending_decision is not None:  # stale: the decision already went in
                    self._update(run_id, pending_decision=None)
                if record.status == "error":
                    self._event(run_id, "run_resumed", {"reason": "retry", "next": list(snapshot.next)})
        result, paused, steps = self._drive(run_id, graph_input, should_stop=should_stop, heartbeat=heartbeat)
        if decision_consumed:
            self._update(run_id, pending_decision=None)
        if paused:
            return AdvanceOutcome(run_id, "paused", result, steps)
        if result.awaiting_approval is not None:
            return AdvanceOutcome(run_id, "awaiting_approval", result, steps)
        rec = self.stores.runs.get(run_id)
        state: AdvanceState = "error" if rec is not None and rec.status == "error" else "finished"
        return AdvanceOutcome(run_id, state, result, steps)

    def get(self, run_id: str) -> RunResult | None:
        return result_from_graph(self.graph, run_id)

    def fork(
        self,
        run_id: str,
        from_step: int,
        override: StepOverride,
        *,
        new_run_id: str | None = None,
        user_id: str | None = None,
    ) -> RunRecord:
        """Copy run `run_id` as it was just before trace step `from_step` into a new run whose
        step re-runs with `override`. Returns the new run's record (status `running`); `advance`
        it like any run. Raises KeyError (unknown run) or ForkError (invalid request)."""
        parent = self.stores.runs.get(run_id)
        result = self.get(run_id)
        if parent is None or result is None:
            raise KeyError(run_id)
        if parent.status not in ("completed", "error"):
            raise ForkError(f"run {run_id} is {parent.status}; fork a finished run")
        if not 0 <= from_step < len(result.trace):
            raise ForkError(f"from_step must be between 0 and {len(result.trace) - 1}")
        step = result.trace[from_step]
        scope = scope_for(step, override)
        snapshot, as_node = fork_point(self.graph, graph_config(run_id), from_step)

        new_run_id = new_run_id or uuid.uuid4().hex
        values = fork_values(dict(snapshot.values), new_run_id, self.deps.clock())
        inherited = len(values.get("trace") or [])
        copied = 0
        for entry in self.stores.ledger.list(run_id):
            if self.stores.ledger.put(LedgerEntry(**{**entry.model_dump(), "run_id": new_run_id})):
                copied += 1
        info = ForkInfo(
            parent_run_id=run_id, from_step=from_step, step_id=step.step_id, node=step.node,
            checkpoint_id=(snapshot.config or {}).get("configurable", {}).get("checkpoint_id"),
            scope=scope, inherited_steps=inherited, copied_ledger_entries=copied,
        )
        record = self.stores.runs.create(
            RunRecord(
                run_id=new_run_id, question=parent.question, profile=parent.profile,
                user_id=user_id if user_id is not None else parent.user_id,
                conversation_id=parent.conversation_id, require_plan_approval=False,
                budget=parent.budget, parent_run_id=run_id, fork=info.model_dump(mode="json"),
            )
        )
        with self._overrides_lock:
            self._overrides[new_run_id] = scope
        self.graph.update_state(graph_config(new_run_id), values, as_node=as_node)
        self._event(
            new_run_id, "run_started",
            {"question": parent.question, "fork": info.model_dump(mode="json")},
            key="run_started",
        )
        return record

    # --- internals ----------------------------------------------------------

    def _override_for(self, run_id: str) -> ScopedOverride | None:
        with self._overrides_lock:
            if run_id in self._overrides:
                return self._overrides[run_id]
        record = self.stores.runs.get(run_id)
        scoped = ScopedOverride.model_validate(record.fork["scope"]) if record is not None and record.fork else None
        with self._overrides_lock:
            if len(self._overrides) > 4096:
                self._overrides.clear()
            self._overrides[run_id] = scoped
        return scoped

    def _drive(
        self,
        run_id: str,
        graph_input: Any,
        *,
        should_stop: Callable[[], bool] | None = None,
        heartbeat: Callable[[], bool] | None = None,
    ) -> tuple[RunResult, bool, int]:
        """Stream the graph. Returns (result, paused, steps produced)."""
        config = graph_config(run_id)
        self._update(run_id, status="running", error=None)
        paused = False
        steps = 0
        spend = Spend()
        try:
            for mode, chunk in self.graph.stream(
                graph_input, config, stream_mode=["updates", "checkpoints"], durability=self.durability
            ):
                if mode == "checkpoints":
                    values = chunk.get("values") or {}
                    spend = values.get("spend") or spend
                    if not chunk.get("next"):
                        continue
                    if heartbeat is not None and not heartbeat():
                        log.warning("run %s: lease lost; stopping this slice", run_id)
                        paused = True
                        break
                    if steps and should_stop is not None and should_stop():
                        paused = True
                        break
                    continue
                for node, update in chunk.items():
                    if node.startswith("__") or not isinstance(update, dict):
                        continue
                    for step in update.get("trace") or []:
                        steps += 1
                        spend = spend + Spend(
                            input_tokens=step.input_tokens, output_tokens=step.output_tokens, cost_usd=step.cost_usd,
                            llm_calls=len(step.llm_calls),
                        )
                        self._event(run_id, "step", step_event(step, _totals(spend)), key=f"step:{step.step_id}")
        except Exception as e:  # e.g. a database outage: the run stays resumable from its checkpoint
            log.exception("run %s stopped with an error", run_id)
            message = _err(e)
            record = self.stores.runs.get(run_id)
            self._update(run_id, status="error", error=message, attempts=(record.attempts + 1) if record else 1)
            self._event(run_id, "run_error", {"error": message})
            try:
                state = dict(self.graph.get_state(config).values)
            except Exception:
                state = {}
            failure = RunFailure(reason="internal_error", node="graph", message=message)
            state["outcome"] = RunOutcome(
                status="failed", message=message, failure=failure, degradations=state.get("degradations") or []
            )
            return RunResult.from_state(run_id, state), False, steps

        snapshot = self.graph.get_state(config)
        pending = awaiting_request(snapshot)
        if paused and snapshot.next and pending is None:
            return RunResult.from_state(run_id, dict(snapshot.values), running=True), True, steps
        result = RunResult.from_state(run_id, dict(snapshot.values), pending)
        if pending is not None:
            ids = [getattr(i, "id", None) for i in snapshot.interrupts]
            self._update(run_id, status="awaiting_approval", cost_usd=result.spend.cost_usd)
            # One event per request: a rejected decision re-asks with a new interrupt id.
            self._event(run_id, "plan_ready", pending.model_dump(mode="json"), key=f"plan_ready:{ids[0]}")
        else:
            self._finish(run_id, result)
        return result, False, steps

    def _finish(self, run_id: str, result: RunResult) -> None:
        assert result.outcome is not None
        self._update(
            run_id, status="completed", outcome_status=result.outcome.status,
            outcome=result.outcome.model_dump(mode="json"), cost_usd=result.spend.cost_usd, error=None,
        )
        self._event(
            run_id, "run_finished",
            {
                "status": result.outcome.status,
                "message": result.outcome.message,
                "cost_usd": result.spend.cost_usd,
                "totals": _totals(result.spend),
            },
            key="run_finished",
        )

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


__all__ += ["ForkError"]

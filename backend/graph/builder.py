"""The Evidence Coach state graph.

    START -> intake --out of scope--> referral -> END
               |
               v
            planner [-> plan_checkpoint (interrupt)] -> dispatch ==Send x N==> researcher -> review --send back (once)==> researcher
                                                              |
                                                              v
            personalize -> position_mapper --consensus / debate unaffordable--> judge -> writer
                                  |                                                ^
                                  +==Send x N==> advocate -> debate_review --------+
                                                    ^              |
                                                    +==Send x N====+  (exactly one rebuttal round)

            writer -> check_answer --rewrite (once)--> writer
                           |
                           v
                        finalize -> END

Any node can set `failure`; every routing function checks it first and sends
the run to `finalize`, which always writes a `RunOutcome`.

Synthesis (graph/synthesis.py) is wired by `_wire_synthesis`, entered at
`SYNTHESIS_ENTRY`. It never fails the run: without a position map the writer
works from findings, without a verdict from the position map.

Persistence (Phase 5): the checkpointer is a parameter; use
`checkpoint.memory_checkpointer()` locally or `store.postgres.postgres_checkpointer`.
With `require_plan_approval`, `plan_checkpoint` interrupts between the planner
and dispatch until `Command(resume=PlanDecision)`; without it the edge skips the
node (auto-approve). `graph/runner.py` drives runs with stores and events.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from backend.graph.budget import Budget, Spend
from backend.graph.nodes import GraphDeps, Nodes
from backend.graph.plan_review import PlanApprovalRequest, PlanDecision
from backend.graph.state import (
    DEBATE_ROUNDS,
    AnswerReview,
    ReviewReport,
    RunFailure,
    RunOutcome,
    RunState,
    all_findings,
)
from backend.graph.synthesis import SynthesisNodes
from backend.graph.trace import TraceStep
from backend.schemas import (
    CoachAnswer,
    DebateTurn,
    Finding,
    Intake,
    PositionMap,
    Record,
    ResearchPlan,
    SubQuestion,
    UserProfile,
    Verdict,
)

log = logging.getLogger(__name__)

SYNTHESIS_ENTRY = "personalize"
RECURSION_LIMIT = 40  # the graph is bounded by design; this only guards against wiring bugs


# --- Routing (supervisor decisions on edges) --------------------------------


def after_intake(state: RunState) -> str:
    if state.get("failure"):
        return "finalize"
    return "planner" if state["intake"].in_scope else "referral"


def after_planner(state: RunState) -> str:
    if state.get("failure"):
        return "finalize"
    return "plan_checkpoint" if state.get("require_plan_approval") else "dispatch"


def after_plan_checkpoint(state: RunState) -> str:
    return "finalize" if state.get("failure") else "dispatch"


def fan_out(state: RunState) -> list[Send] | str:
    if state.get("failure"):
        return "finalize"
    tasks = state.get("pending_tasks") or []
    if not tasks:
        return "review"  # nothing affordable to research; review decides what to do
    return [Send("researcher", task) for task in tasks]


def after_review(state: RunState) -> list[Send] | str:
    if state.get("failure"):
        return "finalize"
    decision = state["reviews"][-1].decision
    if decision == "send_back":
        return fan_out(state)
    if decision == "proceed":
        return SYNTHESIS_ENTRY
    return "finalize"


def after_check(state: RunState) -> str:
    if state.get("failure") or state.get("answer") is not None:
        return "finalize"
    return "writer" if state["answer_reviews"][-1].decision == "rewrite" else "finalize"


def _advocate_sends(state: RunState) -> list[Send]:
    tasks = state.get("pending_advocates") or []
    # The graph, not the prompt, bounds the debate: openings, then one rebuttal round, never more.
    assert all(t.round <= DEBATE_ROUNDS for t in tasks), "debate round limit exceeded"
    return [Send("advocate", t) for t in tasks]


def after_mapper(state: RunState) -> list[Send] | str:
    if state.get("failure"):
        return "finalize"
    if state.get("position_map") is None:
        return "writer"  # mapping skipped or failed: the writer works from findings
    return _advocate_sends(state) or "judge"


def after_debate_review(state: RunState) -> list[Send] | str:
    if state.get("failure"):
        return "finalize"
    if (state.get("debate_rounds") or 0) >= DEBATE_ROUNDS:
        return "judge"
    return _advocate_sends(state) or "judge"


def after_judge(state: RunState) -> str:
    return "finalize" if state.get("failure") else "writer"


def after_personalize(state: RunState) -> str:
    return "finalize" if state.get("failure") else "position_mapper"


def _wire_synthesis(graph: StateGraph, deps: GraphDeps) -> None:
    """Personalization, position mapping, the debate and the judge, between review and the writer."""
    nodes = SynthesisNodes(deps)
    graph.add_node("personalize", nodes.personalize)
    graph.add_node("position_mapper", nodes.position_mapper)
    graph.add_node("advocate", nodes.advocate)
    graph.add_node("debate_review", nodes.debate_review)
    graph.add_node("judge", nodes.judge)
    graph.add_conditional_edges("personalize", after_personalize, ["position_mapper", "finalize"])
    graph.add_conditional_edges("position_mapper", after_mapper, ["advocate", "judge", "writer", "finalize"])
    graph.add_edge("advocate", "debate_review")
    graph.add_conditional_edges("debate_review", after_debate_review, ["advocate", "judge", "finalize"])
    graph.add_conditional_edges("judge", after_judge, ["writer", "finalize"])


# --- Build ------------------------------------------------------------------


def build_graph(deps: GraphDeps, *, checkpointer: Any = None) -> Any:
    nodes = Nodes(deps)
    g = StateGraph(RunState)
    g.add_node("intake", nodes.intake)
    g.add_node("referral", nodes.referral)
    g.add_node("planner", nodes.planner)
    g.add_node("plan_checkpoint", nodes.plan_checkpoint)
    g.add_node("dispatch", nodes.dispatch)
    g.add_node("researcher", nodes.researcher)
    g.add_node("review", nodes.review)
    g.add_node("writer", nodes.writer)
    g.add_node("check_answer", nodes.check_answer)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", after_intake, ["planner", "referral", "finalize"])
    g.add_edge("referral", END)
    g.add_conditional_edges("planner", after_planner, ["plan_checkpoint", "dispatch", "finalize"])
    g.add_conditional_edges("plan_checkpoint", after_plan_checkpoint, ["dispatch", "finalize"])
    g.add_conditional_edges("dispatch", fan_out, ["researcher", "review", "finalize"])
    g.add_edge("researcher", "review")
    g.add_conditional_edges("review", after_review, ["researcher", SYNTHESIS_ENTRY, "finalize"])
    _wire_synthesis(g, deps)
    g.add_edge("writer", "check_answer")
    g.add_conditional_edges("check_answer", after_check, ["writer", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


# --- Run --------------------------------------------------------------------


class RunResult(Record):
    run_id: str
    # None only while the run waits at the plan checkpoint (`awaiting_approval` is set).
    outcome: RunOutcome | None
    awaiting_approval: PlanApprovalRequest | None = None
    answer: CoachAnswer | None = None
    intake: Intake | None = None
    plan: ResearchPlan | None = None
    sub_questions: list[SubQuestion] = []
    plan_decision: PlanDecision | None = None
    waited_s: float = 0.0
    findings: list[Finding] = []
    downweighted: dict[str, str] = {}
    position_map: PositionMap | None = None
    debate: list[DebateTurn] = []
    verdict: Verdict | None = None
    reviews: list[ReviewReport] = []
    answer_reviews: list[AnswerReview] = []
    spend: Spend = Spend()
    trace: list[TraceStep] = []

    @classmethod
    def from_state(
        cls, run_id: str, state: dict[str, Any], awaiting: PlanApprovalRequest | None = None
    ) -> "RunResult":
        outcome = state.get("outcome")
        if outcome is None and awaiting is None:
            outcome = RunOutcome(
                status="failed",
                message="run ended without an outcome",
                failure=RunFailure(reason="internal_error", node="graph", message="run ended without an outcome"),
            )
        return cls(
            run_id=run_id,
            outcome=outcome,
            awaiting_approval=awaiting,
            answer=state.get("answer"),
            intake=state.get("intake"),
            plan=state.get("plan"),
            sub_questions=state.get("sub_questions") or [],
            plan_decision=state.get("plan_decision"),
            waited_s=state.get("waited_s") or 0.0,
            findings=all_findings(state),  # type: ignore[arg-type]
            downweighted=state.get("downweighted") or {},
            position_map=state.get("position_map"),
            debate=state.get("debate") or [],
            verdict=state.get("verdict"),
            reviews=state.get("reviews") or [],
            answer_reviews=state.get("answer_reviews") or [],
            spend=state.get("spend") or Spend(),
            trace=state.get("trace") or [],
        )


def initial_state(
    question: str,
    profile: UserProfile | None,
    budget: Budget,
    run_id: str,
    started_at: float,
    require_plan_approval: bool = False,
) -> RunState:
    return {
        "run_id": run_id,
        "question": question,
        "profile": profile,
        "budget": budget,
        "started_at": started_at,
        "require_plan_approval": require_plan_approval,
        "plan_ready_at": None,
        "waited_s": 0.0,
        "plan_decision": None,
        "send_backs_used": 0,
        "writer_attempts": 0,
        "downweighted": {},
        "position_map": None,
        "verdict": None,
        "debate_rounds": 0,
        "failure": None,
    }


def run_research(
    question: str,
    profile: UserProfile | None = None,
    *,
    deps: GraphDeps,
    budget: Budget | None = None,
    run_id: str | None = None,
    checkpointer: Any = None,
    require_plan_approval: bool = False,
) -> RunResult:
    """Run one question through the graph. Never raises: failures come back in `outcome`.

    With `require_plan_approval` (needs a checkpointer) the result comes back with
    `awaiting_approval` set; continue with `resume_research(..., decision=...)`.
    For runs with stores and progress events use `graph.runner.Runner`.
    """
    run_id = run_id or uuid.uuid4().hex
    state = initial_state(question, profile, budget or Budget(), run_id, deps.clock(), require_plan_approval)
    return _invoke(build_graph(deps, checkpointer=checkpointer), run_id, state, checkpointer)


def resume_research(
    run_id: str,
    *,
    deps: GraphDeps,
    checkpointer: Any,
    decision: PlanDecision | dict[str, Any] | None = None,
) -> RunResult:
    """Continue a run from its checkpoint: after a crash (no decision) or at the plan checkpoint."""
    graph = build_graph(deps, checkpointer=checkpointer)
    if decision is None:
        graph_input: Any = None
    else:
        graph_input = Command(resume=decision.model_dump(mode="json") if isinstance(decision, PlanDecision) else decision)
    return _invoke(graph, run_id, graph_input, checkpointer)


def _invoke(graph: Any, run_id: str, graph_input: Any, checkpointer: Any) -> RunResult:
    from backend.graph.runner import awaiting_request

    config = {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}
    durability = "sync" if checkpointer is not None else "exit"
    try:
        final = graph.invoke(graph_input, config, durability=durability)
    except Exception as e:  # wiring bug or checkpointer outage: fail cleanly with what we have
        log.exception("run %s crashed", run_id)
        partial: dict[str, Any] = dict(graph_input) if isinstance(graph_input, dict) else {}
        if checkpointer is not None:
            try:
                partial = dict(graph.get_state(config).values) or partial
            except Exception:
                pass
        failure = RunFailure(reason="internal_error", node="graph", message=f"{type(e).__name__}: {e}"[:500])
        partial["outcome"] = RunOutcome(
            status="failed",
            message=failure.message,
            failure=failure,
            degradations=partial.get("degradations") or [],
        )
        return RunResult.from_state(run_id, partial)
    if checkpointer is not None:
        snapshot = graph.get_state(config)
        return RunResult.from_state(run_id, dict(snapshot.values), awaiting_request(snapshot))
    return RunResult.from_state(run_id, final)

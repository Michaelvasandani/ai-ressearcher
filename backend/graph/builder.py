"""The Evidence Coach state graph.

    START -> intake --out of scope--> referral -> END
               |
               v
            planner -> dispatch ==Send x N==> researcher -> review --send back (once)==> researcher
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

Extension points:
- Phase 5 (plan checkpoint, persistence): the plan-approval interrupt sits on
  the planner -> dispatch edge (`dispatch` reads `sub_questions`, so an edited
  plan only has to replace that field). The checkpointer is a parameter; use
  `checkpoint.memory_checkpointer()` locally or a Postgres saver built with
  `checkpoint.serializer()`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.graph.budget import Budget, Spend
from backend.graph.nodes import GraphDeps, Nodes
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
    g.add_node("dispatch", nodes.dispatch)
    g.add_node("researcher", nodes.researcher)
    g.add_node("review", nodes.review)
    g.add_node("writer", nodes.writer)
    g.add_node("check_answer", nodes.check_answer)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", after_intake, ["planner", "referral", "finalize"])
    g.add_edge("referral", END)
    g.add_conditional_edges("planner", after_planner, ["dispatch", "finalize"])
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
    outcome: RunOutcome
    answer: CoachAnswer | None = None
    intake: Intake | None = None
    plan: ResearchPlan | None = None
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
    def from_state(cls, run_id: str, state: dict[str, Any]) -> "RunResult":
        outcome = state.get("outcome") or RunOutcome(
            status="failed",
            message="run ended without an outcome",
            failure=RunFailure(reason="internal_error", node="graph", message="run ended without an outcome"),
        )
        return cls(
            run_id=run_id,
            outcome=outcome,
            answer=state.get("answer"),
            intake=state.get("intake"),
            plan=state.get("plan"),
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
) -> RunState:
    return {
        "run_id": run_id,
        "question": question,
        "profile": profile,
        "budget": budget,
        "started_at": started_at,
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
) -> RunResult:
    """Run one question through the graph. Never raises: failures come back in `outcome`."""
    run_id = run_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}
    graph = build_graph(deps, checkpointer=checkpointer)
    state = initial_state(question, profile, budget or Budget(), run_id, deps.clock())
    try:
        final = graph.invoke(state, config)
    except Exception as e:  # wiring bug or checkpointer outage: fail cleanly with what we have
        log.exception("run %s crashed", run_id)
        partial: dict[str, Any] = dict(state)
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
    return RunResult.from_state(run_id, final)

"""Typed run state: the single source of truth for one run.

Everything in here is plain Pydantic or JSON types so a checkpointer can write
it after every node (Phase 5) and the API can serve it (Phase 6). Fields that
parallel researchers write have reducers; everything else is written by one
node at a time.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import Field

from backend.citations import CitationReport
from backend.graph.budget import Allowance, Budget, BudgetKind, Spend, add_spend
from backend.graph.plan_review import PlanDecision
from backend.graph.routing import Route
from backend.graph.trace import TraceStep, append_steps
from backend.schemas import (
    CoachAnswer,
    DebateTurn,
    Finding,
    Intake,
    Position,
    PositionMap,
    Record,
    ResearchPlan,
    SearchStatus,
    SubQuestion,
    UserProfile,
    Verdict,
)

MAX_SEND_BACKS = 1  # the supervisor may send work back exactly once per run
DEBATE_ROUNDS = 2  # opening arguments, then exactly one rebuttal round


# --- Research ---------------------------------------------------------------


class ResearchTask(Record):
    """Payload of one researcher Send: everything the pass needs, nothing shared."""

    run_id: str
    sub_question: SubQuestion  # search_terms may be refined by the supervisor on a send-back
    pass_no: int = 1
    route: Route
    max_searches: int
    allowance: Allowance
    profile: UserProfile | None = None
    reason: str = "planned"


class TopicReuse(Record):
    """A researcher pass served from the topic findings store instead of searching."""

    topic_key: str
    matched_question: str
    source_run_id: str
    similarity: float
    age_s: float
    findings: int


class PassRecord(Record):
    pass_no: int
    model: str
    escalated: bool
    queries: list[str]
    failed_queries: list[str] = Field(default_factory=list)  # transient search failures, retryable
    status: SearchStatus
    finding_ids: list[str]
    budget_exhausted: BudgetKind | None = None
    reused: TopicReuse | None = None


class ResearchRecord(Record):
    """Everything known about one sub-question, across passes."""

    sub_question_id: str
    findings: list[Finding] = Field(default_factory=list)
    passes: list[PassRecord] = Field(default_factory=list)
    searches_used: int = 0
    queries_run: list[str] = Field(default_factory=list)

    @property
    def last_status(self) -> SearchStatus | None:
        return self.passes[-1].status if self.passes else None


def merge_research(
    left: dict[str, ResearchRecord] | None, right: dict[str, ResearchRecord] | None
) -> dict[str, ResearchRecord]:
    """State reducer: a new pass adds to a sub-question's record; findings dedupe by id (newest wins)."""
    out = dict(left or {})
    for sq_id, new in (right or {}).items():
        old = out.get(sq_id)
        if old is None:
            out[sq_id] = new
            continue
        by_id = {f.id: f for f in old.findings}
        by_id.update({f.id: f for f in new.findings})
        out[sq_id] = ResearchRecord(
            sub_question_id=sq_id,
            findings=list(by_id.values()),
            passes=old.passes + new.passes,
            searches_used=old.searches_used + new.searches_used,
            queries_run=old.queries_run + new.queries_run,
        )
    return out


# --- Debate -----------------------------------------------------------------


class AdvocateTask(Record):
    """Payload of one advocate Send: one position, one round, its own allowance."""

    run_id: str
    round: Literal[1, 2]
    position_no: int  # 1-based index into PositionMap.positions
    position: Position
    intake: Intake
    findings: list[Finding]
    downweighted: dict[str, str] = Field(default_factory=dict)
    opponents: list[DebateTurn] = Field(default_factory=list)  # round 2: the other sides' openings
    route: Route
    allowance: Allowance


# --- Supervisor -------------------------------------------------------------

DegradationKind = Literal[
    "max_sub_questions",
    "max_searches",
    "max_tokens",
    "max_usd",
    "wall_clock",
    "sub_question_unanswered",
    "send_back_unavailable",
    "population_mismatch",
    "abstract_only_evidence",
    "provenance_rejected",
    "mapping_skipped",  # no position map: the writer works from findings alone
    "debate_skipped",  # positions were not debated (budget)
    "rebuttal_skipped",  # openings only: the rebuttal round could not run
    "advocate_failed",  # an advocate produced no valid argument
    "judge_skipped",  # no verdict: the writer works from the position map
]


class Degradation(Record):
    """Something the run gave up on to stay within budget or because research came back thin."""

    kind: DegradationKind
    node: str
    detail: str
    sub_question_id: str | None = None


IssueCode = Literal[
    "not_researched",
    "no_findings",
    "only_animal_or_in_vitro",
    "population_mismatch",
    "search_failed",
    "extraction_failed",
    "abstract_only",  # no open-access full text: findings rest on abstracts (paywalled)
    "retracted_only",  # everything relevant that came back was retracted
]


class SubQuestionReview(Record):
    sub_question_id: str
    answered: bool  # at least one human (non animal / in-vitro) finding
    findings: int
    human_findings: int
    issues: list[IssueCode] = Field(default_factory=list)
    search_state: str | None = None
    population_mismatches: dict[str, str] = Field(default_factory=dict)  # finding id -> reason
    another_pass: bool = False
    another_pass_reason: str
    refined_queries: list[str] = Field(default_factory=list)
    escalate: bool = False
    # Why a pass that would help was not sent: a budget kind, or one of these.
    blocked_by: BudgetKind | Literal["send_back_used", "no_untried_query"] | None = None


class ReviewReport(Record):
    pass_no: int
    decision: Literal["proceed", "send_back", "fail"]
    reason: str
    sub_questions: list[SubQuestionReview]
    send_back_ids: list[str] = Field(default_factory=list)


class AnswerReview(Record):
    attempt: int
    decision: Literal["accept", "rewrite", "fail"]
    problems: list[str] = Field(default_factory=list)
    reason: str


# --- Outcome ----------------------------------------------------------------

FailureReason = Literal[
    "budget_exhausted",
    "insufficient_evidence",
    "intake_failed",
    "planner_failed",
    "writer_failed",
    "citations_failed",  # the answer's claims / numbers are not backed by its cited findings
    "internal_error",
]


class RunFailure(Record):
    reason: FailureReason
    node: str
    message: str
    budget_kind: BudgetKind | None = None


class RunOutcome(Record):
    status: Literal["answered", "degraded", "referred", "failed"]
    message: str
    failure: RunFailure | None = None
    degradations: list[Degradation] = Field(default_factory=list)
    referral_message: str | None = None


# --- Graph state ------------------------------------------------------------


class RunState(TypedDict, total=False):
    # Inputs
    run_id: str
    question: str
    profile: UserProfile | None
    budget: Budget
    # Epoch seconds the wall-clock budget counts from. The plan checkpoint moves it forward by the
    # time spent waiting for the user (`waited_s`), so a run can wait for hours without timing out.
    started_at: float
    require_plan_approval: bool  # pause at the plan checkpoint (False: auto-approve, e.g. evals and tests)
    plan_ready_at: float | None  # when the planner finished: the wait for approval starts here
    waited_s: float
    plan_decision: PlanDecision | None

    # Specialist outputs
    intake: Intake
    plan: ResearchPlan
    sub_questions: list[SubQuestion]  # the plan after the max_sub_questions cap
    research: Annotated[dict[str, ResearchRecord], merge_research]
    downweighted: dict[str, str]  # finding id -> why its population differs from the user
    position_map: PositionMap | None
    debate: Annotated[list[DebateTurn], operator.add]
    verdict: Verdict | None
    draft_answer: CoachAnswer | None
    answer: CoachAnswer | None

    # Supervisor bookkeeping
    pending_tasks: list[ResearchTask]  # next researcher fan-out
    pending_advocates: list[AdvocateTask]  # next advocate fan-out (one debate round)
    debate_rounds: int  # rounds run so far; never more than DEBATE_ROUNDS
    reviews: Annotated[list[ReviewReport], operator.add]
    answer_reviews: Annotated[list[AnswerReview], operator.add]
    citation_reports: Annotated[list[CitationReport], operator.add]  # one per writer draft
    send_backs_used: int
    writer_attempts: int
    writer_route: Route | None  # set by the supervisor for an escalated rewrite
    writer_feedback: str | None
    writer_error: str | None
    writer_budget_kind: BudgetKind | None
    degradations: Annotated[list[Degradation], operator.add]
    failure: RunFailure | None
    outcome: RunOutcome

    # Accounting
    spend: Annotated[Spend, add_spend]
    trace: Annotated[list[TraceStep], append_steps]


def all_findings(state: RunState) -> list[Finding]:
    """Findings for the selected sub-questions, in plan order.

    Provenance gate (defense in depth; the researcher node already rejects
    them): a finding without complete provenance never reaches the writer or
    the answer review.
    """
    research = state.get("research") or {}
    out: list[Finding] = []
    for sq in state.get("sub_questions") or []:
        rec = research.get(sq.id)
        if rec:
            out.extend(f for f in rec.findings if f.provenance_problem() is None)
    return out

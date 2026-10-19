"""Request and response bodies of the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.citations import CitationReport
from backend.graph.budget import Budget, Spend
from backend.graph.fork import AnswerDiff, ForkInfo
from backend.graph.plan_review import PlanApprovalRequest
from backend.graph.state import RunOutcome
from backend.graph.trace import TraceStep
from backend.schemas import CoachAnswer, SubQuestion, UserProfile
from backend.store.models import Conversation, RunStatus


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchRequest(Body):
    question: str = Field(min_length=3, max_length=2_000)
    profile: UserProfile | None = None  # default: the caller's saved profile (PUT /profile)
    conversation_id: str | None = None  # default: a new conversation titled after the question
    budget: Budget | None = None
    require_plan_approval: bool = False


class ResearchCreated(BaseModel):
    run_id: str
    conversation_id: str | None
    status: RunStatus
    parent_run_id: str | None = None
    links: dict[str, str]


class RunView(BaseModel):
    """GET /research/{id}: status first; the answer and outcome once completed."""

    run_id: str
    status: RunStatus
    question: str
    conversation_id: str | None = None
    created_at: datetime
    updated_at: datetime
    plan: PlanApprovalRequest | None = None  # set while awaiting_approval
    plan_decision_pending: bool = False  # a decision was accepted and is being applied
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    answer: CoachAnswer | None = None
    outcome: RunOutcome | None = None
    citations: CitationReport | None = None  # the citation gate's report on the final draft
    spend: Spend = Field(default_factory=Spend)
    steps: int = 0
    error: str | None = None
    parent_run_id: str | None = None
    fork: ForkInfo | None = None
    diff: AnswerDiff | None = None  # forks, once completed: against the parent's answer
    links: dict[str, str] = Field(default_factory=dict)


class TraceView(BaseModel):
    run_id: str
    status: RunStatus
    steps: list[TraceStep]
    inherited_steps: int = 0  # forks: steps copied from the parent run (their run_id is the parent's)
    spend: Spend = Field(default_factory=Spend)
    totals: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    run_id: str
    question: str
    status: RunStatus
    outcome_status: str | None = None
    cost_usd: float = 0.0
    parent_run_id: str | None = None
    created_at: datetime


class ConversationView(BaseModel):
    conversation: Conversation
    runs: list[RunSummary]


class ProfileView(BaseModel):
    user_id: str
    profile: UserProfile

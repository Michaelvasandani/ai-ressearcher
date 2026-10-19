"""Rows of the store tables, as Pydantic models (shared by every implementation)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from backend.llm import CallRecord
from backend.schemas import Finding, Record, UserProfile, utcnow

# running: the graph is executing (or the process died mid-run: resume it);
# awaiting_approval: paused at the plan checkpoint; completed: `outcome` is final;
# error: an exception escaped the graph (e.g. a database outage); resumable.
RunStatus = Literal["running", "awaiting_approval", "completed", "error"]


class RunRecord(Record):
    run_id: str
    question: str
    profile: UserProfile | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    require_plan_approval: bool = False
    status: RunStatus = "running"
    outcome_status: str | None = None  # RunOutcome.status once completed
    outcome: dict[str, Any] | None = None  # RunOutcome as JSON
    cost_usd: float = 0.0
    error: str | None = None
    budget: dict[str, Any] | None = None  # Budget as JSON: the worker starts the graph from the record
    # Execution (Phase 6): one worker at a time holds the lease; an expired lease means the worker
    # died (or hit the platform time limit) and the reaper may hand the run to another one.
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    attempts: int = 0  # advances that ended in an error (bounded automatic retries)
    pending_decision: dict[str, Any] | None = None  # a plan decision waiting for the worker
    # Time-travel: a fork records its parent and what it changed (graph.fork.ForkInfo as JSON).
    parent_run_id: str | None = None
    fork: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


EventType = Literal[
    "run_started", "run_resumed", "step", "plan_ready", "plan_decided", "run_finished", "run_error", "progress"
]


class RunEvent(Record):
    """One progress event. `seq` is gap-free and increasing per run, so a poller asks for
    `seq > last_seen`. `key` makes appends idempotent (a replayed step is not logged twice)."""

    run_id: str
    seq: int
    type: EventType
    key: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class TopicFindings(Record):
    """Findings for one topic, reusable by later runs while fresh."""

    topic_key: str
    tokens: list[str]
    question: str
    sub_question_id: str
    findings: list[Finding]
    run_id: str
    created_at: datetime = Field(default_factory=utcnow)


class TopicHit(Record):
    entry: TopicFindings
    similarity: float
    age_s: float


class LedgerEntry(Record):
    """A completed LLM call, keyed by (run_id, call_key). A retried step that makes the
    same request gets this result back instead of paying for the call again."""

    run_id: str
    call_key: str
    record: CallRecord
    output_json: str
    created_at: datetime = Field(default_factory=utcnow)


class Conversation(Record):
    id: str
    user_id: str | None = None
    title: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

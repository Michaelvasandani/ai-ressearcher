"""Storage interfaces. `memory.py` and `postgres.py` implement them.

The graph only needs `CallLedger` and `FindingsStore` (through `GraphDeps`);
the runner adds `RunStore` and `EventStore`; the API (Phase 6) adds
`ConversationStore` and `ProfileStore`. The LangGraph checkpointer is separate
(it has its own interface) but lives in the same database.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from backend.llm import LLMRequest
from backend.schemas import UserProfile
from backend.store.models import (
    Conversation,
    EventType,
    LedgerEntry,
    RunEvent,
    RunRecord,
    TopicFindings,
    TopicHit,
)


class RunStore(Protocol):
    def create(self, run: RunRecord) -> RunRecord: ...

    def get(self, run_id: str) -> RunRecord | None: ...

    def update(self, run_id: str, **fields: Any) -> RunRecord | None: ...

    def list(
        self, *, user_id: str | None = None, conversation_id: str | None = None, limit: int = 50
    ) -> list[RunRecord]: ...


class EventStore(Protocol):
    def append(
        self, run_id: str, type: EventType, data: dict[str, Any], *, key: str | None = None
    ) -> RunEvent | None:
        """Append an event; returns None if an event with this `key` already exists for the run."""
        ...

    def list(self, run_id: str, *, after_seq: int = 0, limit: int = 200) -> list[RunEvent]: ...


class FindingsStore(Protocol):
    def save(self, entry: TopicFindings) -> None:
        """Upsert by topic key: the newest findings for a topic replace older ones."""
        ...

    def lookup(
        self, question: str, *, now: datetime, max_age_s: float, min_similarity: float
    ) -> TopicHit | None:
        """The most similar fresh entry, or None."""
        ...


class CallLedger(Protocol):
    def get(self, run_id: str, call_key: str) -> LedgerEntry | None: ...

    def put(self, entry: LedgerEntry) -> bool:
        """Insert unless (run_id, call_key) exists; True if inserted."""
        ...

    def list(self, run_id: str) -> list[LedgerEntry]: ...


class ConversationStore(Protocol):
    def create(self, conversation: Conversation) -> Conversation: ...

    def get(self, conversation_id: str) -> Conversation | None: ...

    def list(self, user_id: str | None, *, limit: int = 50) -> list[Conversation]: ...

    def touch(self, conversation_id: str, *, title: str | None = None) -> None: ...

    def delete(self, conversation_id: str) -> None: ...


class ProfileStore(Protocol):
    def get(self, user_id: str) -> UserProfile | None: ...

    def put(self, user_id: str, profile: UserProfile) -> None: ...


@dataclass
class Stores:
    """Everything persistent, built together by `store.factory.build_stores`."""

    runs: RunStore
    events: EventStore
    findings: FindingsStore
    ledger: CallLedger
    conversations: ConversationStore
    profiles: ProfileStore
    checkpointer: Any  # a LangGraph BaseCheckpointSaver
    close: Any = field(default=lambda: None, repr=False)  # releases the connection pool


def call_key(request: LLMRequest[Any]) -> str:
    """Stable key for an LLM request within a run.

    Hashes what determines the answer (agent, model, output schema, prompts) but not
    `max_tokens`, which budget metering may lower. Identical requests in one run share
    one ledger entry, which is what makes a retried step free.
    """
    payload = json.dumps(
        [request.agent, request.model, request.output_type.__name__, request.effort, request.system, request.user],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()

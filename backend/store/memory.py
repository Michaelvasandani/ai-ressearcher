"""In-memory stores for tests and local runs. Thread-safe; nothing survives the process."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from backend.schemas import UserProfile, utcnow
from backend.store.base import Stores
from backend.store.models import (
    Conversation,
    EventType,
    LedgerEntry,
    RunEvent,
    RunRecord,
    TopicFindings,
    TopicHit,
)
from backend.store.topics import similarity, topic_tokens


class MemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create(self, run: RunRecord) -> RunRecord:
        with self._lock:
            if run.run_id in self._runs:
                raise ValueError(f"run {run.run_id} exists")
            self._runs[run.run_id] = run
            return run

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def update(self, run_id: str, **fields: Any) -> RunRecord | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            run = run.model_copy(update={**fields, "updated_at": utcnow()})
            self._runs[run_id] = run
            return run

    def list(
        self, *, user_id: str | None = None, conversation_id: str | None = None, limit: int = 50
    ) -> list[RunRecord]:
        runs = [
            r
            for r in self._runs.values()
            if (user_id is None or r.user_id == user_id)
            and (conversation_id is None or r.conversation_id == conversation_id)
        ]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)[:limit]


class MemoryEventStore:
    def __init__(self) -> None:
        self._events: dict[str, list[RunEvent]] = {}
        self._lock = threading.Lock()

    def append(
        self, run_id: str, type: EventType, data: dict[str, Any], *, key: str | None = None
    ) -> RunEvent | None:
        with self._lock:
            events = self._events.setdefault(run_id, [])
            if key is not None and any(e.key == key for e in events):
                return None
            event = RunEvent(run_id=run_id, seq=len(events) + 1, type=type, key=key, data=data)
            events.append(event)
            return event

    def list(self, run_id: str, *, after_seq: int = 0, limit: int = 200) -> list[RunEvent]:
        return [e for e in self._events.get(run_id, []) if e.seq > after_seq][:limit]


class MemoryFindingsStore:
    def __init__(self) -> None:
        self._entries: dict[str, TopicFindings] = {}
        self._lock = threading.Lock()

    def save(self, entry: TopicFindings) -> None:
        with self._lock:
            self._entries[entry.topic_key] = entry

    def lookup(
        self, question: str, *, now: datetime, max_age_s: float, min_similarity: float
    ) -> TopicHit | None:
        tokens = topic_tokens(question)
        best: TopicHit | None = None
        for entry in list(self._entries.values()):
            age = (now - entry.created_at).total_seconds()
            if age > max_age_s:
                continue
            sim = similarity(tokens, entry.tokens)
            if sim >= min_similarity and (best is None or (sim, -age) > (best.similarity, -best.age_s)):
                best = TopicHit(entry=entry, similarity=sim, age_s=max(age, 0.0))
        return best


class MemoryCallLedger:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], LedgerEntry] = {}
        self._lock = threading.Lock()

    def get(self, run_id: str, call_key: str) -> LedgerEntry | None:
        return self._entries.get((run_id, call_key))

    def put(self, entry: LedgerEntry) -> bool:
        with self._lock:
            k = (entry.run_id, entry.call_key)
            if k in self._entries:
                return False
            self._entries[k] = entry
            return True

    def list(self, run_id: str) -> list[LedgerEntry]:
        return sorted((e for (r, _), e in self._entries.items() if r == run_id), key=lambda e: e.created_at)


class MemoryConversationStore:
    def __init__(self) -> None:
        self._items: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def create(self, conversation: Conversation) -> Conversation:
        with self._lock:
            self._items[conversation.id] = conversation
            return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        return self._items.get(conversation_id)

    def list(self, user_id: str | None, *, limit: int = 50) -> list[Conversation]:
        items = [c for c in self._items.values() if c.user_id == user_id]
        return sorted(items, key=lambda c: c.updated_at, reverse=True)[:limit]

    def touch(self, conversation_id: str, *, title: str | None = None) -> None:
        with self._lock:
            c = self._items.get(conversation_id)
            if c is not None:
                update: dict[str, Any] = {"updated_at": utcnow()}
                if title is not None:
                    update["title"] = title
                self._items[conversation_id] = c.model_copy(update=update)

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._items.pop(conversation_id, None)


class MemoryProfileStore:
    def __init__(self) -> None:
        self._items: dict[str, UserProfile] = {}

    def get(self, user_id: str) -> UserProfile | None:
        return self._items.get(user_id)

    def put(self, user_id: str, profile: UserProfile) -> None:
        self._items[user_id] = profile


def memory_stores() -> Stores:
    from backend.graph.checkpoint import memory_checkpointer

    return Stores(
        runs=MemoryRunStore(),
        events=MemoryEventStore(),
        findings=MemoryFindingsStore(),
        ledger=MemoryCallLedger(),
        conversations=MemoryConversationStore(),
        profiles=MemoryProfileStore(),
        checkpointer=memory_checkpointer(),
    )

"""Postgres (Supabase) stores and the LangGraph Postgres checkpointer.

One psycopg connection pool serves the checkpointer and every store.
Connections are opened with:

- `autocommit=True`: every statement is its own transaction, which is what
  Supabase's transaction pooler (Supavisor / pgbouncer, port 6543) expects;
  multi-statement writes use an explicit `conn.transaction()`.
- `prepare_threshold=None`: no server-side prepared statements. A transaction
  pooler hands each transaction to a different server connection, where a
  statement prepared on another one does not exist ("prepared statement
  _pg3_0 does not exist"). LangGraph's own `from_conn_string` uses
  `prepare_threshold=0` (always prepare), so we build the saver ourselves.
- `check=ConnectionPool.check_connection`: a pooled connection is checked
  before use, since serverless instances sit idle and poolers close idle
  connections.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import TypeAdapter

from backend.llm import CallRecord
from backend.schemas import Finding, UserProfile
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

log = logging.getLogger(__name__)

_FINDINGS = TypeAdapter(list[Finding])
MAX_TOPIC_CANDIDATES = 50


def connection_kwargs() -> dict[str, Any]:
    return {"autocommit": True, "prepare_threshold": None, "row_factory": dict_row}


def connect(database_url: str, *, min_size: int | None = None, max_size: int | None = None) -> ConnectionPool:
    """A connection pool sized for serverless: small, and lazy beyond the first connection."""
    pool = ConnectionPool(
        database_url,
        min_size=min_size if min_size is not None else int(os.environ.get("DATABASE_POOL_MIN", "1")),
        max_size=max_size if max_size is not None else int(os.environ.get("DATABASE_POOL_MAX", "5")),
        kwargs=connection_kwargs(),
        check=ConnectionPool.check_connection,
        open=False,
        name="evidence-coach",
    )
    pool.open(wait=True, timeout=float(os.environ.get("DATABASE_CONNECT_TIMEOUT_S", "15")))
    return pool


def postgres_checkpointer(pool: ConnectionPool) -> Any:
    """LangGraph's PostgresSaver on our pool, with the run-state serializer allowlist."""
    from langgraph.checkpoint.postgres import PostgresSaver

    from backend.graph.checkpoint import serializer

    return PostgresSaver(pool, serde=serializer())  # type: ignore[arg-type]


class _PG:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def _one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            return conn.execute(sql, tuple(params)).fetchone()  # type: ignore[return-value]

    def _all(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return conn.execute(sql, tuple(params)).fetchall()  # type: ignore[return-value]

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self.pool.connection() as conn:
            conn.execute(sql, tuple(params))


# --- Runs ---------------------------------------------------------------------

_RUN_COLUMNS = (
    "run_id, user_id, conversation_id, question, profile, require_plan_approval, status, "
    "outcome_status, outcome, cost_usd, error, budget, lease_owner, lease_expires_at, attempts, "
    "pending_decision, parent_run_id, fork, created_at, updated_at"
)
_RUN_UPDATABLE = frozenset(
    {
        "status", "outcome_status", "outcome", "cost_usd", "error", "conversation_id", "profile", "user_id",
        "budget", "lease_owner", "lease_expires_at", "attempts", "pending_decision", "fork",
    }
)
_RUN_JSON = frozenset({"profile", "outcome", "budget", "pending_decision", "fork"})


def _run(row: dict[str, Any]) -> RunRecord:
    return RunRecord.model_validate(row)


class PostgresRunStore(_PG):
    def create(self, run: RunRecord) -> RunRecord:
        def js(v: Any) -> Any:
            return Jsonb(v) if v is not None else None

        row = self._one(
            f"""insert into runs ({_RUN_COLUMNS})
            values ({", ".join(["%s"] * 20)}) returning {_RUN_COLUMNS}""",
            (
                run.run_id, run.user_id, run.conversation_id, run.question,
                js(run.profile.model_dump(mode="json") if run.profile else None),
                run.require_plan_approval, run.status, run.outcome_status, js(run.outcome),
                run.cost_usd, run.error, js(run.budget), run.lease_owner, run.lease_expires_at, run.attempts,
                js(run.pending_decision), run.parent_run_id, js(run.fork), run.created_at, run.updated_at,
            ),
        )
        assert row is not None
        return _run(row)

    @staticmethod
    def _sets(fields: dict[str, Any]) -> tuple[list[str], list[Any]]:
        unknown = set(fields) - _RUN_UPDATABLE
        if unknown:
            raise ValueError(f"cannot update run columns: {sorted(unknown)}")
        sets, params = [], []
        for name, value in fields.items():
            if name == "profile" and isinstance(value, UserProfile):
                value = value.model_dump(mode="json")
            if name in _RUN_JSON and value is not None:
                value = Jsonb(value)
            sets.append(f"{name} = %s")
            params.append(value)
        sets.append("updated_at = now()")
        return sets, params

    def get(self, run_id: str) -> RunRecord | None:
        row = self._one(f"select {_RUN_COLUMNS} from runs where run_id = %s", (run_id,))
        return _run(row) if row else None

    def update(self, run_id: str, **fields: Any) -> RunRecord | None:
        sets, params = self._sets(fields)
        row = self._one(
            f"update runs set {', '.join(sets)} where run_id = %s returning {_RUN_COLUMNS}", (*params, run_id)
        )
        return _run(row) if row else None

    def transition(self, run_id: str, from_status: tuple[str, ...], **fields: Any) -> RunRecord | None:
        sets, params = self._sets(fields)
        row = self._one(
            f"update runs set {', '.join(sets)} where run_id = %s and status = any(%s) returning {_RUN_COLUMNS}",
            (*params, run_id, list(from_status)),
        )
        return _run(row) if row else None

    # Leases: one conditional UPDATE each. Under READ COMMITTED a concurrent UPDATE of the same
    # row waits for the first to commit and then re-checks the WHERE clause, so two workers
    # racing for a free lease cannot both win. Expiry uses the database clock.

    def acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> RunRecord | None:
        row = self._one(
            f"""update runs set lease_owner = %s, lease_expires_at = now() + make_interval(secs => %s)
            where run_id = %s and status = 'running'
              and (lease_expires_at is null or lease_expires_at < now() or lease_owner = %s)
            returning {_RUN_COLUMNS}""",
            (owner, ttl_s, run_id, owner),
        )
        return _run(row) if row else None

    def renew_lease(self, run_id: str, owner: str, ttl_s: float) -> bool:
        row = self._one(
            """update runs set lease_expires_at = now() + make_interval(secs => %s)
            where run_id = %s and lease_owner = %s returning run_id""",
            (ttl_s, run_id, owner),
        )
        return row is not None

    def release_lease(self, run_id: str, owner: str) -> bool:
        row = self._one(
            """update runs set lease_owner = null, lease_expires_at = null, updated_at = now()
            where run_id = %s and lease_owner = %s returning run_id""",
            (run_id, owner),
        )
        return row is not None

    def reapable(self, *, stale_after_s: float, max_attempts: int, limit: int = 20) -> list[RunRecord]:
        rows = self._all(
            f"""select {_RUN_COLUMNS} from runs
            where (status = 'running' and (
                    lease_expires_at < now()
                    or (lease_expires_at is null and updated_at < now() - make_interval(secs => %s))))
               or (status = 'error' and attempts < %s and updated_at < now() - make_interval(secs => %s))
            order by updated_at limit %s""",
            (stale_after_s, max_attempts, stale_after_s, limit),
        )
        return [_run(r) for r in rows]

    def list(
        self, *, user_id: str | None = None, conversation_id: str | None = None, limit: int = 50
    ) -> list[RunRecord]:
        where, params = [], []
        if user_id is not None:
            where.append("user_id = %s")
            params.append(user_id)
        if conversation_id is not None:
            where.append("conversation_id = %s")
            params.append(conversation_id)
        clause = f"where {' and '.join(where)}" if where else ""
        rows = self._all(
            f"select {_RUN_COLUMNS} from runs {clause} order by created_at desc limit %s", (*params, limit)
        )
        return [_run(r) for r in rows]


# --- Events -------------------------------------------------------------------


class PostgresEventStore(_PG):
    MAX_ATTEMPTS = 5

    def append(
        self, run_id: str, type: EventType, data: dict[str, Any], *, key: str | None = None
    ) -> RunEvent | None:
        # seq = max + 1 per run. A concurrent append can take the same seq; the primary key
        # rejects it (do nothing) and we try again. A duplicate `key` means the event exists.
        for _ in range(self.MAX_ATTEMPTS):
            row = self._one(
                """insert into run_events (run_id, seq, type, key, data)
                select %s, coalesce(max(seq), 0) + 1, %s, %s, %s from run_events where run_id = %s
                on conflict do nothing
                returning run_id, seq, type, key, data, created_at""",
                (run_id, type, key, Jsonb(data), run_id),
            )
            if row is not None:
                return RunEvent.model_validate(row)
            if key is not None and self._one(
                "select 1 from run_events where run_id = %s and key = %s", (run_id, key)
            ):
                return None
        raise RuntimeError(f"could not append event to run {run_id}")

    def list(self, run_id: str, *, after_seq: int = 0, limit: int = 200) -> list[RunEvent]:
        rows = self._all(
            """select run_id, seq, type, key, data, created_at from run_events
            where run_id = %s and seq > %s order by seq limit %s""",
            (run_id, after_seq, limit),
        )
        return [RunEvent.model_validate(r) for r in rows]


# --- Topic findings -----------------------------------------------------------


class PostgresFindingsStore(_PG):
    def save(self, entry: TopicFindings) -> None:
        self._exec(
            """insert into topic_findings (topic_key, tokens, question, sub_question_id, findings, run_id, created_at)
            values (%s, %s, %s, %s, %s, %s, %s)
            on conflict (topic_key) do update set
                tokens = excluded.tokens, question = excluded.question,
                sub_question_id = excluded.sub_question_id, findings = excluded.findings,
                run_id = excluded.run_id, created_at = excluded.created_at""",
            (
                entry.topic_key, entry.tokens, entry.question, entry.sub_question_id,
                Jsonb([f.model_dump(mode="json") for f in entry.findings]), entry.run_id, entry.created_at,
            ),
        )

    def lookup(
        self, question: str, *, now: datetime, max_age_s: float, min_similarity: float
    ) -> TopicHit | None:
        tokens = topic_tokens(question)
        if not tokens:
            return None
        # Narrow by keyword overlap on the GIN index, then score exactly as the memory store does.
        rows = self._all(
            """select topic_key, tokens, question, sub_question_id, findings, run_id, created_at
            from topic_findings where tokens && %s and created_at >= %s
            order by created_at desc limit %s""",
            (tokens, now - timedelta(seconds=max_age_s), MAX_TOPIC_CANDIDATES),
        )
        best: TopicHit | None = None
        for row in rows:
            sim = similarity(tokens, row["tokens"])
            if sim < min_similarity or (best is not None and sim <= best.similarity):
                continue
            entry = TopicFindings(
                topic_key=row["topic_key"], tokens=row["tokens"], question=row["question"],
                sub_question_id=row["sub_question_id"], findings=_FINDINGS.validate_python(row["findings"]),
                run_id=row["run_id"], created_at=row["created_at"],
            )
            best = TopicHit(entry=entry, similarity=sim, age_s=max((now - entry.created_at).total_seconds(), 0.0))
        return best


# --- LLM call ledger ----------------------------------------------------------


class PostgresCallLedger(_PG):
    def get(self, run_id: str, call_key: str) -> LedgerEntry | None:
        row = self._one(
            "select run_id, call_key, record, output_json, created_at from llm_calls where run_id = %s and call_key = %s",
            (run_id, call_key),
        )
        return LedgerEntry.model_validate(row) if row else None

    def put(self, entry: LedgerEntry) -> bool:
        rec: CallRecord = entry.record
        row = self._one(
            """insert into llm_calls (run_id, call_key, call_id, agent, model, cost_usd, record, output_json, created_at)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (run_id, call_key) do nothing returning call_key""",
            (
                entry.run_id, entry.call_key, rec.call_id, rec.agent, rec.model, rec.cost_usd,
                Jsonb(rec.model_dump(mode="json")), entry.output_json, entry.created_at,
            ),
        )
        return row is not None

    def list(self, run_id: str) -> list[LedgerEntry]:
        rows = self._all(
            "select run_id, call_key, record, output_json, created_at from llm_calls where run_id = %s order by created_at",
            (run_id,),
        )
        return [LedgerEntry.model_validate(r) for r in rows]


# --- Conversations and profiles -------------------------------------------------

_CONV_COLUMNS = "id, user_id, title, created_at, updated_at"


class PostgresConversationStore(_PG):
    def create(self, conversation: Conversation) -> Conversation:
        c = conversation
        row = self._one(
            f"insert into conversations ({_CONV_COLUMNS}) values (%s, %s, %s, %s, %s) returning {_CONV_COLUMNS}",
            (c.id, c.user_id, c.title, c.created_at, c.updated_at),
        )
        return Conversation.model_validate(row)

    def get(self, conversation_id: str) -> Conversation | None:
        row = self._one(f"select {_CONV_COLUMNS} from conversations where id = %s", (conversation_id,))
        return Conversation.model_validate(row) if row else None

    def list(self, user_id: str | None, *, limit: int = 50) -> list[Conversation]:
        rows = self._all(
            f"select {_CONV_COLUMNS} from conversations where user_id is not distinct from %s "
            "order by updated_at desc limit %s",
            (user_id, limit),
        )
        return [Conversation.model_validate(r) for r in rows]

    def touch(self, conversation_id: str, *, title: str | None = None) -> None:
        self._exec(
            "update conversations set updated_at = now(), title = coalesce(%s, title) where id = %s",
            (title, conversation_id),
        )

    def delete(self, conversation_id: str) -> None:
        self._exec("delete from conversations where id = %s", (conversation_id,))


class PostgresProfileStore(_PG):
    def get(self, user_id: str) -> UserProfile | None:
        row = self._one("select profile from profiles where user_id = %s", (user_id,))
        return UserProfile.model_validate(row["profile"]) if row else None

    def put(self, user_id: str, profile: UserProfile) -> None:
        self._exec(
            """insert into profiles (user_id, profile, updated_at) values (%s, %s, now())
            on conflict (user_id) do update set profile = excluded.profile, updated_at = now()""",
            (user_id, Jsonb(profile.model_dump(mode="json"))),
        )


def postgres_stores(database_url: str, *, pool: ConnectionPool | None = None) -> Stores:
    pool = pool or connect(database_url)
    return Stores(
        runs=PostgresRunStore(pool),
        events=PostgresEventStore(pool),
        findings=PostgresFindingsStore(pool),
        ledger=PostgresCallLedger(pool),
        conversations=PostgresConversationStore(pool),
        profiles=PostgresProfileStore(pool),
        checkpointer=postgres_checkpointer(pool),
        close=pool.close,
    )

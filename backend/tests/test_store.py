"""Store contract tests: every test runs on the in-memory stores, and on Postgres when
TEST_DATABASE_URL is set (`postgres` marker)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from backend.graph import Budget, GraphDeps, Runner
from backend.llm import CallRecord
from backend.schemas import UserProfile, utcnow
from backend.store import Conversation, LedgerEntry, RunRecord, TopicFindings
from backend.store.factory import build_stores, database_url
from backend.store.topics import similarity, topic_key, topic_tokens
from backend.tests.conftest import make_finding
from backend.tests.graph_fakes import fake_llm

VOLUME_Q = "Is there a dose-response relationship between weekly sets per muscle and hypertrophy in resistance-trained adults?"


# --- Topic keys (pure) -----------------------------------------------------------


def test_topic_key_normalizes_wording():
    assert topic_key("How many SETS per week for Hypertrophy?") == topic_key("how many set per week for hypertrophy")
    assert "the" not in topic_tokens("What is the effect of the protein?")
    assert topic_tokens("dose-response relationships") == ["dose", "relationship", "response"]


def test_similarity_matches_rewordings_not_other_topics():
    reworded = "Dose-response relationship between weekly sets per muscle and hypertrophy in resistance trained adults"
    other = "Does long-term creatine monohydrate supplementation harm renal health in healthy adults?"
    assert similarity(topic_tokens(VOLUME_Q), topic_tokens(reworded)) >= 0.9
    assert similarity(topic_tokens(VOLUME_Q), topic_tokens(other)) < 0.2


def test_build_stores_defaults_to_memory():
    assert database_url({}) is None
    stores = build_stores({})
    assert type(stores.runs).__name__ == "MemoryRunStore"


# --- Contract tests ----------------------------------------------------------------


def test_runs_create_update_list(open_stores):
    s = open_stores()
    conv = s.conversations.create(Conversation(id="c1", user_id="u1", title="Sets"))
    profile = UserProfile(sex="female", training_status="intermediate")
    s.runs.create(RunRecord(run_id="r1", question="q1", profile=profile, user_id="u1", conversation_id=conv.id))
    s.runs.create(RunRecord(run_id="r2", question="q2", user_id="u2"))
    with pytest.raises(Exception):
        s.runs.create(RunRecord(run_id="r1", question="dup"))

    got = s.runs.get("r1")
    assert got is not None and got.profile == profile and got.status == "running"
    upd = s.runs.update("r1", status="completed", outcome_status="answered", outcome={"status": "answered"}, cost_usd=0.012)
    assert upd is not None and upd.status == "completed" and upd.outcome == {"status": "answered"}
    assert upd.updated_at >= got.updated_at
    assert s.runs.update("missing", status="error") is None
    assert [r.run_id for r in s.runs.list(user_id="u1")] == ["r1"]
    assert [r.run_id for r in s.runs.list(conversation_id="c1")] == ["r1"]
    assert {r.run_id for r in s.runs.list()} == {"r1", "r2"}


def test_events_are_sequenced_and_idempotent_by_key(open_stores):
    s = open_stores()
    s.runs.create(RunRecord(run_id="r1", question="q"))
    e1 = s.events.append("r1", "run_started", {"q": 1}, key="run_started")
    e2 = s.events.append("r1", "step", {"node": "intake"}, key="step:a")
    assert s.events.append("r1", "step", {"node": "intake"}, key="step:a") is None  # replayed step
    e3 = s.events.append("r1", "run_resumed", {})  # no key: always appended
    assert (e1.seq, e2.seq, e3.seq) == (1, 2, 3)
    assert [e.seq for e in s.events.list("r1")] == [1, 2, 3]
    assert [e.type for e in s.events.list("r1", after_seq=1)] == ["step", "run_resumed"]
    assert s.events.list("r1", after_seq=1)[0].data == {"node": "intake"}
    assert s.events.list("other") == []


def test_findings_store_matches_similar_fresh_topics(open_stores):
    s = open_stores()
    now = utcnow()
    findings = [make_finding("sq1:FIXTURE-0001"), make_finding("sq1:FIXTURE-0002", study_type="rct", grade="moderate")]
    s.findings.save(
        TopicFindings(
            topic_key=topic_key(VOLUME_Q), tokens=topic_tokens(VOLUME_Q), question=VOLUME_Q,
            sub_question_id="sq1", findings=findings, run_id="r1", created_at=now - timedelta(hours=2),
        )
    )
    day = 86_400.0
    hit = s.findings.lookup(
        "Dose-response relationship between weekly sets per muscle and hypertrophy in resistance trained adults",
        now=now, max_age_s=day, min_similarity=0.5,
    )
    assert hit is not None and hit.entry.run_id == "r1" and hit.similarity > 0.9
    assert 7_100 < hit.age_s < 7_300
    assert [f.id for f in hit.entry.findings] == [f.id for f in findings]
    assert hit.entry.findings[0].provenance == findings[0].provenance
    # Too old, or a different topic: no hit.
    assert s.findings.lookup(VOLUME_Q, now=now, max_age_s=3_600, min_similarity=0.5) is None
    assert s.findings.lookup("creatine renal safety", now=now, max_age_s=day, min_similarity=0.5) is None

    # Same topic again: the newer findings replace the old ones.
    s.findings.save(
        TopicFindings(
            topic_key=topic_key(VOLUME_Q), tokens=topic_tokens(VOLUME_Q), question=VOLUME_Q,
            sub_question_id="sq2", findings=findings[:1], run_id="r2", created_at=now,
        )
    )
    hit = s.findings.lookup(VOLUME_Q, now=now, max_age_s=day, min_similarity=0.5)
    assert hit is not None and hit.entry.run_id == "r2" and len(hit.entry.findings) == 1


def test_call_ledger_is_keyed(open_stores):
    s = open_stores()
    rec = CallRecord(call_id="c1", agent="writer", model="m", output_type="X", started_at=utcnow(), latency_ms=5, cost_usd=0.01)
    entry = LedgerEntry(run_id="r1", call_key="k1", record=rec, output_json='{"a": 1}')
    assert s.ledger.put(entry) is True
    assert s.ledger.put(entry.model_copy(update={"output_json": '{"a": 2}'})) is False  # first write wins
    got = s.ledger.get("r1", "k1")
    assert got is not None and got.output_json == '{"a": 1}' and got.record == rec
    assert s.ledger.get("r2", "k1") is None
    assert [e.call_key for e in s.ledger.list("r1")] == ["k1"]


def test_conversations_and_profiles(open_stores):
    s = open_stores()
    s.conversations.create(Conversation(id="c1", user_id="u1", title="Sets"))
    s.conversations.create(Conversation(id="c2", user_id="u1", title="Protein"))
    s.conversations.create(Conversation(id="c3", user_id="u2", title="Other"))
    s.conversations.touch("c1", title="Sets per week")
    assert [c.id for c in s.conversations.list("u1")] == ["c1", "c2"]  # most recently updated first
    assert s.conversations.get("c1").title == "Sets per week"
    s.conversations.delete("c2")
    assert s.conversations.get("c2") is None

    assert s.profiles.get("u1") is None
    s.profiles.put("u1", UserProfile(goal="hypertrophy", days_per_week=4))
    s.profiles.put("u1", UserProfile(goal="strength"))
    assert s.profiles.get("u1") == UserProfile(goal="strength")


def test_runner_records_run_and_events(open_stores, search_tool):
    s = open_stores()
    llm = fake_llm(search_tool)
    runner = Runner(GraphDeps(llm=llm, search_tool=search_tool), s)
    r = runner.start("How many sets per week for hypertrophy?", run_id="run-1", user_id="u1", budget=Budget())
    assert r.outcome is not None and r.outcome.status == "answered"

    row = s.runs.get("run-1")
    assert row is not None and row.status == "completed" and row.outcome_status == "answered"
    assert row.cost_usd == pytest.approx(r.spend.cost_usd) and row.user_id == "u1"
    events = s.events.list("run-1")
    assert events[0].type == "run_started" and events[-1].type == "run_finished"
    steps = [e for e in events if e.type == "step"]
    assert [e.data["node"] for e in steps] == [t.node for t in r.trace]  # one event per trace step
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert len(s.ledger.list("run-1")) == r.spend.llm_calls


@pytest.mark.postgres
def test_rls_lets_users_read_only_their_own_rows(clean_pg):
    import psycopg

    a, b = "00000000-0000-0000-0000-00000000000a", "00000000-0000-0000-0000-00000000000b"
    with psycopg.connect(clean_pg, autocommit=True) as conn:
        conn.execute("insert into runs (run_id, question, user_id) values ('ra', 'q', %s), ('rb', 'q', %s)", (a, b))
        conn.execute("insert into run_events (run_id, seq, type) values ('ra', 1, 'step'), ('rb', 1, 'step')")
        conn.execute("insert into topic_findings values ('k', '{k}', 'q', 'sq1', '[]', 'ra', now())")
        with conn.transaction():
            conn.execute("set local role authenticated")
            conn.execute("select set_config('request.jwt.claim.sub', %s, true)", (a,))
            assert [r[0] for r in conn.execute("select run_id from runs")] == ["ra"]
            assert [r[0] for r in conn.execute("select run_id from run_events")] == ["ra"]
            assert conn.execute("select count(*) from topic_findings").fetchone()[0] == 0  # service role only
        with conn.transaction():
            conn.execute("set local role anon")
            assert conn.execute("select count(*) from runs").fetchone()[0] == 0

"""Phase 6 API: FastAPI TestClient on the in-memory stores (and Postgres when TEST_DATABASE_URL is
set) with FakeLLM + fixture search. Covers every endpoint, SSE (live and resumed from
Last-Event-ID), the plan flow, citations in the answer, the trace, forks and diffs, the worker
lease, sliced execution through the HTTP executor, and the reaper."""

from __future__ import annotations

import json
import threading
from datetime import timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.api.app import Services, create_app
from backend.api.executor import HttpExecutor, InlineExecutor, ThreadExecutor
from backend.api.settings import ApiSettings
from backend.api.worker import Worker
from backend.graph import GraphDeps, Runner
from backend.graph.overrides import ADDENDUM_HEADER
from backend.llm import FakeLLM, LLMRequest
from backend.schemas import utcnow
from backend.tests.graph_fakes import INTAKE, fake_llm, writer_response

QUESTION = "I'm a woman, lifting 3 years, 4 days a week. How many sets per week for hypertrophy?"
SECRET = "s3cret"
AUTH = {"Authorization": f"Bearer {SECRET}"}


class Api:
    """One app on one set of stores. `executor`: thread (default), inline or http (through the app)."""

    def __init__(self, stores, search_tool, *, llm: FakeLLM | None = None, executor: str = "thread", **settings: Any):
        self.stores = stores
        self.llm = llm or fake_llm(search_tool)
        self.runner = Runner(GraphDeps(llm=self.llm, search_tool=search_tool), stores)
        defaults: dict[str, Any] = dict(
            executor=executor, worker_secret=SECRET, sse_poll_s=0.01, sse_max_s=10.0, sse_keepalive_s=5.0,
            reap_stale_s=60.0,
        )
        self.settings = ApiSettings(**{**defaults, **settings})
        self.worker = Worker(self.runner, self.settings)
        self.advance_calls = 0
        if executor == "inline":
            self.executor: Any = InlineExecutor(self.worker)
        elif executor == "thread":
            self.executor = ThreadExecutor(self.worker, threads=4)
        else:
            self.executor = None  # wired below, once the app exists
        app = create_app(Services(self.settings, self.runner, self.worker, self._executor_proxy()))
        self.client = TestClient(app)
        if executor == "http":
            api = self

            class Counting(httpx.BaseTransport):
                def handle_request(self, request: httpx.Request) -> httpx.Response:
                    api.advance_calls += 1
                    resp = api.client._transport.handle_request(request)
                    return httpx.Response(resp.status_code, headers=resp.headers, content=resp.read())

            self.executor = HttpExecutor("http://testserver", SECRET, transport=Counting())

    def _executor_proxy(self) -> Any:
        api = self

        class Proxy:
            def submit(self, run_id: str) -> None:
                api.executor.submit(run_id)

            def shutdown(self) -> None:
                pass

        return Proxy()

    def wait(self, run_id: str) -> dict[str, Any]:
        if isinstance(self.executor, ThreadExecutor):
            self.executor.wait(run_id, timeout=30)
        return self.get(run_id)

    def get(self, run_id: str, **headers: str) -> dict[str, Any]:
        r = self.client.get(f"/research/{run_id}", headers=headers)
        assert r.status_code == 200, r.text
        return r.json()

    def start(self, question: str = QUESTION, headers: dict[str, str] | None = None, **body: Any) -> str:
        r = self.client.post("/research", json={"question": question, **body}, headers=headers or {})
        assert r.status_code == 202, r.text
        return r.json()["run_id"]

    def close(self) -> None:
        if isinstance(self.executor, ThreadExecutor):
            self.executor.shutdown()
        self.client.close()


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.postgres)])
def stores(request: pytest.FixtureRequest):
    if request.param == "memory":
        from backend.store.memory import memory_stores

        yield memory_stores()
        return
    from backend.store.postgres import connect, postgres_stores

    url = request.getfixturevalue("clean_pg")
    s = postgres_stores(url, pool=connect(url, min_size=1, max_size=8))
    yield s
    s.close()


@pytest.fixture
def make_api(stores, search_tool):
    made: list[Api] = []

    def make(**kw: Any) -> Api:
        api = Api(stores, search_tool, **kw)
        made.append(api)
        return api

    yield make
    for api in made:
        api.close()


def sse(api: Api, run_id: str, headers: dict[str, str] | None = None, query: str = "") -> list[dict[str, Any]]:
    """Read one SSE response to its end; returns [{id, event, data}, ...] (comments skipped)."""
    out: list[dict[str, Any]] = []
    with api.client.stream("GET", f"/research/{run_id}/events{query}", headers=headers or {}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        block: dict[str, Any] = {}
        for line in resp.iter_lines():
            if not line:
                if block:
                    out.append(block)
                block = {}
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(": ")
            block[field] = json.loads(value) if field == "data" else value
    if block:
        out.append(block)
    return [b for b in out if "event" in b]


class Gate:
    """Blocks a fake model response until released (to observe a run while it is running)."""

    def __init__(self, response: Any):
        self.response = response
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, request: LLMRequest[Any]) -> Any:
        self.entered.set()
        assert self.release.wait(10), "gate never released"
        return self.response(request) if callable(self.response) else self.response


# --- POST /research, GET /research/{id} -------------------------------------------------


def test_post_returns_a_run_id_at_once_then_the_answer(make_api, search_tool):
    gate = Gate(INTAKE)
    api = make_api(llm=fake_llm(search_tool, intake=gate))
    created = api.client.post("/research", json={"question": QUESTION})
    assert created.status_code == 202
    body = created.json()
    run_id = body["run_id"]
    assert body["status"] == "running" and body["links"]["events"] == f"/research/{run_id}/events"
    assert gate.entered.wait(10)
    running = api.get(run_id)
    assert running["status"] == "running" and running["answer"] is None and running["outcome"] is None

    gate.release.set()
    done = api.wait(run_id)
    assert done["status"] == "completed"
    assert done["outcome"]["status"] == "answered"
    answer = done["answer"]
    assert answer["disclaimer"] and "[1]" in answer["recommendation"]
    assert done["citations"]["ok"] is True and done["citations"]["numbers_checked"] >= 1
    assert done["spend"]["cost_usd"] > 0 and done["steps"] == 14
    assert len(done["sub_questions"]) == 3


def test_unknown_and_foreign_runs_are_404(make_api):
    api = make_api(executor="inline")
    assert api.client.get("/research/nope").status_code == 404
    run_id = api.start(headers={"X-User-Id": "alice"})
    assert api.client.get(f"/research/{run_id}", headers={"X-User-Id": "alice"}).status_code == 200
    for headers in ({"X-User-Id": "bob"}, {}):
        assert api.client.get(f"/research/{run_id}", headers=headers).status_code == 404
        assert api.client.get(f"/research/{run_id}/trace", headers=headers).status_code == 404


def test_request_validation(make_api):
    api = make_api(executor="inline")
    assert api.client.post("/research", json={"question": ""}).status_code == 422
    assert api.client.post("/research", json={"question": QUESTION, "budget": {"max_usd": -1}}).status_code == 422
    assert api.client.post("/research", json={"question": QUESTION, "extra": 1}).status_code == 422
    assert api.client.post("/research", json={"question": QUESTION, "conversation_id": "missing"}).status_code == 404


def test_budget_from_the_request_is_enforced(make_api):
    api = make_api(executor="inline")
    run_id = api.start(budget={"max_sub_questions": 2, "max_usd": 0.5})
    done = api.get(run_id)
    assert len(done["sub_questions"]) == 2
    assert any(d["kind"] == "max_sub_questions" for d in done["outcome"]["degradations"])


# --- Plan checkpoint ---------------------------------------------------------------


def test_plan_approve_flow(make_api):
    api = make_api()
    run_id = api.start(require_plan_approval=True)
    waiting = api.wait(run_id)
    assert waiting["status"] == "awaiting_approval"
    plan = waiting["plan"]
    assert [sq["id"] for sq in plan["sub_questions"]] == ["sq1", "sq2", "sq3"] and plan["error"] is None
    assert api.llm.calls_for("researcher") == []

    r = api.client.post(f"/research/{run_id}/plan", json={"action": "approve"})
    assert r.status_code == 202 and r.json()["status"] == "running"
    done = api.wait(run_id)
    assert done["status"] == "completed" and done["outcome"]["status"] == "answered" and done["plan"] is None
    # Deciding again is a conflict.
    assert api.client.post(f"/research/{run_id}/plan", json={"action": "approve"}).status_code == 409


def test_plan_edit_flow(make_api):
    api = make_api()
    run_id = api.start(require_plan_approval=True)
    api.wait(run_id)
    edit = {
        "action": "edit",
        "sub_questions": [
            {"id": "sq1"},
            {"question": "Does weekly training frequency change hypertrophy when weekly set volume is equated?"},
        ],
        "note": "drop the plateau question",
    }
    edit["sub_questions"][0]["question"] = api.get(run_id)["plan"]["sub_questions"][0]["question"]
    assert api.client.post(f"/research/{run_id}/plan", json=edit).status_code == 202
    done = api.wait(run_id)
    assert done["status"] == "completed"
    assert [sq["id"] for sq in done["sub_questions"]] == ["sq1", "sq4"]  # a new question gets a fresh id
    assert "frequency" in done["sub_questions"][1]["question"]
    researched = {req.user.split("Sub-question: ")[1].split("\n")[0] for req in api.llm.calls_for("researcher")}
    assert not any("plateau" in q.lower() for q in researched)
    events = [e["event"] for e in sse(api, run_id)]
    assert "plan_ready" in events and "plan_decided" in events


def test_invalid_plan_edit_is_rejected_and_the_run_keeps_waiting(make_api):
    api = make_api()
    run_id = api.start(require_plan_approval=True)
    api.wait(run_id)
    bad = {"action": "edit", "sub_questions": [{"id": "sq9", "question": "A question about something unknown?"}]}
    r = api.client.post(f"/research/{run_id}/plan", json=bad)
    assert r.status_code == 422 and "unknown sub-question ids" in r.json()["detail"]
    assert api.client.post(f"/research/{run_id}/plan", json={"action": "edit"}).status_code == 422
    assert api.get(run_id)["status"] == "awaiting_approval"


def test_plan_on_a_run_that_is_not_waiting_is_a_conflict(make_api):
    api = make_api(executor="inline")
    run_id = api.start()
    assert api.client.post(f"/research/{run_id}/plan", json={"action": "approve"}).status_code == 409


# --- SSE ---------------------------------------------------------------------------


def test_sse_streams_useful_progress_to_the_end(make_api):
    api = make_api(executor="inline")
    run_id = api.start()
    events = sse(api, run_id)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_started" and kinds[-2:] == ["run_finished", "end"]
    ids = [int(e["id"]) for e in events if "id" in e]
    assert ids == list(range(1, len(ids) + 1))  # gap-free
    steps = [e["data"] for e in events if e["event"] == "step"]
    messages = [s["message"] for s in steps]
    assert "Planned 3 sub-questions" in messages
    assert any(m.startswith("Fixture library: ") and "→" in m and "kept" in m for m in messages)
    assert any(m.startswith("Citations: ") and m.endswith("backed") for m in messages)
    assert "Answer accepted" in messages and messages[-1] == "Done: answered"
    costs = [s["totals"]["cost_usd"] for s in steps]
    tokens = [s["totals"]["tokens"] for s in steps]
    assert all(b >= a - 1e-6 for a, b in zip(costs, costs[1:])) and costs[-1] > 0  # running totals
    assert tokens == sorted(tokens) and tokens[-1] > 0
    final = events[-2]["data"]
    assert final["totals"]["cost_usd"] == pytest.approx(costs[-1])
    assert events[-1]["data"]["status"] == "completed"


def test_sse_debate_messages(stores, search_tool):
    from backend.tests.test_graph_synthesis import contested_llm

    api = Api(stores, search_tool, llm=contested_llm(search_tool), executor="inline")
    try:
        run_id = api.start()
        messages = [e["data"]["message"] for e in sse(api, run_id) if e["event"] == "step"]
        assert "Debate: round 1 · P1 opening" in messages and "Debate: round 2 · P2 rebuttal" in messages
        assert any(m.startswith("Positions mapped: genuinely contested, 2 sides") for m in messages)
        assert any(m.startswith("Verdict: ") for m in messages)
    finally:
        api.close()


def test_sse_resumes_after_last_event_id(make_api):
    api = make_api(executor="inline")
    run_id = api.start()
    full = [e for e in sse(api, run_id) if "id" in e]
    resumed = [e for e in sse(api, run_id, headers={"Last-Event-ID": "5"}) if "id" in e]
    assert [int(e["id"]) for e in resumed] == [int(e["id"]) for e in full][5:]
    assert resumed[0]["data"] == full[5]["data"]
    by_query = [e for e in sse(api, run_id, query="?after=10") if "id" in e]
    assert [int(e["id"]) for e in by_query] == [int(e["id"]) for e in full][10:]


def test_sse_follows_a_live_run_and_reconnects(make_api, search_tool):
    gate = Gate(writer_response())
    api = make_api(llm=fake_llm(search_tool, writer=gate), sse_max_s=0.3)
    run_id = api.start()
    assert gate.entered.wait(10)
    # The response ends after sse_max_s without `end` (the run is still going) ...
    first = sse(api, run_id)
    assert first and first[-1]["event"] == "step" and "end" not in [e["event"] for e in first]
    last_id = first[-1]["id"]
    # ... and the client reconnects with Last-Event-ID once the run moves on.
    gate.release.set()
    api.wait(run_id)
    second = sse(api, run_id, headers={"Last-Event-ID": last_id})
    assert int(second[0]["id"]) == int(last_id) + 1
    assert second[-1]["event"] == "end"
    seen = [int(e["id"]) for e in first + second if "id" in e]
    assert seen == list(range(1, len(seen) + 1))


def test_sse_live_stream_to_completion(make_api):
    api = make_api()
    run_id = api.start()
    events = sse(api, run_id)  # opened while the worker thread runs
    assert events[-1]["event"] == "end" and events[-1]["data"]["status"] == "completed"
    assert sum(1 for e in events if e["event"] == "run_finished") == 1


# --- Trace -----------------------------------------------------------------------


def test_trace_has_the_full_step_log(make_api):
    api = make_api(executor="inline")
    run_id = api.start()
    trace = api.client.get(f"/research/{run_id}/trace").json()
    steps = trace["steps"]
    assert [s["node"] for s in steps][:3] == ["intake", "planner", "dispatch"]
    assert [s["index"] for s in steps] == list(range(len(steps)))
    researcher = next(s for s in steps if s["node"] == "researcher")
    assert researcher["agent"] == "researcher" and researcher["sub_question_id"]
    assert researcher["tool_calls"] and researcher["tool_calls"][0]["tool"] == "search:fixture"
    assert researcher["llm_calls"][0]["model"] == config.HAIKU
    assert researcher["input_tokens"] > 0 and researcher["cost_usd"] > 0 and researcher["latency_ms"] >= 0
    assert researcher["input_summary"] and researcher["output"]["findings"]
    gate = next(s for s in steps if s["node"] == "validate_citations")
    assert gate["output"]["ok"] is True
    assert trace["totals"]["cost_usd"] == pytest.approx(sum(s["cost_usd"] for s in steps))
    assert trace["inherited_steps"] == 0


# --- Citation gate through the API ------------------------------------------------------


def test_unbacked_number_is_rewritten_once(make_api, search_tool):
    llm = fake_llm(search_tool, writer=[writer_response(unbacked_number="17"), writer_response()])
    api = make_api(llm=llm, executor="inline")
    done = api.get(api.start())
    assert done["outcome"]["status"] == "answered" and "17" not in done["answer"]["practical_protocol"]
    assert done["citations"]["ok"] is True and done["citations"]["attempt"] == 2


def test_answer_failing_citation_check_twice_is_not_returned(make_api, search_tool):
    api = make_api(llm=fake_llm(search_tool, writer=writer_response(unbacked_number="17")), executor="inline")
    run_id = api.start()
    done = api.get(run_id)
    assert done["status"] == "completed" and done["answer"] is None
    assert done["outcome"]["status"] == "failed"
    assert done["outcome"]["failure"]["reason"] == "citations_failed"
    assert done["citations"]["ok"] is False
    assert any(i["number"] == "17" for i in done["citations"]["issues"])
    messages = [e["data"]["message"] for e in sse(api, run_id) if e["event"] == "step"]
    assert "Answer sent back for one rewrite" in messages
    assert any(m.startswith("Citations: 1 problem") for m in messages)


# --- Time-travel forks -------------------------------------------------------------------


def _step(trace: list[dict[str, Any]], node: str, **match: Any) -> dict[str, Any]:
    return next(s for s in trace if s["node"] == node and all(s.get(k) == v for k, v in match.items()))


def test_fork_writer_with_prompt_addendum_diffs_the_answer(make_api, search_tool):
    plain = writer_response()

    def writer(request: LLMRequest[Any]) -> Any:
        draft = plain(request)
        if "Lead with frequency" in request.system:  # the override changes what the writer says
            ids = draft.cited_finding_ids
            draft = draft.model_copy(update={"recommendation": f"Train each muscle at least twice a week [{ids[0]}]."})
        return draft

    api = make_api(llm=fake_llm(search_tool, writer=writer), executor="inline")
    original = api.start(headers={"X-User-Id": "alice"})
    trace = api.client.get(f"/research/{original}/trace", headers={"X-User-Id": "alice"}).json()["steps"]
    writer_step = _step(trace, "writer")
    before = {agent: len(api.llm.calls_for(agent)) for agent in ("intake", "planner", "researcher", "judge", "writer")}

    r = api.client.post(
        f"/research/{original}/fork?from_step={writer_step['index']}",
        json={"prompt_addendum": "Lead with frequency."},
        headers={"X-User-Id": "alice"},
    )
    assert r.status_code == 202, r.text
    fork_id = r.json()["run_id"]
    assert fork_id != original and r.json()["parent_run_id"] == original

    fork = api.get(fork_id, **{"X-User-Id": "alice"})
    assert fork["status"] == "completed" and fork["parent_run_id"] == original
    assert fork["fork"]["from_step"] == writer_step["index"] and fork["fork"]["scope"]["agent"] == "writer"
    # Only the writer ran again; nothing before the fork point was re-run or re-paid.
    after = {agent: len(api.llm.calls_for(agent)) for agent in before}
    assert {a: after[a] - before[a] for a in after} == {"intake": 0, "planner": 0, "researcher": 0, "judge": 0, "writer": 1}
    assert ADDENDUM_HEADER.strip() in api.llm.calls_for("writer")[-1].system
    diff = fork["diff"]
    assert diff["changed"] is True and diff["original_run_id"] == original and diff["fork_run_id"] == fork_id
    changed = {c["field"]: c for c in diff["changes"]}
    assert set(changed) == {"recommendation"}  # status and confidence come from the judge, not the writer
    assert changed["recommendation"]["after"] == "Train each muscle at least twice a week [1]."
    assert diff["positions_added"] == [] and diff["positions_removed"] == []
    assert diff["new_cost_usd"] > 0
    fork_trace = api.client.get(f"/research/{fork_id}/trace", headers={"X-User-Id": "alice"}).json()
    assert fork_trace["inherited_steps"] == writer_step["index"]
    assert all(s["run_id"] == original for s in fork_trace["steps"][: writer_step["index"]])
    assert fork_trace["steps"][writer_step["index"]]["node"] == "writer"
    assert fork_trace["steps"][writer_step["index"]]["run_id"] == fork_id
    # The original is untouched.
    assert "twice a week" not in api.get(original, **{"X-User-Id": "alice"})["answer"]["recommendation"]


def test_fork_researcher_with_tool_result_replays_its_siblings(make_api, search_tool):
    api = make_api(executor="inline")
    original = api.start()
    trace = api.client.get(f"/research/{original}/trace").json()["steps"]
    step = _step(trace, "researcher", sub_question_id="sq2")
    record = next(r for r in search_tool.records if r.source_id == "FIXTURE-0005").model_dump(mode="json")
    before = [req.user for req in api.llm.calls_for("researcher")]

    r = api.client.post(f"/research/{original}/fork?from_step={step['index']}", json={"tool_result": [record]})
    assert r.status_code == 202, r.text
    fork_id = r.json()["run_id"]
    fork = api.get(fork_id)
    assert fork["status"] == "completed"
    new_calls = [req.user for req in api.llm.calls_for("researcher")][len(before):]
    # Only the overridden researcher ran (its off-population record may earn it a second pass);
    # its siblings replayed their recorded calls.
    sq2_question = step["input_summary"].split(": ", 1)[1].split(" | ")[0]
    assert new_calls and all(sq2_question[:40] in u for u in new_calls)
    assert "FIXTURE-0005" in new_calls[0]
    fork_trace = api.client.get(f"/research/{fork_id}/trace").json()["steps"]
    sq2 = [s for s in fork_trace if s["node"] == "researcher" and s["sub_question_id"] == "sq2" and s["run_id"] == fork_id]
    assert sq2 and {c["tool"] for c in sq2[0]["tool_calls"]} == {"search:override"}
    assert [f["source_id"] for f in sq2[0]["output"]["findings"]] == ["FIXTURE-0005"]
    diff = fork["diff"]
    # sq2 lost its original studies: their citations are gone from the answer, and the
    # off-population replacement makes the outcome degraded.
    assert diff["changed"] and "sq2:FIXTURE-0001" in diff["citations_removed"]
    assert {"field": "outcome", "before": "answered", "after": "degraded"} in diff["changes"]
    assert diff["positions_changed"] and "sq2:FIXTURE-0001" in diff["positions_changed"][0]["findings_removed"]


def test_fork_from_the_first_step_with_a_model_override(make_api):
    api = make_api(executor="inline")
    original = api.start()
    r = api.client.post(f"/research/{original}/fork?from_step=0", json={"model": config.OPUS})
    assert r.status_code == 202, r.text
    fork = api.get(r.json()["run_id"])
    assert fork["status"] == "completed" and fork["outcome"]["status"] == "answered"
    assert api.llm.calls_for("intake")[-1].model == config.OPUS
    assert fork["diff"]["changed"] is False  # same answer from the stronger model
    trace = api.client.get(f"/research/{fork['run_id']}/trace").json()
    assert trace["inherited_steps"] == 0 and trace["steps"][0]["model"] == config.OPUS
    assert trace["steps"][0]["llm_calls"][0]["model"] == config.OPUS


def test_fork_validation(make_api):
    api = make_api(executor="inline")
    original = api.start()
    trace = api.client.get(f"/research/{original}/trace").json()["steps"]
    dispatch = _step(trace, "dispatch")["index"]
    writer = _step(trace, "writer")["index"]
    post = api.client.post
    assert post(f"/research/{original}/fork?from_step={dispatch}", json={"model": config.OPUS}).status_code == 422
    assert post(f"/research/{original}/fork?from_step=999", json={"model": config.OPUS}).status_code == 422
    assert post(f"/research/{original}/fork?from_step={writer}", json={"tool_result": []}).status_code == 422
    assert post(f"/research/{original}/fork?from_step={writer}", json={"model": "gpt-4"}).status_code == 422
    assert post(f"/research/{original}/fork?from_step={writer}", json={}).status_code == 422
    assert post(f"/research/{original}/fork", json={"model": config.OPUS}).status_code == 422  # from_step required
    assert post("/research/nope/fork?from_step=0", json={"model": config.OPUS}).status_code == 404


def test_fork_of_an_unfinished_run_is_a_conflict(make_api):
    api = make_api()
    run_id = api.start(require_plan_approval=True)
    api.wait(run_id)
    r = api.client.post(f"/research/{run_id}/fork?from_step=0", json={"model": config.OPUS})
    assert r.status_code == 409


# --- Lease, slices and the reaper ---------------------------------------------------------


def test_lease_only_one_of_many_concurrent_advancers_wins(stores, search_tool):
    runner = Runner(GraphDeps(llm=fake_llm(search_tool), search_tool=search_tool), stores)
    run_id = runner.create(QUESTION).run_id
    barrier = threading.Barrier(8)
    won: list[str] = []

    def contend(i: int) -> None:
        barrier.wait()
        if stores.runs.acquire_lease(run_id, f"w{i}", 60) is not None:
            won.append(f"w{i}")

    threads = [threading.Thread(target=contend, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(won) == 1
    owner = won[0]
    assert stores.runs.acquire_lease(run_id, "other", 60) is None
    assert stores.runs.acquire_lease(run_id, owner, 60) is not None  # re-entrant for its owner
    assert stores.runs.renew_lease(run_id, owner, 60) and not stores.runs.renew_lease(run_id, "other", 60)
    assert not stores.runs.release_lease(run_id, "other")
    assert stores.runs.release_lease(run_id, owner)
    assert stores.runs.acquire_lease(run_id, "other", 60) is not None
    # An expired lease can be taken over.
    stores.runs.update(run_id, lease_expires_at=utcnow() - timedelta(seconds=1))
    assert stores.runs.acquire_lease(run_id, "third", 60) is not None


def test_two_workers_on_one_run_only_one_does_the_work(stores, search_tool):
    gate = Gate(INTAKE)
    llm = fake_llm(search_tool, intake=gate)
    runner = Runner(GraphDeps(llm=llm, search_tool=search_tool), stores)
    worker = Worker(runner, ApiSettings(worker_secret=SECRET))
    run_id = runner.create(QUESTION).run_id
    results: dict[str, Any] = {}
    first = threading.Thread(target=lambda: results.__setitem__("first", worker.run_to_completion(run_id)))
    first.start()
    assert gate.entered.wait(10)
    results["second"] = worker.advance(run_id)  # while the first holds the lease
    gate.release.set()
    first.join(30)
    assert results["second"].state == "busy"
    assert results["first"].state == "finished"
    assert len(llm.calls_for("intake")) == 1
    assert worker.advance(run_id).state == "not_running"  # completed runs are not advanced again
    rec = stores.runs.get(run_id)
    assert rec.status == "completed" and rec.lease_owner is None


def test_http_executor_runs_the_graph_in_slices(make_api):
    api = make_api(executor="http", slice_s=0.0)  # stop at every superstep boundary
    run_id = api.start()
    done = api.get(run_id)
    assert done["status"] == "completed" and done["outcome"]["status"] == "answered"
    # One /internal/runs/{id}/advance request per superstep, each re-triggering the next.
    assert api.advance_calls >= 10
    assert len(api.llm.calls_for("intake")) == 1 and len(api.llm.calls_for("planner")) == 1
    events = sse(api, run_id)
    assert [e["event"] for e in events].count("run_finished") == 1


def test_internal_routes_need_the_secret(make_api):
    api = make_api(executor="inline")
    run_id = api.start()
    assert api.client.post(f"/internal/runs/{run_id}/advance").status_code == 401
    assert api.client.post(f"/internal/runs/{run_id}/advance", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = api.client.post(f"/internal/runs/{run_id}/advance", headers=AUTH)
    assert ok.status_code == 200 and ok.json()["state"] == "not_running"
    assert api.client.get("/internal/cron/reap").status_code == 401
    assert api.client.get("/internal/cron/reap", headers=AUTH).status_code == 200
    closed = make_api(executor="inline", worker_secret=None)
    assert closed.client.post(f"/internal/runs/{run_id}/advance", headers=AUTH).status_code == 503


def test_cron_secret_also_opens_the_reaper(make_api):
    api = make_api(executor="inline", cron_secret="cron!")
    assert api.client.get("/internal/cron/reap", headers={"Authorization": "Bearer cron!"}).status_code == 200


def test_reaper_resumes_a_run_whose_worker_died(make_api):
    api = make_api(executor="inline")
    run_id = api.runner.create(QUESTION).run_id
    # A worker ran one slice and then died holding the lease.
    first = api.worker.advance(run_id, slice_s=0.0)
    assert first.state == "paused"
    api.stores.runs.acquire_lease(run_id, "dead-worker", 300)
    assert api.client.get("/internal/cron/reap", headers=AUTH).json()["dispatched"] == []  # lease still valid
    assert api.worker.advance(run_id).state == "busy"
    api.stores.runs.update(run_id, lease_expires_at=utcnow() - timedelta(seconds=1))  # ... until it expires

    reaped = api.client.get("/internal/cron/reap", headers=AUTH).json()
    assert reaped["dispatched"] == [run_id]
    done = api.get(run_id)
    assert done["status"] == "completed" and done["outcome"]["status"] == "answered"
    assert len(api.llm.calls_for("intake")) == 1  # resumed, not replayed
    assert api.client.get("/internal/cron/reap", headers=AUTH).json()["dispatched"] == []


def test_reaper_dispatches_lost_hand_offs_and_retries_errors_but_not_waiting_runs(make_api, search_tool):
    api = make_api(executor="inline", reap_stale_s=0.0, max_auto_retries=2)
    lost = api.runner.create(QUESTION).run_id  # created, but the first advance never arrived
    waiting = api.runner.create(QUESTION, require_plan_approval=True).run_id
    api.worker.advance(waiting)
    assert api.stores.runs.get(waiting).status == "awaiting_approval"
    failed = api.runner.create(QUESTION).run_id
    api.stores.runs.update(failed, status="error", error="database went away", attempts=1)
    exhausted = api.runner.create(QUESTION).run_id
    api.stores.runs.update(exhausted, status="error", error="again", attempts=2)

    reaped = api.client.get("/internal/cron/reap", headers=AUTH).json()
    assert set(reaped["dispatched"]) == {lost, failed} and reaped["retried"] == [failed]
    for run_id in (lost, failed):
        assert api.get(run_id)["status"] == "completed"
    assert api.get(waiting)["status"] == "awaiting_approval"
    assert api.get(exhausted)["status"] == "error"


def test_advance_error_marks_the_run_and_counts_attempts(make_api):
    api = make_api(executor="inline")
    run_id = api.runner.create(QUESTION).run_id
    original = api.runner.graph.stream

    def exploding(*a, **kw):
        raise RuntimeError("checkpointer outage")

    api.runner.graph.stream = exploding  # type: ignore[method-assign]
    assert api.worker.advance(run_id).state == "error"
    rec = api.stores.runs.get(run_id)
    assert rec.status == "error" and rec.attempts == 1 and "outage" in rec.error and rec.lease_owner is None
    api.runner.graph.stream = original  # type: ignore[method-assign]
    api.stores.runs.transition(run_id, ("error",), status="running")
    assert api.worker.run_to_completion(run_id).state == "finished"


# --- Conversations and profile -------------------------------------------------------------


def test_conversations_and_profile(make_api):
    api = make_api(executor="inline")
    alice, bob = {"X-User-Id": "alice"}, {"X-User-Id": "bob"}
    assert api.client.get("/profile").status_code == 401
    assert api.client.get("/profile", headers=alice).json() == {"user_id": "alice", "profile": {**_empty_profile()}}
    profile = {"sex": "female", "training_status": "advanced", "days_per_week": 5}
    put = api.client.put("/profile", json=profile, headers=alice)
    assert put.status_code == 200 and put.json()["profile"]["days_per_week"] == 5
    assert api.client.put("/profile", json={"sex": "robot"}, headers=alice).status_code == 422

    first = api.client.post("/research", json={"question": QUESTION}, headers=alice).json()
    conv_id = first["conversation_id"]
    assert api.stores.runs.get(first["run_id"]).profile.days_per_week == 5  # saved profile feeds intake
    follow = api.start("What if I only have 3 days a week for training?", headers=alice, conversation_id=conv_id)

    convs = api.client.get("/conversations", headers=alice).json()
    assert [c["id"] for c in convs] == [conv_id] and convs[0]["title"].startswith("I'm a woman")
    view = api.client.get(f"/conversations/{conv_id}", headers=alice).json()
    assert [r["run_id"] for r in view["runs"]] == [first["run_id"], follow]
    assert all(r["status"] == "completed" for r in view["runs"])

    assert api.client.get("/conversations", headers=bob).json() == []
    assert api.client.get(f"/conversations/{conv_id}", headers=bob).status_code == 404
    assert api.client.post("/research", json={"question": QUESTION, "conversation_id": conv_id}, headers=bob).status_code == 404
    assert api.client.delete(f"/conversations/{conv_id}", headers=bob).status_code == 404
    assert api.client.delete(f"/conversations/{conv_id}", headers=alice).status_code == 204
    assert api.client.get("/conversations", headers=alice).json() == []
    assert api.get(follow, **alice)["status"] == "completed"  # runs outlive their conversation


def _empty_profile() -> dict[str, Any]:
    from backend.schemas import UserProfile

    return UserProfile().model_dump(mode="json")


def test_supabase_jwt_auth(make_api):
    import jwt

    api = make_api(executor="inline", auth="jwt", supabase_jwt_secret="a-test-jwt-secret-that-is-32-bytes-long")
    token = jwt.encode({"sub": "user-1", "aud": "authenticated", "role": "authenticated"}, "a-test-jwt-secret-that-is-32-bytes-long", algorithm="HS256")
    ok = api.client.get("/profile", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200 and ok.json()["user_id"] == "user-1"
    forged = jwt.encode({"sub": "user-1", "aud": "authenticated"}, "another-secret-that-is-32-bytes-long!", algorithm="HS256")
    assert api.client.get("/profile", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    assert api.client.get("/profile", headers={"X-User-Id": "user-1"}).status_code == 401  # header ignored


def test_health(make_api):
    api = make_api(executor="inline")
    assert api.client.get("/health").json()["ok"] is True


def test_settings_from_env():
    s = ApiSettings.from_env({"VERCEL": "1", "VERCEL_URL": "x.vercel.app", "WORKER_SECRET": "w", "SSE_MAX_S": "30"})
    assert s.executor == "http" and s.worker_base_url == "https://x.vercel.app" and s.sse_max_s == 30
    assert ApiSettings.from_env({}).executor == "thread"
    assert ApiSettings.from_env({"SUPABASE_JWT_SECRET": "k"}).auth == "jwt"
    with pytest.raises(ValueError):
        ApiSettings.from_env({"EXECUTOR": "celery"})

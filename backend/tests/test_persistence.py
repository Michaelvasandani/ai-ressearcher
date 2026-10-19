"""Phase 5: resume instead of replay, idempotent retries, the plan checkpoint and topic reuse.

Each test runs on the in-memory stores, and on Postgres when TEST_DATABASE_URL is set.
A "process" is a Runner with its own FakeLLM and search tool; `open_stores()` gives it
the stores (for Postgres a new connection pool), so state only crosses "processes"
through the checkpointer and the stores.
"""

from __future__ import annotations

import pytest

from backend.graph import (
    GraphDeps,
    PlanDecision,
    Runner,
    memory_checkpointer,
    resume_research,
    run_research,
)
from backend.graph.budget import Budget
from backend.llm import FakeLLM
from backend.store.memory import MemoryCallLedger
from backend.tests.graph_fakes import FakeClock, ScriptedSearch, fake_llm

QUESTION = "I'm a woman, lifting 3 years, 4 days a week. How many sets per week for hypertrophy?"


class SimulatedCrash(BaseException):
    """Stands in for the process dying: not an Exception, so no node or agent catches it."""


class CrashingSearch(ScriptedSearch):
    """Kills the "process" the first time a query containing `pattern` runs."""

    def __init__(self, base, pattern: str):
        super().__init__(base)
        self.pattern = pattern
        self.crashed = False

    def search(self, query: str, max_results: int = 10):
        if self.pattern in query and not self.crashed:
            self.crashed = True
            raise SimulatedCrash(f"killed while searching {query!r}")
        return super().search(query, max_results)


class CrashAfterRecording:
    """Wraps a ledger: the process dies right after the first `agent` call is recorded,
    before the node's writes reach the checkpoint."""

    def __init__(self, inner, agent: str):
        self.inner, self.agent, self.crashed = inner, agent, False

    def get(self, run_id, call_key):
        return self.inner.get(run_id, call_key)

    def list(self, run_id):
        return self.inner.list(run_id)

    def put(self, entry):
        inserted = self.inner.put(entry)
        if entry.record.agent == self.agent and not self.crashed:
            self.crashed = True
            raise SimulatedCrash(f"killed after the {self.agent} call was recorded")
        return inserted


def process(stores, search_tool, *, llm: FakeLLM | None = None, search=None, clock=None, **kw) -> tuple[Runner, FakeLLM]:
    llm = llm or fake_llm(search_tool)
    deps = GraphDeps(llm=llm, search_tool=search or search_tool, **({"clock": clock} if clock else {}))
    return Runner(deps, stores, **kw), llm


def total_spent(*llms: FakeLLM) -> float:
    return sum(llm.total_cost_usd for llm in llms)


# --- Resume, don't replay ---------------------------------------------------------


def test_crash_in_parallel_researcher_resumes_without_replay(open_stores, search_tool):
    first, llm1 = process(
        open_stores(), search_tool, search=CrashingSearch(search_tool, "high volume"), reuse_findings=False
    )
    with pytest.raises(SimulatedCrash):
        first.start(QUESTION, run_id="run-crash")
    assert llm1.calls_for("intake") and llm1.calls_for("planner")

    stores = open_stores()  # a new process: nothing shared but the database
    assert stores.runs.get("run-crash").status == "running"  # what a killed process leaves behind
    second, llm2 = process(stores, search_tool, reuse_findings=False)
    r = second.resume("run-crash")

    assert r.outcome is not None and r.outcome.status == "answered"
    # Completed nodes did not run again: no intake / planner calls, only the crashed researcher re-ran.
    assert llm2.calls_for("intake") == [] and llm2.calls_for("planner") == []
    # The crashed researcher re-ran. A sibling still in flight when the process died may
    # re-run too (timing-dependent), but no researcher call paid for in process 1 is paid again.
    rerun = [req.user for req in llm2.calls_for("researcher")]
    assert all(u.count("Sub-question:") == 1 for u in rerun)
    assert any("plateau" in u.lower() for u in rerun)
    assert not {req.user for req in llm1.calls_for("researcher")} & set(rerun)
    # One trace step per researcher, and one step event per trace step: nothing logged twice.
    assert sorted(s.sub_question_id for s in r.trace if s.node == "researcher") == ["sq1", "sq2", "sq3"]
    steps = [e for e in stores.events.list("run-crash") if e.type == "step"]
    assert len(steps) == len(r.trace) and len({e.key for e in steps}) == len(steps)
    # Spend is counted once: the state equals what the two processes actually paid.
    assert r.spend.cost_usd == pytest.approx(total_spent(llm1, llm2))
    assert r.spend.llm_calls == len(llm1.records) + len(llm2.records)
    assert r.spend.cost_usd == pytest.approx(sum(s.cost_usd for s in r.trace))
    row = stores.runs.get("run-crash")
    assert row.status == "completed" and row.cost_usd == pytest.approx(r.spend.cost_usd)


def test_retried_step_replays_its_recorded_call_instead_of_paying_again(open_stores, search_tool):
    stores = open_stores()
    stores.ledger = CrashAfterRecording(stores.ledger, "writer")
    first, llm1 = process(stores, search_tool, reuse_findings=False)
    with pytest.raises(SimulatedCrash):
        first.start(QUESTION, run_id="run-ledger")
    assert len(llm1.calls_for("writer")) == 1  # paid for, but the writer's writes never reached the checkpoint

    stores = open_stores()
    second, llm2 = process(stores, search_tool, reuse_findings=False)
    r = second.resume("run-ledger")

    assert r.outcome is not None and r.outcome.status == "answered"
    assert llm2.requests == []  # the writer re-ran, but its call came from the ledger
    writer = [s for s in r.trace if s.node == "writer"]
    assert len(writer) == 1 and [c.replayed for c in writer[0].llm_calls] == [True]
    assert writer[0].cost_usd > 0  # the call was paid once (by the first process) and is counted once
    assert r.spend.cost_usd == pytest.approx(total_spent(llm1, llm2))
    assert r.spend.cost_usd == pytest.approx(sum(e.record.cost_usd for e in stores.ledger.list("run-ledger")))
    assert r.spend.llm_calls == len(stores.ledger.list("run-ledger")) == len(llm1.records)
    replayed = [e for e in stores.events.list("run-ledger") if e.type == "step" and e.data["replayed_calls"]]
    assert [e.data["node"] for e in replayed] == ["writer"]


def test_resuming_a_finished_run_returns_the_cached_result(open_stores, search_tool):
    first, _ = process(open_stores(), search_tool)
    done = first.start(QUESTION, run_id="run-done")

    stores = open_stores()
    events_before = len(stores.events.list("run-done"))
    second, llm2 = process(stores, search_tool)
    again = second.resume("run-done")

    assert llm2.requests == []
    assert again.answer == done.answer and again.spend == done.spend
    assert [s.step_id for s in again.trace] == [s.step_id for s in done.trace]
    assert len(stores.events.list("run-done")) == events_before
    assert second.get("run-done").outcome == done.outcome
    with pytest.raises(KeyError):
        second.resume("no-such-run")


def test_checkpoint_holds_run_state_after_every_node(open_stores, search_tool):
    stores = open_stores()
    runner, _ = process(stores, search_tool)
    r = runner.start(QUESTION, run_id="run-state")
    history = list(runner.graph.get_state_history({"configurable": {"thread_id": "run-state"}}))
    assert len(history) >= len({s.node for s in r.trace})

    # Read it back through a fresh checkpointer (another process).
    fresh = Runner(GraphDeps(llm=FakeLLM(), search_tool=search_tool), open_stores())
    values = fresh.graph.get_state({"configurable": {"thread_id": "run-state"}}).values
    assert values["run_id"] == "run-state" and values["plan"] == r.plan
    assert set(values["research"]) == {"sq1", "sq2", "sq3"}  # completed sub-questions with their findings
    assert values["position_map"] == r.position_map and values["debate"] == r.debate
    assert values["spend"] == r.spend and values["spend"].cost_usd > 0
    assert len(values["trace"]) == len(r.trace)


def test_run_research_resumes_with_a_shared_checkpointer(search_tool):
    """The plain helpers (no stores) resume too."""
    saver = memory_checkpointer()
    deps = GraphDeps(llm=fake_llm(search_tool), search_tool=CrashingSearch(search_tool, "high volume"))
    with pytest.raises(SimulatedCrash):
        run_research(QUESTION, deps=deps, run_id="plain", checkpointer=saver)
    llm2 = fake_llm(search_tool)
    r = resume_research("plain", deps=GraphDeps(llm=llm2, search_tool=search_tool), checkpointer=saver)
    assert r.outcome.status == "answered" and llm2.calls_for("planner") == []


# --- Plan checkpoint ---------------------------------------------------------------


def test_plan_checkpoint_waits_for_hours_and_resumes_with_an_edit(open_stores, search_tool):
    clock = FakeClock()
    budget = Budget(wall_clock_s=300)
    first, llm1 = process(open_stores(), search_tool, clock=clock)
    paused = first.start(QUESTION, run_id="run-plan", require_plan_approval=True, budget=budget)

    assert paused.outcome is None and paused.awaiting_approval is not None
    req = paused.awaiting_approval
    assert [sq.id for sq in req.sub_questions] == ["sq1", "sq2", "sq3"] and req.error is None
    assert llm1.calls_for("researcher") == []  # nothing searched before approval
    stores = open_stores()
    assert stores.runs.get("run-plan").status == "awaiting_approval"
    plan_ready = [e for e in stores.events.list("run-plan") if e.type == "plan_ready"]
    assert len(plan_ready) == 1 and len(plan_ready[0].data["sub_questions"]) == 3

    # Resuming without a decision just returns the pending request.
    second, llm2 = process(stores, search_tool, clock=clock)
    assert second.resume("run-plan").awaiting_approval == req

    clock.advance(5 * 3600)  # the user answers five hours later, in another process
    keep = req.sub_questions[0]
    decision = PlanDecision(
        action="edit",
        sub_questions=[
            {"id": keep.id, "question": keep.question},
            {"question": "Does weekly training frequency change hypertrophy when weekly set volume is equated?",
             "search_terms": ["training frequency volume equated hypertrophy meta-analysis"]},
        ],
    )
    r = second.resume("run-plan", decision)

    assert r.outcome is not None and r.outcome.status == "answered", r.outcome
    assert [sq.id for sq in r.sub_questions] == ["sq1", "sq4"]  # a new question never reuses a planner id
    assert r.sub_questions[0].search_terms == keep.search_terms  # unchanged question keeps the planner's terms
    assert r.plan_decision is not None and r.plan_decision.action == "edit"
    assert {s.sub_question_id for s in r.trace if s.node == "researcher"} == {"sq1", "sq4"}
    assert llm2.calls_for("planner") == [] and len(llm1.calls_for("planner")) == 1
    # The five-hour wait is not charged to the five-minute wall-clock budget.
    assert r.waited_s == pytest.approx(5 * 3600)
    assert "wall_clock" not in {d.kind for d in r.outcome.degradations}
    checkpoint = [s for s in r.trace if s.node == "plan_checkpoint"]
    assert len(checkpoint) == 1 and checkpoint[0].agent == "human" and "waited 18000s" in checkpoint[0].output_summary


def test_invalid_plan_edit_asks_again(open_stores, search_tool):
    runner, llm = process(open_stores(), search_tool)
    paused = runner.start(QUESTION, run_id="run-bad-edit", require_plan_approval=True)
    bad = runner.resume("run-bad-edit", {"action": "edit", "sub_questions": [{"id": "sq9", "question": "Unknown id here?"}]})
    assert bad.outcome is None and bad.awaiting_approval is not None
    assert "unknown sub-question ids: sq9" in bad.awaiting_approval.error
    assert bad.awaiting_approval.sub_questions == paused.awaiting_approval.sub_questions
    assert llm.calls_for("researcher") == []

    r = runner.resume("run-bad-edit", {"action": "approve"})
    assert r.outcome is not None and r.outcome.status == "answered"
    assert [sq.id for sq in r.sub_questions] == ["sq1", "sq2", "sq3"]
    with pytest.raises(ValueError):
        PlanDecision(action="edit")


def test_auto_approve_skips_the_checkpoint(search_tool):
    r = run_research(QUESTION, deps=GraphDeps(llm=fake_llm(search_tool), search_tool=search_tool), checkpointer=memory_checkpointer())
    assert r.outcome.status == "answered" and "plan_checkpoint" not in {s.node for s in r.trace}
    assert r.waited_s == 0


# --- Topic findings store ----------------------------------------------------------


def test_overlapping_question_reuses_fresh_topic_findings(open_stores, search_tool):
    clock = FakeClock()
    first, llm1 = process(open_stores(), search_tool, clock=clock)
    r1 = first.start(QUESTION, run_id="run-a")
    saved = [c for s in r1.trace for c in s.tool_calls if c.tool == "findings_store"]
    assert [c.state for c in saved] == ["saved"] * 3

    clock.advance(3600)
    search2 = ScriptedSearch(search_tool)
    second, llm2 = process(open_stores(), search_tool, search=search2, clock=clock)
    r2 = second.start("How many weekly sets should a trained woman do for muscle growth?", run_id="run-b")

    assert r2.outcome is not None and r2.outcome.status == "answered"
    assert llm2.calls_for("researcher") == [] and search2.queries == []  # no search, no extraction
    reused = [s for s in r2.trace if s.node == "researcher"]
    assert len(reused) == 3
    for step in reused:
        call = step.tool_calls[0]
        assert call.tool == "findings_store" and call.state == "reused" and "run run-a" in call.message
        assert step.cost_usd == 0 and step.llm_calls == []
    assert {f.source_id for f in r2.findings} == {f.source_id for f in r1.findings}
    assert all(f.id.startswith(f"{f.sub_question_id}:") for f in r2.findings)
    assert r2.spend.cost_usd < r1.spend.cost_usd and r2.spend.searches == 0
    events = [e for e in open_stores().events.list("run-b") if e.type == "step" and e.data["reused"]]
    assert len(events) == 3


def test_stale_topic_findings_are_researched_again(open_stores, search_tool):
    clock = FakeClock()
    first, _ = process(open_stores(), search_tool, clock=clock)
    first.start(QUESTION, run_id="run-old")
    clock.advance(31 * 86_400)  # past the 30-day TTL
    second, llm2 = process(open_stores(), search_tool, clock=clock)
    r = second.start(QUESTION, run_id="run-new")
    assert len(llm2.calls_for("researcher")) == 3
    assert not any(c.state == "reused" for s in r.trace for c in s.tool_calls)


def test_reuse_can_be_turned_off(search_tool):
    from backend.store.memory import memory_stores

    stores = memory_stores()
    Runner(GraphDeps(llm=fake_llm(search_tool), search_tool=search_tool), stores).start(QUESTION, run_id="a")
    llm = fake_llm(search_tool)
    Runner(GraphDeps(llm=llm, search_tool=search_tool), stores, reuse_findings=False).start(QUESTION, run_id="b")
    assert len(llm.calls_for("researcher")) == 3


def test_ledger_survives_a_broken_store(search_tool):
    """A ledger outage loses idempotency, never the run."""

    class Broken(MemoryCallLedger):
        def get(self, run_id, call_key):
            raise ConnectionError("db down")

        def put(self, entry):
            raise ConnectionError("db down")

    deps = GraphDeps(llm=fake_llm(search_tool), search_tool=search_tool, ledger=Broken())
    r = run_research(QUESTION, deps=deps, checkpointer=memory_checkpointer())
    assert r.outcome.status == "answered"

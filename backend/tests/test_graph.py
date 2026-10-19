"""End-to-end graph tests: FakeLLM + fixture search, fully offline."""

from __future__ import annotations

import time

import pytest

from backend import config
from backend.graph import Budget, GraphDeps, build_graph, memory_checkpointer, run_research
from backend.graph.builder import RunResult, initial_state
from backend.llm import FakeLLM, LLMError
from backend.tests.graph_fakes import (
    BAD_WRITER_DRAFT,
    FakeClock,
    ScriptedSearch,
    extractor,
    fake_llm,
    plan,
    writer_response,
)

QUESTION = "I'm a woman, lifting 3 years, 4 days a week. How many sets per week for hypertrophy?"


def run(llm: FakeLLM, search, budget: Budget | None = None, clock=None, question: str = QUESTION) -> RunResult:
    kw = {"clock": clock} if clock else {}
    deps = GraphDeps(llm=llm, search_tool=search, **kw)
    return run_research(question, deps=deps, budget=budget, checkpointer=memory_checkpointer())


def nodes(r: RunResult) -> list[str]:
    return [s.node for s in r.trace]


def researcher_steps(r: RunResult, sq: str | None = None):
    return [s for s in r.trace if s.node == "researcher" and (sq is None or s.sub_question_id == sq)]


def research_models(llm: FakeLLM, marker: str) -> list[str]:
    return [req.model for req in llm.calls_for("researcher") if f"Sub-question: {marker}" in req.user]


def degradation_kinds(r: RunResult) -> set[str]:
    return {d.kind for d in r.outcome.degradations}


# --- Happy path ---------------------------------------------------------------


def test_happy_path_end_to_end(search_tool):
    llm = fake_llm(search_tool)
    r = run(llm, search_tool)

    assert r.outcome.status == "answered" and r.outcome.failure is None
    assert nodes(r) == [
        "intake", "planner", "dispatch", "researcher", "researcher", "researcher",
        "review", "personalize", "position_mapper", "judge", "writer", "check_answer", "finalize",
    ]
    assert [s.index for s in r.trace] == list(range(len(r.trace)))
    assert {s.sub_question_id for s in researcher_steps(r)} == {"sq1", "sq2", "sq3"}

    # Routing by cost: Haiku extracts, Sonnet plans, maps, judges and writes, Opus is never used.
    models = {req.agent: set() for req in llm.requests}
    for req in llm.requests:
        models[req.agent].add(req.model)
    assert models == {
        "intake": {config.HAIKU}, "planner": {config.SONNET},
        "researcher": {config.HAIKU}, "position_mapper": {config.SONNET}, "judge": {config.SONNET},
        "writer": {config.SONNET},
    }

    # The answer only cites findings the run produced; all from peer-reviewed human studies.
    found = {f.id: f for f in r.findings}
    assert r.answer is not None and r.answer.all_cited_ids() <= found.keys()
    assert all(found[i].study_type not in ("animal", "preprint") for i in r.answer.cited_finding_ids)

    # Spend adds up across the trace, the ledger and the state, including parallel branches.
    assert r.spend.cost_usd == pytest.approx(sum(s.cost_usd for s in r.trace))
    assert r.spend.cost_usd == pytest.approx(llm.total_cost_usd)
    assert r.spend.llm_calls == len(llm.records) == 8
    assert r.spend.searches == sum(len(s.tool_calls) for s in researcher_steps(r)) == 5
    for s in researcher_steps(r):
        assert s.model == config.HAIKU and s.llm_calls and s.tool_calls and s.cost_usd > 0
        assert s.started_at <= s.finished_at and s.input_summary.startswith(s.sub_question_id)
    assert [rv.decision for rv in r.reviews] == ["proceed"]


def test_state_is_checkpointed_and_round_trips(search_tool, recwarn):
    graph = build_graph(GraphDeps(llm=fake_llm(search_tool), search_tool=search_tool), checkpointer=memory_checkpointer())
    cfg = {"configurable": {"thread_id": "run-1"}}
    graph.invoke(initial_state(QUESTION, None, Budget(), "run-1", time.time()), cfg)
    snap = graph.get_state(cfg)
    assert snap.values["outcome"].status == "answered"
    assert len(list(graph.get_state_history(cfg))) >= 8  # a checkpoint per superstep
    assert not [w for w in recwarn if "unregistered type" in str(w.message)]


# --- Out of scope -------------------------------------------------------------


def test_out_of_scope_rule_screen_short_circuits(search_tool):
    llm = FakeLLM()
    r = run(llm, search_tool, question="I'm pregnant. How much protein should I eat to build muscle?")
    assert r.outcome.status == "referred" and r.outcome.referral_message
    assert nodes(r) == ["intake", "referral"]
    assert llm.requests == [] and r.answer is None and r.plan is None
    assert r.trace[0].decision == "referral" and r.trace[0].model is None


def test_out_of_scope_by_model_never_reaches_planner(search_tool):
    intake = {"scope_category": "injury", "scope_reason": "asks about training through pain",
              "scoped_question": "Should I train through lower back pain?", "profile": {}}
    llm = fake_llm(search_tool, intake=intake)
    r = run(llm, search_tool, question="My lower back aches after deadlifts, should I keep going?")
    assert r.outcome.status == "referred"
    assert nodes(r) == ["intake", "referral"]
    assert llm.calls_for("planner") == []


# --- Budgets ------------------------------------------------------------------


def test_budget_max_sub_questions_caps_the_plan(search_tool):
    r = run(fake_llm(search_tool), search_tool, Budget(max_sub_questions=2))
    assert len(r.plan.sub_questions) == 3
    assert {s.sub_question_id for s in researcher_steps(r)} == {"sq1", "sq2"}
    assert r.outcome.status == "degraded" and r.answer is not None
    assert "max_sub_questions" in degradation_kinds(r)


def test_budget_max_searches_per_sub_question(search_tool):
    llm = fake_llm(search_tool, planner=plan("volume", "frequency", "empty"))
    r = run(llm, search_tool, Budget(max_searches_per_sub_question=1))
    assert all(len(s.tool_calls) == 1 for s in researcher_steps(r))
    assert len(researcher_steps(r)) == 3  # no second pass: sq3 has no searches left
    sq3 = next(x for x in r.reviews[0].sub_questions if x.sub_question_id == "sq3")
    assert not sq3.answered and sq3.blocked_by == "max_searches"
    assert {"max_searches", "sub_question_unanswered"} <= degradation_kinds(r)
    assert r.outcome.status == "degraded" and r.answer is not None


def test_budget_max_tokens_before_research_fails_cleanly(search_tool):
    budget = Budget(max_total_tokens=6_000, writer_reserve_fraction=0.9)
    llm = fake_llm(search_tool)
    r = run(llm, search_tool, budget)
    assert r.outcome.status == "failed"
    assert r.outcome.failure.reason == "budget_exhausted" and r.outcome.failure.budget_kind == "max_tokens"
    assert llm.calls_for("researcher") == [] and llm.calls_for("writer") == []
    assert r.spend.total_tokens <= budget.max_total_tokens
    assert nodes(r)[-1] == "finalize"


@pytest.mark.parametrize(
    ("budget", "node", "kind"),
    [
        (Budget(max_total_tokens=1_500), "intake", "max_tokens"),
        (Budget(max_usd=0.004), "planner", "max_usd"),  # Sonnet planner call cannot fit
    ],
)
def test_budget_exhausted_inside_a_node_fails_cleanly(search_tool, budget, node, kind):
    llm = fake_llm(search_tool)
    r = run(llm, search_tool, budget)
    assert r.outcome.status == "failed"
    assert (r.outcome.failure.reason, r.outcome.failure.node, r.outcome.failure.budget_kind) == (
        "budget_exhausted", node, kind,
    )
    assert r.spend.cost_usd <= budget.max_usd and r.spend.total_tokens <= budget.max_total_tokens
    assert llm.calls_for(node) == []  # refused before the call, so nothing was spent on it


def test_budget_max_usd_degrades_and_still_answers(search_tool):
    """A $0.05 run: a verbose researcher uses its share, so the Opus re-research is unaffordable,
    and the writer answers from the reserve."""
    budget = Budget.capped(0.05)
    llm = fake_llm(search_tool, ("volume", "frequency", "empty"), researcher=extractor(search_tool.records, use_max_tokens=True))
    r = run(llm, search_tool, budget)
    assert r.outcome.status == "degraded" and r.answer is not None
    sq3 = next(x for x in r.reviews[0].sub_questions if x.sub_question_id == "sq3")
    assert sq3.blocked_by == "max_usd" and not sq3.another_pass
    assert "max_usd" in degradation_kinds(r)
    assert len(r.reviews) == 1 and config.OPUS not in {req.model for req in llm.requests}
    assert r.spend.cost_usd <= budget.max_usd


def test_wall_clock_mid_research_degrades_and_still_answers(search_tool):
    clock = FakeClock()
    budget = Budget(wall_clock_s=100, writer_reserve_s=30)
    slow = extractor(search_tool.records, on_call=lambda _: clock.advance(80))
    llm = fake_llm(search_tool, ("volume", "frequency", "empty"), researcher=slow)
    r = run(llm, search_tool, budget, clock=clock)
    assert r.outcome.status == "degraded" and r.answer is not None
    sq3 = next(x for x in r.reviews[0].sub_questions if x.sub_question_id == "sq3")
    assert sq3.blocked_by == "wall_clock"
    assert "wall_clock" in degradation_kinds(r)


def test_wall_clock_before_research_fails_cleanly(search_tool):
    clock = FakeClock()

    def slow_planner(_req):
        clock.advance(500)
        return plan("volume", "plateau", "frequency")

    llm = fake_llm(search_tool, planner=slow_planner)
    r = run(llm, search_tool, Budget(wall_clock_s=300), clock=clock)
    assert r.outcome.status == "failed"
    assert r.outcome.failure.reason == "budget_exhausted" and r.outcome.failure.budget_kind == "wall_clock"
    assert llm.calls_for("researcher") == [] and llm.calls_for("writer") == []


# --- Send-back and escalation -------------------------------------------------


def test_reviewer_sends_back_exactly_once(search_tool):
    llm = fake_llm(search_tool, ("volume", "frequency", "empty"))
    r = run(llm, search_tool)
    assert [rv.decision for rv in r.reviews] == ["send_back", "proceed"]
    assert r.reviews[0].send_back_ids == ["sq3"]
    assert [s.attempt for s in researcher_steps(r, "sq2")] == [1]
    assert [s.attempt for s in researcher_steps(r, "sq3")] == [1, 2]
    assert [s.attempt for s in researcher_steps(r, "sq1")] == [1]
    # The second pass used new queries, and the second review could not send back again.
    second = researcher_steps(r, "sq3")[1]
    assert second.tool_calls and all(c.input != "creatine kidney function lifters" for c in second.tool_calls)
    sq3 = next(x for x in r.reviews[1].sub_questions if x.sub_question_id == "sq3")
    assert not sq3.answered and sq3.blocked_by == "send_back_used" and not sq3.another_pass
    assert r.outcome.status == "degraded" and r.answer is not None
    assert "sub_question_unanswered" in degradation_kinds(r)


def test_failed_review_escalates_re_research_to_opus(search_tool):
    llm = fake_llm(search_tool, ("volume", "frequency", "animal"))
    r = run(llm, search_tool)
    sq3_review = next(x for x in r.reviews[0].sub_questions if x.sub_question_id == "sq3")
    assert sq3_review.issues == ["only_animal_or_in_vitro"] and sq3_review.escalate
    assert all("humans randomized controlled trial" in q for q in sq3_review.refined_queries)
    assert research_models(llm, "Does progressive overload") == [config.HAIKU, config.OPUS]
    assert research_models(llm, "Is there a dose-response") == [config.HAIKU]
    second = researcher_steps(r, "sq3")[1]
    assert second.escalated and second.model == config.OPUS and "only_animal_or_in_vitro" in second.route_reason
    assert {req.model for req in llm.calls_for("writer")} == {config.SONNET}
    assert r.answer is not None


def test_population_mismatch_sends_back_with_profile_qualifiers(search_tool):
    # Every human finding for sq1 studied men; the user is a trained woman.
    men_only = {f"FIXTURE-{i:04d}": "30 resistance-trained men" for i in range(1, 12) if i != 7}
    llm = fake_llm(search_tool, ("volume", "frequency", "plateau"), researcher=extractor(search_tool.records, population=men_only))
    r = run(llm, search_tool)
    first = r.reviews[0]
    assert first.decision == "send_back"
    sq1 = next(x for x in first.sub_questions if x.sub_question_id == "sq1")
    assert "population_mismatch" in sq1.issues and sq1.answered
    assert all(q.endswith("trained women") for q in sq1.refined_queries)
    assert "studied men only; user is female" in set(sq1.population_mismatches.values())
    assert "population_mismatch" in degradation_kinds(r)  # still mismatched after the one pass
    assert r.answer is not None


# --- Researcher failures --------------------------------------------------------


def test_transient_search_failure_is_retried_without_escalation(search_tool):
    search = ScriptedSearch(search_tool, failures={"plateau": ("rate_limited", 1), "diminishing": ("rate_limited", 1)})
    llm = fake_llm(search_tool)
    r = run(llm, search)
    states = [s.output["status"]["state"] for s in researcher_steps(r, "sq2")]
    assert states == ["rate_limited", "ok"]
    sq2 = next(x for x in r.reviews[0].sub_questions if x.sub_question_id == "sq2")
    assert "search_failed" in sq2.issues and not sq2.escalate
    assert set(sq2.refined_queries) == {
        "high volume resistance training plateau hypertrophy", "diminishing returns weekly sets muscle growth",
    }
    second = researcher_steps(r, "sq2")[1]
    assert second.model == config.HAIKU and not second.escalated
    assert r.outcome.status == "answered"


def test_crashing_search_tool_is_a_status_not_an_exception(search_tool):
    search = ScriptedSearch(search_tool, raise_on="equated")
    r = run(fake_llm(search_tool), search)
    sq3 = researcher_steps(r, "sq3")
    assert sq3[0].output["status"]["state"] == "error" and "RuntimeError" in sq3[0].tool_calls[0].message
    assert r.answer is not None and r.outcome.status == "degraded"


def test_all_research_failing_ends_with_structured_failure(search_tool):
    search = ScriptedSearch(search_tool, failures={"": ("timed_out", None)})
    llm = fake_llm(search_tool)
    r = run(llm, search)
    assert [rv.decision for rv in r.reviews] == ["send_back", "fail"]
    assert r.outcome.status == "failed" and r.outcome.failure.reason == "insufficient_evidence"
    assert llm.calls_for("writer") == [] and nodes(r)[-1] == "finalize"


# --- Writer failures --------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "error", "synthesis"),
    [
        ("not json", "StructuredOutputError", True),  # with a verdict: invalid output
        (BAD_WRITER_DRAFT, "WriterError", False),  # without one (mapper down): no position backed by a finding
    ],
)
def test_writer_error_triggers_one_escalated_rewrite(search_tool, bad, error, synthesis):
    over = {} if synthesis else {"position_mapper": LLMError("overloaded")}
    llm = fake_llm(search_tool, writer=[bad, writer_response()], **over)
    r = run(llm, search_tool)
    writes = llm.calls_for("writer")
    assert [w.model for w in writes] == [config.SONNET, config.OPUS]
    assert "rejected by review" in writes[1].user and "rejected by review" not in writes[0].user
    assert [a.decision for a in r.answer_reviews] == ["rewrite", "accept"]
    assert error in r.answer_reviews[0].problems[0]
    assert r.outcome.status == ("answered" if synthesis else "degraded") and r.answer is not None
    assert (r.verdict is not None) == synthesis
    assert [s.node for s in r.trace][-5:] == ["writer", "check_answer", "writer", "check_answer", "finalize"]


def test_writer_failing_twice_fails_cleanly(search_tool):
    llm = fake_llm(search_tool, writer=LLMError("overloaded"))
    r = run(llm, search_tool)
    assert len(llm.calls_for("writer")) == 2
    assert r.outcome.status == "failed" and r.outcome.failure.reason == "writer_failed"
    assert r.answer is None and r.findings  # the evidence gathered is still returned


def test_writer_error_after_research_send_back_is_not_retried(search_tool):
    llm = fake_llm(search_tool, ("volume", "frequency", "empty"), writer="not json")
    r = run(llm, search_tool)
    assert r.reviews[0].decision == "send_back"
    assert len(llm.calls_for("writer")) == 1
    assert r.outcome.failure.reason == "writer_failed"


def test_answer_resting_on_preprints_is_rewritten(search_tool):
    # Without a verdict (mapper down) the writer picks its own citations; review catches preprint-only ones.
    llm = fake_llm(
        search_tool, writer=[writer_response(only_types={"preprint"}), writer_response()],
        position_mapper=LLMError("overloaded"),
    )
    r = run(llm, search_tool)
    assert "preprint" in r.answer_reviews[0].problems[0]
    assert [a.decision for a in r.answer_reviews] == ["rewrite", "accept"]
    assert r.outcome.status == "degraded" and degradation_kinds(r) == {"mapping_skipped"}

"""Unit tests for the deterministic pieces of the graph: budgets, routing, supervisor checks."""

from __future__ import annotations

import pytest

from backend import config
from backend.agents.researcher import ExtractionBatch
from backend.graph.budget import (
    MIN_OUTPUT_TOKENS,
    Allowance,
    Budget,
    BudgetExceeded,
    MeteredLLM,
    MeteredSearch,
    Spend,
    affordable_count,
    research_pool,
)
from backend.graph.routing import ModelRouter
from backend.graph.state import PassRecord, ResearchRecord, merge_research
from backend.graph.supervisor import population_mismatch, refine_queries, review_answer, review_sub_question
from backend.graph.trace import StepTimer, append_steps
from backend.llm import FakeLLM, LLMError, LLMRequest
from backend.schemas import CoachAnswer, DISCLAIMER, Position, SearchStatus, SubQuestion, UserProfile
from backend.tests.conftest import make_finding

TRAINED_WOMAN = UserProfile(sex="female", training_status="intermediate", age=28)


def _request(model: str = config.HAIKU, max_tokens: int = 16_000) -> LLMRequest:
    return LLMRequest(agent="researcher", system="s" * 700, user="u" * 700, output_type=ExtractionBatch,
                      model=model, max_tokens=max_tokens)


# --- Budget -----------------------------------------------------------------


def test_metered_llm_clamps_max_tokens_to_what_the_money_allows():
    llm = FakeLLM({"researcher": {"studies": []}})
    metered = MeteredLLM(llm, Allowance(usd=0.02, tokens=1_000_000, deadline=float("inf")))
    metered.complete(_request(model=config.SONNET))
    sent = llm.requests[0].max_tokens
    assert MIN_OUTPUT_TOKENS <= sent < 2_000  # $0.02 buys < 2k Sonnet output tokens
    assert metered.records and metered.spend.cost_usd > 0


@pytest.mark.parametrize(
    ("allowance", "kind"),
    [
        (Allowance(usd=0.0001, tokens=1_000_000, deadline=float("inf")), "max_usd"),
        (Allowance(usd=10, tokens=1_000, deadline=float("inf")), "max_tokens"),
        (Allowance(usd=10, tokens=1_000_000, deadline=0.0), "wall_clock"),
    ],
)
def test_metered_llm_refuses_without_calling(allowance, kind):
    llm = FakeLLM({"researcher": {"studies": []}})
    metered = MeteredLLM(llm, allowance)
    with pytest.raises(BudgetExceeded) as exc:
        metered.complete(_request())
    assert exc.value.kind == kind and metered.exhausted == kind and llm.requests == []


def test_metered_llm_keeps_spend_of_failed_calls():
    llm = FakeLLM({"researcher": "not json"})
    metered = MeteredLLM(llm, Allowance(usd=1, tokens=1_000_000, deadline=float("inf")))
    with pytest.raises(LLMError):
        metered.complete(_request())
    assert len(metered.records) == 1 and not metered.records[0].ok and metered.spend.input_tokens > 0


def test_metered_search_caps_and_logs(search_tool):
    search = MeteredSearch(search_tool, max_searches=1, deadline=float("inf"))
    assert search.search("resistance training volume hypertrophy meta-analysis").state == "ok"
    refused = search.search("weekly sets muscle thickness trained")
    assert refused.state == "error" and search.exhausted == "max_searches"
    assert [c.state for c in search.calls] == ["ok", "refused"] and search.executed == 1


def test_research_pool_keeps_the_writer_reserve():
    budget = Budget(max_usd=1.0, max_total_tokens=100_000, writer_reserve_fraction=0.3, wall_clock_s=300, writer_reserve_s=60)
    pool, kind = research_pool(budget, Spend(cost_usd=0.2, input_tokens=10_000), started_at=0, now=10)
    assert kind is None and pool.usd == pytest.approx(0.5) and pool.tokens == 60_000 and pool.deadline == 240
    assert research_pool(budget, Spend(cost_usd=0.7), 0, 10) == (None, "max_usd")
    assert research_pool(budget, Spend(), 0, 241) == (None, "wall_clock")


def test_affordable_count():
    pool = Allowance(usd=0.03, tokens=1_000_000, deadline=1)
    assert affordable_count(pool, config.HAIKU, 3) == (3, None)
    assert affordable_count(pool, config.OPUS, 3) == (1, "max_usd")


# --- Routing ----------------------------------------------------------------


def test_router_routes_by_cost_and_escalates_only_on_request():
    router = ModelRouter()
    assert router.route("intake").model == config.HAIKU
    assert router.route("researcher").model == config.HAIKU
    assert router.route("planner").model == config.SONNET
    assert router.route("writer").model == config.SONNET
    up = router.route("researcher", escalate=True)
    assert up.model == config.OPUS and up.escalated
    assert router.route("planner", escalate=True).model == config.SONNET  # no escalation path
    with pytest.raises(KeyError):
        router.route("nobody")


# --- Supervisor -------------------------------------------------------------


@pytest.mark.parametrize(
    ("population", "expected"),
    [
        ("34 resistance-trained men", ["studied men only; user is female"]),
        ("28 untrained young women", ["studied untrained people; user is trained"]),
        ("40 trained adults (22 men, 18 women)", []),
        ("2066 adults aged 70-79", ["studied older adults; user is 28"]),
        ("67 studies, 2058 participants", []),
    ],
)
def test_population_mismatch(population, expected):
    assert population_mismatch(TRAINED_WOMAN, population) == expected


def test_population_mismatch_needs_a_profile():
    assert population_mismatch(None, "34 trained men") == []
    assert population_mismatch(UserProfile(), "34 trained men") == []


def _record(findings, state="ok", queries=("q1",), failed=(), records_seen=5) -> ResearchRecord:
    status = SearchStatus(sub_question_id="sq1", state=state, records_seen=records_seen)
    return ResearchRecord(
        sub_question_id="sq1",
        findings=findings,
        passes=[PassRecord(pass_no=1, model=config.HAIKU, escalated=False, queries=list(queries),
                           failed_queries=list(failed), status=status, finding_ids=[f.id for f in findings])],
        searches_used=len(queries),
        queries_run=list(queries),
    )


SQ = SubQuestion(id="sq1", question="Does volume matter for hypertrophy in trained women?", rationale="r",
                 search_terms=["q1", "q2"])


def test_review_flags_animal_only_and_escalates():
    rec = _record([make_finding("sq1:A", study_type="animal", grade="very_low")])
    review = review_sub_question(SQ, rec, TRAINED_WOMAN, searches_left=3, send_back_available=True, budget_block=None)
    assert not review.answered and review.issues == ["only_animal_or_in_vitro"]
    assert review.another_pass and review.escalate
    assert review.refined_queries == ["q1 humans randomized controlled trial", "q2 humans randomized controlled trial"]


def test_review_blocks_in_priority_order():
    rec = _record([])
    kw = dict(searches_left=3, send_back_available=True, budget_block=None)
    assert review_sub_question(SQ, rec, None, **(kw | {"send_back_available": False})).blocked_by == "send_back_used"
    assert review_sub_question(SQ, rec, None, **(kw | {"budget_block": "max_usd"})).blocked_by == "max_usd"
    assert review_sub_question(SQ, rec, None, **(kw | {"searches_left": 0})).blocked_by == "max_searches"
    ok = review_sub_question(SQ, rec, None, **kw)
    assert ok.another_pass and ok.refined_queries == ["q2", SQ.question.rstrip("?")]


def test_refine_queries_retries_failed_and_never_repeats():
    rec = _record([], state="rate_limited", queries=("q1", "q2"), failed=("q2",), records_seen=0)
    queries = refine_queries(SQ, rec, None, ["no_findings", "search_failed"], limit=5)
    assert queries == ["q2", SQ.question.rstrip("?")]
    assert refine_queries(SQ, rec, None, ["no_findings", "search_failed"], limit=1) == ["q2"]


def test_merge_research_combines_passes():
    a = _record([make_finding("sq1:A")])
    b = _record([make_finding("sq1:A", claim="newer"), make_finding("sq1:B")], queries=("q3",))
    merged = merge_research({"sq1": a}, {"sq1": b})["sq1"]
    assert {f.id for f in merged.findings} == {"sq1:A", "sq1:B"}
    assert next(f for f in merged.findings if f.id == "sq1:A").claim == "newer"
    assert len(merged.passes) == 2 and merged.searches_used == 2 and merged.queries_run == ["q1", "q3"]


def _answer(ids: list[str], confidence: str = "moderate") -> CoachAnswer:
    return CoachAnswer(
        status="consensus", recommendation="r", confidence=confidence, why="w",
        positions=[Position(stance="s", supporting_findings=ids, evidence_grade="moderate", best_counterargument="c")],
        what_would_change_this="x", applies_to_you="x", practical_protocol="x", whats_new=None,
        cited_finding_ids=ids, disclaimer=DISCLAIMER,
    )


def test_review_answer():
    rct = make_finding("sq1:R", study_type="rct", grade="moderate")
    pre = make_finding("sq1:P", study_type="preprint", grade="low")
    assert review_answer(_answer(["sq1:R"]), [rct, pre], None) == []
    assert "preprint" in review_answer(_answer(["sq1:P"]), [rct, pre], None)[0]
    low = make_finding("sq1:L", study_type="rct", grade="low")
    assert "confidence is high" in review_answer(_answer(["sq1:L"], "high"), [low], None)[0]
    assert review_answer(None, [rct], "WriterError: boom") == ["writer failed: WriterError: boom"]


# --- Trace ------------------------------------------------------------------


def test_append_steps_numbers_in_order():
    t = StepTimer("run", "x")
    steps = append_steps([], [t.step(input_summary="a", output_summary="a")])
    steps = append_steps(steps, [t.step(input_summary="b", output_summary="b"), t.step(input_summary="c", output_summary="c")])
    assert [s.index for s in steps] == [0, 1, 2] and len({s.step_id for s in steps}) == 3

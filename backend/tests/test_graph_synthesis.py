"""Phase 4 in the graph: personalization, position mapping, debate and judgment. FakeLLM + fixture search."""

from __future__ import annotations

import time

import pytest

from backend import config
from backend.graph import Budget, GraphDeps, build_graph, memory_checkpointer, run_research
from backend.graph.builder import RunResult, after_debate_review, initial_state
from backend.graph.state import DEBATE_ROUNDS
from backend.llm import FakeLLM, LLMError, LLMRequest
from backend.tests.graph_fakes import (
    FakeClock,
    advocate_response,
    extractor,
    fake_llm,
    judge_response,
    mapper_response,
    plan,
)

QUESTION = "I'm a woman, lifting 3 years, 4 days a week. Should I take my sets to failure?"
CREATINE = "I'm a woman, lifting 3 years, 4 days a week. Is creatine safe?"

# Training to failure: two high-grade meta-analyses (and a trial each) disagree.
FAILURE_SIDES = {"FIXTURE-0012": 1, "FIXTURE-0014": 1, "FIXTURE-0013": 2, "FIXTURE-0015": 2}


def by_source(sides: dict[str, int]):
    return lambda fid, _st: sides.get(fid.split(":")[1], 0)


def contested_llm(search_tool, **over) -> FakeLLM:
    responses = {
        "planner": plan("failure_similar", "failure_proximity", "failure_rir"),
        "position_mapper": mapper_response("genuinely_contested", by_source(FAILURE_SIDES)),
        "judge": judge_response(recommended=1, status="genuinely_contested", confidence="high"),
    }
    return fake_llm(search_tool, **(responses | over))


def creatine_llm(search_tool, **over) -> FakeLLM:
    """The mapper sees a 'harm' side made of a case report and a rat study and calls it contested."""
    harm = lambda _fid, st: 2 if st in ("case_report", "animal") else 1  # noqa: E731
    responses = {
        "planner": plan("creatine_safety", "creatine_longterm", "creatine_reports"),
        "position_mapper": mapper_response("genuinely_contested", harm),
    }
    return fake_llm(search_tool, **(responses | over))


def run(llm: FakeLLM, search, question: str = QUESTION, budget: Budget | None = None, clock=None) -> RunResult:
    deps = GraphDeps(llm=llm, search_tool=search, **({"clock": clock} if clock else {}))
    return run_research(question, deps=deps, budget=budget, checkpointer=memory_checkpointer())


def nodes(r: RunResult) -> list[str]:
    return [s.node for s in r.trace]


def steps(r: RunResult, node: str):
    return [s for s in r.trace if s.node == node]


def kinds(r: RunResult) -> set[str]:
    return {d.kind for d in r.outcome.degradations}


# --- Consensus -------------------------------------------------------------------


def test_consensus_skips_the_debate_and_false_balance_is_downgraded(search_tool):
    """'Is creatine safe?' must not become a debate: the harm side has only a case report and a rat study."""
    llm = creatine_llm(search_tool)
    r = run(llm, search_tool, CREATINE)

    assert r.outcome.status == "answered", r.outcome
    assert "advocate" not in nodes(r) and "debate_review" not in nodes(r)
    assert nodes(r)[-6:] == ["personalize", "position_mapper", "judge", "writer", "check_answer", "finalize"]
    assert llm.calls_for("advocate") == []

    pm = r.position_map
    assert pm.proposed_status == "genuinely_contested" and pm.status == "consensus"
    assert len(pm.positions) == 1 and pm.positions[0].evidence_grade == "high"
    assert len(pm.set_aside) == 1 and pm.set_aside[0].evidence_grade == "very_low"
    assert any("false balance" in reason for reason in pm.status_reasons)
    mapper_step = steps(r, "position_mapper")[0]
    assert mapper_step.decision == "judge" and "consensus: debate skipped" in mapper_step.output_summary

    # The judge still sees the set-aside side and every finding with its grade.
    judge_prompt = llm.calls_for("judge")[0].user
    assert "Set aside as too weak" in judge_prompt and "(case_report, grade very_low" in judge_prompt
    assert "No debate was held" in judge_prompt
    assert r.verdict.status == "consensus" and r.answer.status == "consensus" and len(r.answer.positions) == 1


# --- Contested ---------------------------------------------------------------------


def test_contested_runs_the_debate_with_exactly_one_rebuttal_round(search_tool):
    llm = contested_llm(search_tool)
    r = run(llm, search_tool)
    assert r.outcome.status == "answered", r.outcome

    assert r.position_map.status == "genuinely_contested" and len(r.position_map.positions) == 2
    assert [p.evidence_grade for p in r.position_map.positions] == ["high", "high"]
    assert nodes(r)[-11:] == [
        "position_mapper", "advocate", "advocate", "debate_review", "advocate", "advocate", "debate_review",
        "judge", "writer", "check_answer", "finalize",
    ]
    # Each position argued exactly twice: an opening, then one rebuttal.
    rounds = sorted((t.position, t.round) for t in r.debate)
    assert rounds == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert all(t.argument is not None and not t.rejected_claims for t in r.debate)
    rebuttals = [t for t in r.debate if t.round == 2]
    assert {t.argument.rebuttal_of.split(":")[0] for t in rebuttals} == {"P1", "P2"}
    assert [s.attempt for s in steps(r, "advocate")] == [1, 1, 2, 2]
    assert len(llm.calls_for("advocate")) == 2 * DEBATE_ROUNDS
    rebuttal_prompts = [c.user for c in llm.calls_for("advocate") if "Write your rebuttal." in c.user]
    assert len(rebuttal_prompts) == 2 and all("Opening arguments of the other sides" in u for u in rebuttal_prompts)

    # The judge saw openings, rebuttals and the graded findings, and its confidence was capped in code.
    judge_prompt = llm.calls_for("judge")[0].user
    assert "P1 opening" in judge_prompt and "P2 rebuttal" in judge_prompt and "grade high" in judge_prompt
    v = r.verdict
    assert v.status == "genuinely_contested" and v.confidence == "moderate" and v.confidence_cap == "moderate"
    assert any("contested evidence caps confidence" in a for a in v.adjustments)

    # The writer explains the verdict; the answer shows both sides.
    assert r.answer.status == "genuinely_contested" and r.answer.confidence == "moderate"
    assert len(r.answer.positions) == 2 and r.answer.all_cited_ids() <= {f.id for f in r.findings}
    assert "judge's verdict is final" in llm.calls_for("writer")[0].user

    # Every synthesis call ran on Sonnet and is accounted for in the trace and the spend.
    assert {req.model for req in llm.requests if req.agent in ("position_mapper", "advocate", "judge")} == {config.SONNET}
    assert r.spend.cost_usd == pytest.approx(sum(s.cost_usd for s in r.trace)) == pytest.approx(llm.total_cost_usd)
    assert [s.index for s in r.trace] == list(range(len(r.trace)))


def test_debate_round_limit_is_enforced_by_routing():
    state = {"debate_rounds": DEBATE_ROUNDS, "pending_advocates": ["would-be third round"]}
    assert after_debate_review(state) == "judge"


def test_contested_run_is_checkpointed(search_tool, recwarn):
    graph = build_graph(GraphDeps(llm=contested_llm(search_tool), search_tool=search_tool), checkpointer=memory_checkpointer())
    cfg = {"configurable": {"thread_id": "debate-1"}}
    graph.invoke(initial_state(QUESTION, None, Budget(), "debate-1", time.time()), cfg)
    values = graph.get_state(cfg).values
    assert values["outcome"].status == "answered" and len(values["debate"]) == 4 and values["verdict"] is not None
    assert not [w for w in recwarn if "unregistered type" in str(w.message)]


# --- Advocates argue from findings only ----------------------------------------------


def test_advocate_uncited_claims_are_rejected_and_never_reach_the_judge(search_tool):
    bad = [
        {"text": "Everyone at my gym trains to failure.", "finding_ids": []},
        {"text": "A 2025 study proves it.", "finding_ids": ["sq9:FIXTURE-9999"]},
    ]
    llm = contested_llm(search_tool, advocate=advocate_response(extra_claims=bad))
    r = run(llm, search_tool)
    assert r.outcome.status == "answered"
    for t in r.debate:
        assert t.argument is not None
        assert {c.reason.split(":")[0] for c in t.rejected_claims} == {"cites no finding id", "cites unknown finding ids"}
        assert t.argument.cited_ids() <= {f.id for f in r.findings}
    assert all(s.status == "degraded" and "2 claim(s) rejected" in s.output_summary for s in steps(r, "advocate"))
    judge_prompt = llm.calls_for("judge")[0].user
    assert "Everyone at my gym" not in judge_prompt and "FIXTURE-9999" not in judge_prompt


def test_failed_opening_skips_the_rebuttal_round(search_tool):
    def advocate(request: LLMRequest):
        if "Your position (P2)" in request.user:
            return LLMError("overloaded")
        return advocate_response()(request)

    r = run(contested_llm(search_tool, advocate=advocate), search_tool)
    assert [(t.position, t.round, t.argument is not None) for t in sorted(r.debate, key=lambda t: t.position)] == [
        (1, 1, True), (2, 1, False),
    ]
    assert {"advocate_failed", "rebuttal_skipped"} <= kinds(r)
    assert r.outcome.status == "degraded" and r.verdict is not None and r.answer is not None


# --- Personalization ------------------------------------------------------------------


def test_downweighting_is_recorded_in_the_trace(search_tool):
    """The default fixtures studied men only (FIXTURE-0002) and untrained women (FIXTURE-0005); the user
    is a trained woman. A rat study is non-human."""
    llm = fake_llm(search_tool, ("volume", "frequency", "animal"))
    r = run(llm, search_tool)
    step = steps(r, "personalize")[0]
    down = step.output["downweighted"]
    assert down["sq1:FIXTURE-0002"] == "studied men only; user is female"
    assert down["sq1:FIXTURE-0005"] == "studied untrained people; user is trained"
    assert down["sq3:FIXTURE-0007"] == "non-human study (animal)"
    assert "sq1:FIXTURE-0001" not in down  # mixed trained / untrained men and women
    assert step.output["weight"] == 0.5 and f"{len(down)} of" in step.output_summary
    assert r.downweighted == down and r.verdict.downweighted_findings == down
    # The mapper and the judge see the label; the findings are kept, not dropped.
    for agent in ("position_mapper", "judge"):
        assert "down-weighted: studied men only; user is female" in llm.calls_for(agent)[0].user
    assert {"sq1:FIXTURE-0002", "sq1:FIXTURE-0005"} <= {f.id for f in r.findings}


def test_downweighting_lowers_position_grades_and_confidence(search_tool):
    """Everything was studied in men: the trained woman gets indirect evidence and a lower cap."""
    men = {f"FIXTURE-{i:04d}": "trained men" for i in (12, 13, 14, 15)}
    llm = contested_llm(search_tool, researcher=extractor(search_tool.records, population=men))
    r = run(llm, search_tool)
    assert all(f.id in r.downweighted for f in r.findings if f.source_id in men)
    assert [p.evidence_grade for p in r.position_map.positions] == ["moderate", "moderate"]
    assert r.verdict.confidence_cap == "moderate" and r.verdict.confidence == "moderate"


# --- Budgets ---------------------------------------------------------------------------


def test_budget_that_cannot_afford_the_debate_skips_it(search_tool):
    llm = contested_llm(search_tool)
    budget = Budget(max_usd=0.10)
    r = run(llm, search_tool, budget=budget)
    assert r.position_map is not None and r.position_map.status == "genuinely_contested"
    assert "advocate" not in nodes(r) and llm.calls_for("advocate") == []
    deg = next(d for d in r.outcome.degradations if d.kind == "debate_skipped")
    assert deg.detail.startswith("max_usd") and deg.node == "position_mapper"
    assert steps(r, "position_mapper")[0].decision == "judge"
    assert r.verdict is not None and r.answer is not None and r.outcome.status == "degraded"
    assert r.spend.cost_usd <= budget.max_usd


def test_wall_clock_skips_the_debate(search_tool):
    clock = FakeClock()

    def slow_mapper(request: LLMRequest):
        clock.advance(150)  # the run now has ~150s left: enough for the judge and writer, not a debate
        return mapper_response("genuinely_contested", by_source(FAILURE_SIDES))(request)

    r = run(contested_llm(search_tool, position_mapper=slow_mapper), search_tool, clock=clock)
    assert "advocate" not in nodes(r)
    assert any(d.kind == "debate_skipped" and d.detail.startswith("wall_clock") for d in r.outcome.degradations)
    assert r.verdict is not None and r.answer is not None


def test_tiny_budget_skips_synthesis_but_still_answers(search_tool):
    llm = contested_llm(search_tool)
    r = run(llm, search_tool, budget=Budget(max_usd=0.05))
    assert steps(r, "position_mapper")[0].status == "skipped" and llm.calls_for("position_mapper") == []
    assert "mapping_skipped" in kinds(r) and r.verdict is None
    assert r.answer is not None and r.outcome.status == "degraded"


# --- Synthesis failures degrade, never fail the run -------------------------------------


def test_judge_failure_falls_back_to_the_position_map(search_tool):
    # The judge recommends a position that does not exist: rejected in code, the writer uses the map.
    r = run(contested_llm(search_tool, judge=judge_response(recommended=7)), search_tool)
    assert r.verdict is None and "judge_skipped" in kinds(r)
    j = steps(r, "judge")[0]
    assert j.status == "failed" and "JudgeError" in j.error and j.decision == "writer"
    assert r.answer is not None and r.answer.status == "genuinely_contested"


def test_mapper_failure_falls_back_to_the_writer(search_tool):
    r = run(contested_llm(search_tool, position_mapper=LLMError("overloaded")), search_tool)
    assert nodes(r)[-5:] == ["personalize", "position_mapper", "writer", "check_answer", "finalize"]
    assert r.position_map is None and "mapping_skipped" in kinds(r) and r.answer is not None

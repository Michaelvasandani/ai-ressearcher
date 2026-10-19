"""Live position mapping, debate and judgment: real models, fixture search. Run with:
RUN_LIVE=1 pytest -m live -s tests/live/test_live_synthesis.py

Two questions: one the synthetic fixtures make genuinely contested (training to
failure: two meta-analyses disagree) and one that must not become a debate
(creatine safety: trials and a meta-analysis against a case report and a rat
study). The search is scoped to each topic's fixture studies, with a looser
keyword match, because the live planner writes its own queries. Each run is
capped by its own budget (a run costs roughly $0.10-0.20).
"""

from __future__ import annotations

import pytest

from backend.graph import Budget, GraphDeps, RunResult, memory_checkpointer, run_research
from backend.llm import AnthropicLLM
from backend.tools.search import FixtureSearchTool

pytestmark = pytest.mark.live

FAILURE_STUDIES = {"FIXTURE-0012", "FIXTURE-0013", "FIXTURE-0014", "FIXTURE-0015"}
CREATINE_STUDIES = {"FIXTURE-0016", "FIXTURE-0017", "FIXTURE-0018", "FIXTURE-0019"}
BUDGET = Budget(max_usd=0.40, max_sub_questions=3, wall_clock_s=300)


def _run(question: str, studies: set[str]) -> RunResult:
    search = FixtureSearchTool.from_file(only=studies, min_fraction=0.34)
    r = run_research(
        question,
        deps=GraphDeps(llm=AnthropicLLM(), search_tool=search),
        budget=BUDGET,
        checkpointer=memory_checkpointer(),
    )
    print(f"\n{question}\noutcome: {r.outcome.status} - {r.outcome.message}")
    for s in r.trace:
        print(
            f"{s.index:>2} {s.node:<15} {s.attempt} {s.model or '':<18} {s.status:<8} ${s.cost_usd:.4f} "
            f"{s.latency_ms:>6}ms  {s.output_summary[:110]}"
        )
    if r.position_map:
        print("map:", r.position_map.status, "| proposed", r.position_map.proposed_status)
        for p in r.position_map.positions + r.position_map.set_aside:
            print(f"  - [{p.evidence_grade}] {p.stance} {p.supporting_findings}")
        for reason in r.position_map.status_reasons:
            print("   ·", reason)
    for t in r.debate:
        n = len(t.argument.claims) if t.argument else 0
        print(f"  debate P{t.position} round {t.round}: {n} claims, {len(t.rejected_claims)} rejected {t.error or ''}")
    if r.verdict:
        print("verdict:", r.verdict.status, r.verdict.confidence, "cap", r.verdict.confidence_cap, r.verdict.adjustments)
        print("  ", r.verdict.recommendation)
    print(f"spend: ${r.spend.cost_usd:.4f}, {r.spend.total_tokens} tokens, {r.spend.llm_calls} calls")

    assert r.outcome.status in ("answered", "degraded"), r.outcome
    assert r.answer is not None and r.answer.all_cited_ids() <= {f.id for f in r.findings}
    assert r.spend.cost_usd <= BUDGET.max_usd
    assert [s.index for s in r.trace] == list(range(len(r.trace)))
    assert r.position_map is not None and r.verdict is not None
    return r


def test_live_contested_question_is_debated():
    r = _run(
        "I'm a 30 year old man, lifting for 4 years. Should I take my sets to failure for muscle growth?",
        FAILURE_STUDIES,
    )
    assert r.position_map.status in ("genuinely_contested", "leaning")
    n = len(r.position_map.positions)
    assert n >= 2
    assert sorted(t.round for t in r.debate) == [1] * n + [2] * n  # openings, then exactly one rebuttal round
    assert r.verdict.confidence != "high"  # contested evidence caps confidence at moderate
    assert len(r.answer.positions) >= 2 or r.answer.status == "consensus"


def test_live_creatine_safety_is_not_a_debate():
    r = _run(
        "I'm a 30 year old man, lifting for 4 years. Is creatine safe to take long term?",
        CREATINE_STUDIES,
    )
    assert r.position_map.status == "consensus"
    assert r.debate == [] and "advocate" not in {s.node for s in r.trace}
    assert r.answer.status == "consensus"

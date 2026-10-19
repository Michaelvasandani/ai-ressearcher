"""Live prompt checks against the real API. Run with: RUN_LIVE=1 pytest -m live -s

Kept to a handful of small calls (a few cents in total). Assertions are about
shape and obvious quality bars, not exact wording.
"""

from __future__ import annotations

import pytest

from backend.agents import IntakeAgent, PlannerAgent, ResearcherAgent, WriterAgent
from backend.agents.planner import lint_sub_question
from backend.llm import AnthropicLLM
from backend.schemas import UserProfile
from backend.tools.search import FixtureSearchTool

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def llm() -> AnthropicLLM:
    return AnthropicLLM()


@pytest.fixture(scope="module")
def sets_intake(llm):
    return IntakeAgent(llm).run(
        "I'm a 28 year old woman, lifting for about 3 years, 4 days a week. How many sets per week for hypertrophy?"
    )


def test_intake_in_scope_and_profile(sets_intake):
    print("\n", sets_intake.model_dump_json(indent=2))
    assert sets_intake.in_scope
    p = sets_intake.profile
    assert p.sex == "female" and p.age == 28 and p.days_per_week == 4
    assert p.training_status in ("intermediate", "advanced")


@pytest.mark.parametrize(
    "question",
    [
        "My lower back hurts after deadlifts, should I keep training through it?",
        "I've been making myself throw up after big meals, how do I still gain muscle?",
    ],
)
def test_intake_refers_out_of_scope(llm, question):
    intake = IntakeAgent(llm).run(question)
    print("\n", intake.scope_category, intake.classified_by, intake.scope_reason)
    assert not intake.in_scope and intake.referral_message


def test_planner_sets_question(llm, sets_intake):
    plan = PlannerAgent(llm).run(sets_intake)
    print("\n", plan.model_dump_json(indent=2))
    assert 3 <= len(plan.sub_questions) <= 6
    flagged = {sq.id: lint_sub_question(sq) for sq in plan.sub_questions if lint_sub_question(sq)}
    assert not flagged, flagged
    text = " ".join(sq.question.lower() for sq in plan.sub_questions)
    assert "set" in text or "volume" in text


def test_researcher_and_writer_on_fixtures(llm, sets_intake):
    from backend.schemas import SubQuestion

    sq = SubQuestion(
        id="sq1",
        question="Is there a dose-response relationship between weekly sets per muscle and hypertrophy in resistance-trained adults?",
        rationale="core question",
        search_terms=["resistance training volume hypertrophy meta-analysis", "weekly sets muscle thickness trained"],
    )
    tool = FixtureSearchTool.from_file()
    result = ResearcherAgent(llm, tool).run(sq, sets_intake.profile)
    print("\n", result.status.model_dump_json(indent=2))
    for f in result.findings:
        print(f" - {f.id} [{f.study_type}/{f.grade}] {f.claim}\n   snippet: {f.snippet[:120]}")
    ids = {f.source_id for f in result.findings}
    assert {"FIXTURE-0001", "FIXTURE-0002"} <= ids
    assert "FIXTURE-0008" not in ids  # protein study is irrelevant to volume
    assert "FIXTURE-0006" not in ids  # retracted

    answer = WriterAgent(llm).run(sets_intake, result.findings)
    print("\n", answer.model_dump_json(indent=2))
    assert answer.all_cited_ids() <= {f.id for f in result.findings}
    assert answer.disclaimer
    print("\nLLM spend:", f"${llm.total_cost_usd:.4f}", "tokens:", llm.total_tokens)
    for r in llm.records:
        print(f"  {r.agent:<10} {r.model:<18} in={r.input_tokens:<6} out={r.output_tokens:<6} ${r.cost_usd:.4f} {r.latency_ms}ms ok={r.ok}")

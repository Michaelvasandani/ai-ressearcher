import pytest

from backend.agents.planner import MAX_SUB_QUESTIONS, PlannerAgent, PlannerError, lint_sub_question
from backend.llm import FakeLLM
from backend.schemas import SubQuestion
from backend.tests.conftest import load_fixture


def test_plan_from_fixture(sets_intake):
    llm = FakeLLM({"planner": load_fixture("plan_sets.json")})
    plan = PlannerAgent(llm).run(sets_intake)
    assert plan.question == sets_intake.scoped_question
    assert 3 <= len(plan.sub_questions) <= MAX_SUB_QUESTIONS
    assert [sq.id for sq in plan.sub_questions] == [f"sq{i}" for i in range(1, len(plan.sub_questions) + 1)]
    questions = [sq.question for sq in plan.sub_questions]
    assert len(questions) == len(set(questions))  # exact duplicate removed
    # Profile reaches the prompt so sub-questions can target the user's population.
    assert "intermediate" in llm.requests[0].user


def test_plan_is_capped_at_six(sets_intake):
    draft = {
        "sub_questions": [
            {"question": f"Does variable {i} change hypertrophy in trained adults?", "rationale": "r", "search_terms": [f"variable {i} hypertrophy"]}
            for i in range(9)
        ]
    }
    plan = PlannerAgent(FakeLLM({"planner": draft})).run(sets_intake)
    assert len(plan.sub_questions) == MAX_SUB_QUESTIONS


def test_too_few_usable_sub_questions_fails(sets_intake):
    draft = {
        "sub_questions": [
            {"question": "Does volume matter for hypertrophy in trained adults?", "rationale": "r", "search_terms": ["volume hypertrophy"]},
            {"question": "Does volume matter for hypertrophy in trained adults?", "rationale": "dup", "search_terms": ["x y"]},
            {"question": "Does frequency matter for hypertrophy in trained adults?", "rationale": "r", "search_terms": ["  "]},
        ]
    }
    with pytest.raises(PlannerError):
        PlannerAgent(FakeLLM({"planner": draft})).run(sets_intake)


def test_out_of_scope_never_planned(sets_intake, fake_llm):
    oos = sets_intake.model_copy(update={"in_scope": False, "scope_category": "injury", "referral_message": "see a doctor"})
    with pytest.raises(PlannerError):
        PlannerAgent(fake_llm).run(oos)
    assert fake_llm.requests == []


def test_lint_flags_vague_sub_questions():
    good = SubQuestion(
        id="sq1",
        question="Does training a muscle with 10 or more weekly sets produce more hypertrophy than fewer than 10 sets in trained adults?",
        rationale="r",
        search_terms=["weekly sets hypertrophy trained"],
    )
    assert lint_sub_question(good) == []
    vague = SubQuestion(id="sq2", question="Volume in general", rationale="r", search_terms=["volume"])
    problems = lint_sub_question(vague)
    assert {"too short to be specific", "not phrased as a question", "contains vague wording", "search terms are single words"} <= set(problems)


def test_fixture_plan_lint_catches_the_planted_vague_item(sets_intake):
    plan = PlannerAgent(FakeLLM({"planner": load_fixture("plan_sets.json")})).run(sets_intake)
    flagged = [sq.id for sq in plan.sub_questions if lint_sub_question(sq)]
    assert flagged == ["sq5"]

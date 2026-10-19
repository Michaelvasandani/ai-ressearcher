import pytest

from backend.agents.intake import REFERRALS, IntakeAgent, rule_screen
from backend.llm import FakeLLM
from backend.schemas import UserProfile
from backend.tests.conftest import load_fixture


@pytest.mark.parametrize(
    "question, category",
    [
        ("I'm 20 weeks pregnant, can I keep deadlifting?", "pregnancy"),
        ("How do I recover from anorexia and build muscle?", "eating_disorder"),
        ("Does metformin blunt hypertrophy from lifting?", "medication"),
        ("I'm on Ozempic, how much protein should I eat?", "medication"),
        ("I have type 2 diabetes, how many carbs before training?", "medical_condition"),
        ("I tore my ACL, what leg exercises can I do?", "injury"),
        ("I have a herniated disc, is squatting okay?", "injury"),
        ("Can you diagnose why my knee clicks?", "diagnosis"),
    ],
)
def test_rule_screen_catches_red_flags(question, category):
    assert rule_screen(question) == category


@pytest.mark.parametrize(
    "question",
    [
        "How many sets per week for hypertrophy?",
        "How much protein do I need on a cut?",
        "Do I have to train to failure to build muscle?",
        "How should I prescribe weekly volume for a beginner?",
        "Does insulin response to carbs matter for fat loss?",
        "Should I wear a belt for heavy squats?",
    ],
)
def test_rule_screen_lets_ordinary_questions_through(question):
    assert rule_screen(question) is None


def test_rule_hit_skips_the_model(fake_llm: FakeLLM):
    intake = IntakeAgent(fake_llm).run("I'm pregnant, is creatine safe?")
    assert not intake.in_scope
    assert intake.scope_category == "pregnancy" and intake.classified_by == "rule"
    assert intake.referral_message == REFERRALS["pregnancy"]
    assert fake_llm.requests == []


def test_in_scope_question_extracts_and_merges_profile():
    llm = FakeLLM({"intake": load_fixture("intake_sets.json")})
    stored = UserProfile(sex="female", days_per_week=3, equipment="full gym")
    intake = IntakeAgent(llm).run(
        "  I'm a woman who's lifted for 3 years, 4 days a week. How many sets per week for hypertrophy? ", stored
    )
    assert intake.in_scope and intake.scope_category == "in_scope" and intake.classified_by == "llm"
    assert intake.referral_message is None
    assert intake.original_question.startswith("I'm a woman")
    assert "hypertrophy" in intake.scoped_question
    p = intake.profile
    assert p.days_per_week == 4  # stated in the question beats stored settings
    assert p.equipment == "full gym"  # kept from settings
    assert p.training_status == "intermediate" and p.goal == "hypertrophy"
    req = llm.requests[0]
    assert "full gym" in req.user and "<message>" in req.user


def test_model_out_of_scope_gets_fixed_referral():
    draft = load_fixture("intake_sets.json") | {
        "scope_category": "injury",
        "scope_reason": "Asks about training around shoulder pain.",
    }
    llm = FakeLLM({"intake": draft})
    intake = IntakeAgent(llm).run("My shoulder hurts when I bench, what should I do?")
    assert not intake.in_scope and intake.classified_by == "llm"
    assert intake.referral_message == REFERRALS["injury"]  # code-chosen text, not model text


def test_off_topic_gets_scope_message():
    draft = load_fixture("intake_sets.json") | {"scope_category": "off_topic", "scope_reason": "Not fitness."}
    intake = IntakeAgent(FakeLLM({"intake": draft})).run("What's a good laptop?")
    assert not intake.in_scope and intake.referral_message == REFERRALS["off_topic"]


def test_empty_question_rejected(fake_llm):
    with pytest.raises(ValueError):
        IntakeAgent(fake_llm).run("   ")

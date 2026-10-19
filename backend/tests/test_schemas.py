import pytest
from pydantic import ValidationError

from backend.schemas import (
    DISCLAIMER,
    Argument,
    ArgumentClaim,
    CoachAnswer,
    Intake,
    Position,
    ResearchPlan,
    SubQuestion,
    UserProfile,
)
from backend.tests.conftest import make_finding


def _sq(i: int) -> SubQuestion:
    return SubQuestion(id=f"sq{i}", question=f"Question {i}?", rationale="r", search_terms=["a b"])


def _position(*ids: str) -> Position:
    return Position(stance="s", supporting_findings=list(ids), evidence_grade="moderate", best_counterargument="c")


def _answer(**kw) -> dict:
    data = dict(
        status="consensus",
        recommendation="Do 10-20 sets.",
        confidence="moderate",
        why="Because.",
        positions=[_position("f1")],
        what_would_change_this="x",
        applies_to_you="y",
        practical_protocol="z",
        whats_new=None,
        disclaimer=DISCLAIMER,
    )
    data.update(kw)
    return data


def test_plan_requires_3_to_6_sub_questions():
    ResearchPlan(question="q", sub_questions=[_sq(i) for i in range(1, 4)])
    ResearchPlan(question="q", sub_questions=[_sq(i) for i in range(1, 7)])
    with pytest.raises(ValidationError):
        ResearchPlan(question="q", sub_questions=[_sq(1), _sq(2)])
    with pytest.raises(ValidationError):
        ResearchPlan(question="q", sub_questions=[_sq(i) for i in range(1, 8)])


def test_plan_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        ResearchPlan(question="q", sub_questions=[_sq(1), _sq(1), _sq(2)])


def test_sub_question_needs_search_terms():
    with pytest.raises(ValidationError):
        SubQuestion(id="sq1", question="q?", rationale="r", search_terms=[])


def test_intake_scope_consistency():
    with pytest.raises(ValidationError):  # out of scope without referral
        Intake(
            original_question="q", scoped_question="q", in_scope=False, scope_category="injury",
            scope_reason="r", classified_by="rule",
        )
    with pytest.raises(ValidationError):  # flag and category disagree
        Intake(
            original_question="q", scoped_question="q", in_scope=True, scope_category="injury",
            scope_reason="r", classified_by="llm",
        )


def test_finding_requires_provenance():
    with pytest.raises(ValidationError):
        make_finding("sq1:X", source_url="  ")


def test_argument_claims_must_cite_findings():
    with pytest.raises(ValidationError):
        ArgumentClaim(text="Volume matters.", finding_ids=[])
    arg = Argument(stance="more", claims=[ArgumentClaim(text="t", finding_ids=["f1", "f2"])])
    arg.check_citations({"f1", "f2", "f3"})
    with pytest.raises(ValueError, match="unknown finding ids"):
        arg.check_citations({"f1"})


def test_position_must_cite_a_finding():
    with pytest.raises(ValidationError):
        _position()


def test_coach_answer_requires_disclaimer():
    with pytest.raises(ValidationError):
        CoachAnswer(**_answer(disclaimer="   "))
    data = _answer()
    del data["disclaimer"]
    with pytest.raises(ValidationError):
        CoachAnswer(**data)


def test_contested_answer_needs_two_positions():
    with pytest.raises(ValidationError):
        CoachAnswer(**_answer(status="genuinely_contested"))
    ans = CoachAnswer(**_answer(status="genuinely_contested", positions=[_position("f1"), _position("f2")], cited_finding_ids=["f3"]))
    assert ans.all_cited_ids() == {"f1", "f2", "f3"}


def test_profile_merge_prefers_newer_values():
    stored = UserProfile(sex="female", days_per_week=3, goal="strength")
    stated = UserProfile(days_per_week=4)
    merged = stored.merged_with(stated)
    assert merged.days_per_week == 4 and merged.sex == "female" and merged.goal == "strength"
    assert UserProfile().is_empty() and not merged.is_empty()


def test_records_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        UserProfile(favourite_color="blue")

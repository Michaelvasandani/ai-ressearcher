from backend.agents.grading import grade_finding


def test_baselines():
    assert grade_finding("meta_analysis", 500, False)[0] == "high"
    assert grade_finding("rct", 40, None)[0] == "moderate"
    assert grade_finding("cohort", 2000, None)[0] == "low"
    assert grade_finding("preprint", 2000, None)[0] == "low"
    assert grade_finding("animal", None, None)[0] == "very_low"


def test_downgrades_are_recorded_and_floored():
    grade, reasons = grade_finding("rct", 12, True)
    assert grade == "very_low"
    assert any("small trial" in r for r in reasons) and any("funding" in r for r in reasons)
    assert grade_finding("animal", None, True)[0] == "very_low"
    assert grade_finding("rct", None, None)[0] == "moderate"  # unknown n is not penalized

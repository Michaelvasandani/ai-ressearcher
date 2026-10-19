"""Citation gate: number extraction, sentence splitting, marker numbering and answer validation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from backend.citations import (
    extract_numbers,
    marker_numbers,
    matches,
    number_citations,
    sentences,
    validate_answer,
)
from backend.schemas import DISCLAIMER, CoachAnswer, Position, UserProfile
from backend.tests.conftest import make_finding

# --- Findings used throughout --------------------------------------------------------

F1 = make_finding(
    "sq1:PMID:1",
    claim="Each additional weekly set per muscle increased muscle size by 0.37%.",
    effect="+0.37% per weekly set (ES difference 0.24, 95% CI 0.04 to 0.44)",
    snippet="Ten or more weekly sets per muscle produced larger gains than fewer than 5 weekly sets.",
    population="390 adults across 15 studies",
    sample_size=390,
)
F2 = make_finding(
    "sq2:PMID:2",
    claim="Protein intakes above 1.62 g/kg/day gave no further gain in fat-free mass.",
    effect="plateau at 1.62 g/kg/day",
    snippet="Gains plateaued at 1.62 g/kg/day (95% CI 1.03 to 2.20).",
    population="1,863 participants in 49 trials",
    sample_size=1863,
    study_type="meta_analysis",
)
F3 = make_finding(
    "sq3:PMID:3",
    claim="Training each muscle twice a week matched once a week when weekly sets were equal.",
    effect="no difference",
    snippet="Volume-equated frequency of 2 vs 1 sessions per week did not change hypertrophy.",
    population="trained men",
    sample_size=20,
    study_type="rct",
    grade="moderate",
)
FINDINGS = [F1, F2, F3]
PROFILE = UserProfile(sex="female", training_status="intermediate", training_age_years=3, days_per_week=4, age=29)


def answer(**over) -> CoachAnswer:
    data = dict(
        status="consensus",
        recommendation="Do 10 or more hard sets per muscle each week [1].",
        confidence="moderate",
        why="Each extra weekly set added about 0.37% muscle size [1]. Protein above 1.62 g/kg/day added nothing more [2].",
        positions=[
            Position(
                stance="More weekly volume gives more growth",
                supporting_findings=["sq1:PMID:1"],
                evidence_grade="high",
                best_counterargument="Gains per set shrink at high volumes.",
            )
        ],
        what_would_change_this="Trials in trained women at high volumes.",
        applies_to_you="With 3 years of training you are close to the trained groups studied.",
        practical_protocol="Spread the sets across your 4 training days. Eat about 1.6 g/kg of protein a day [2].",
        whats_new=None,
        cited_finding_ids=["sq1:PMID:1", "sq2:PMID:2"],
        disclaimer=DISCLAIMER,
    )
    data.update(over)
    return CoachAnswer(**data)


def check(a: CoachAnswer, **kw):
    return validate_answer(a, FINDINGS, question=kw.pop("question", "How many sets per week?"), profile=kw.pop("profile", PROFILE))


def kinds(report) -> list[str]:
    return [i.kind for i in report.issues]


# --- Number extraction -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("10-20 sets", ["10", "20"]),
        ("10–20 sets", ["10", "20"]),
        ("10 to 20 sets", ["10", "20"]),
        ("about 0.37% per set", ["0.37"]),
        ("1,863 participants", ["1863"]),
        ("an effect of −0.2 (SMD)", ["0.2"]),
        ("+0.24 and -1.5", ["0.24", "1.5"]),
        ("1.6 g/kg/day", ["1.6"]),
        ("5x5 for 8 weeks", ["5", "5", "8"]),
        ("ten sets a week", ["10"]),
        ("twenty-five reps", ["25"]),
        ("two hundred grams", ["200"]),
        ("three training days", ["3"]),
        ("at 80% 1RM", ["80"]),
        ("in P1 and sq2, the 3rd and 21st week", []),
        ("position 2 and side 1 disagree", []),
        ("one study and the two sides", []),  # spelled numbers count only before a unit
        ("growth [1] and [2, 3]", []),  # markers are not numbers
        (".5 g", ["0.5"]),
    ],
)
def test_extract_numbers(text, values):
    assert [str(n.value) for n in extract_numbers(text)] == [str(Decimal(v)) for v in values]


def test_spelled_numbers_all_on_the_finding_side():
    assert [int(n.value) for n in extract_numbers("Ten or more sets; one of two groups", spelled="all")] == [10, 1, 2]


@pytest.mark.parametrize(
    ("answer_num", "finding", "ok"),
    [
        ("0.37", "0.37", True),
        ("0.370", "0.37", True),  # same value
        ("1.6", "1.62", True),  # rounds to the answer's precision
        ("0.4", "0.37", True),
        ("1.7", "1.62", False),
        ("10", "9.7", False),  # an integer must match exactly
        ("2", "1.62", False),
        ("20", "20", True),
        ("37", "0.37", False),  # no percent <-> fraction conversion
    ],
)
def test_number_matching(answer_num, finding, ok):
    a = extract_numbers(answer_num)[0]
    f = extract_numbers(finding)[0]
    assert matches(a, [(f.value, f.places)]) is ok


# --- Sentences and markers -----------------------------------------------------------


def test_sentences_keep_decimals_abbreviations_and_trailing_markers():
    text = "Gains were 0.37% per set. [1] Use e.g. Ten sets [2]. Is that all? Yes!"
    assert sentences(text) == ["Gains were 0.37% per set. [1]", "Use e.g. Ten sets [2].", "Is that all?", "Yes!"]
    assert sentences("") == [] and sentences("   ") == []


def test_marker_numbers():
    assert marker_numbers("1, 2") == [1, 2]
    assert marker_numbers("1-3") == [1, 2, 3]
    assert marker_numbers("2;4") == [2, 4]


def test_number_citations_turns_ids_into_numbered_markers():
    known = ["sq1:PMID:1", "sq2:PMID:2", "sq3:PMID:3"]
    fields, refs, invented = number_citations(
        {
            "why": "Volume helps [sq1:PMID:1]. Protein plateaus [sq2:PMID:2; sq1:PMID:1].",
            "recommendation": "Do more [sq9:PMID:999] .",
            "applies_to_you": "As noted [sic], you train 4 days.",
            "whats_new": None,
        },
        known,
        extra_ids=["sq3:PMID:3", "sq1:PMID:1", "unknown"],
    )
    assert fields["why"] == "Volume helps [1]. Protein plateaus [2, 1]."
    assert fields["recommendation"] == "Do more."  # an invented id is dropped (and then caught as uncited)
    assert fields["applies_to_you"] == "As noted [sic], you train 4 days."
    assert fields["whats_new"] is None
    assert refs == ["sq1:PMID:1", "sq2:PMID:2", "sq3:PMID:3"]
    assert invented == ["sq9:PMID:999"]


# --- Validation: passes ------------------------------------------------------------------


def test_valid_answer_passes():
    r = check(answer())
    assert r.ok, r.issues
    assert r.claims_checked >= 4 and r.citations == 4
    assert r.numbers_checked == 6  # 10, 0.37, 1.62, 3 (profile), 4 (profile), 1.6 (rounded 1.62)
    assert r.numbers_from_user == 2
    assert "every claim and number is backed" in r.summary()


def test_spelled_number_in_finding_backs_digits_in_answer():
    # F1's snippet says "Ten or more weekly sets": the answer may say 10.
    assert check(answer(recommendation="At least 10 weekly sets per muscle [1]."))


def test_sample_size_and_year_are_part_of_the_record():
    a = answer(why="A meta-analysis of 390 adults published in 2020 found 0.37% per set [1].")
    assert check(a).ok


def test_numbers_from_the_question_need_no_citation():
    a = answer(applies_to_you="With 5 sessions a week, split the volume.")
    assert not check(a).ok
    assert check(a, question="I can train 5 sessions a week. How many sets?").ok


def test_advice_sentences_without_numbers_need_no_citation():
    a = answer(applies_to_you="Track your sets and adjust.", what_would_change_this="New trials in women.")
    assert check(a).ok


# --- Validation: failures -------------------------------------------------------------------


def test_uncited_claim_in_evidence_fields_fails():
    r = check(answer(why="More volume is better for everyone."))
    assert kinds(r) == ["uncited_claim"]
    assert "makes a claim without citing a finding" in r.problems()[0]


def test_number_not_in_cited_finding_fails_with_a_hint():
    # 1.62 is in F2, but this sentence cites F1.
    r = check(answer(why="Protein above 1.62 g/kg/day added nothing [1]."))
    assert kinds(r) == ["unbacked_number"]
    issue = r.issues[0]
    assert issue.number == "1.62" and issue.cited == ["sq1:PMID:1"] and issue.found_in == ["sq2:PMID:2"]
    assert "it appears in sq2:PMID:2" in issue.detail


def test_invented_number_fails():
    r = check(answer(recommendation="Do 12-14 sets per muscle per week [1]."))
    assert kinds(r) == ["unbacked_number", "unbacked_number"]
    assert {i.number for i in r.issues} == {"12", "14"}
    assert all(i.found_in == [] for i in r.issues)


def test_range_needs_both_endpoints():
    # 10 is in F1 ("Ten or more"), 20 is not.
    r = check(answer(recommendation="Do 10-20 sets per muscle per week [1]."))
    assert [i.number for i in r.issues] == ["20"]


def test_uncited_number_in_advice_fails():
    r = check(answer(practical_protocol="Rest 90 seconds between sets."))
    assert kinds(r) == ["uncited_number"]


def test_spelled_quantity_in_answer_is_checked():
    r = check(answer(practical_protocol="Do twelve sets a week [1]."))
    assert kinds(r) == ["unbacked_number"] and r.issues[0].number.lower() == "twelve"


def test_marker_out_of_range_fails():
    r = check(answer(recommendation="Do more sets [7]."))
    assert "bad_marker" in kinds(r)
    assert "uncited_claim" in kinds(r)  # a bad marker cites nothing


def test_raw_id_left_in_text_fails():
    r = check(answer(why="Each extra weekly set added about 0.37% [1]. Also see [sq9:PMID:9]."))
    assert "raw_citation" in kinds(r)


def test_unknown_cited_id_fails():
    a = answer(cited_finding_ids=["sq1:PMID:1", "sq2:PMID:2", "made-up"])
    assert "unknown_finding" in kinds(check(a))


def test_position_stance_numbers_must_be_in_its_supporting_findings():
    pos = Position(
        stance="Gains plateau above 1.62 g/kg/day",
        supporting_findings=["sq1:PMID:1"],  # F1 does not say 1.62
        evidence_grade="high",
        best_counterargument="Some trials used 2.5 g/kg/day.",  # in no finding at all
    )
    r = check(answer(positions=[pos]))
    assert sorted((i.field, i.kind) for i in r.issues) == [
        ("positions[1].best_counterargument", "unbacked_number"),
        ("positions[1].stance", "unbacked_number"),
    ]


def test_counterargument_may_use_any_finding_the_answer_cites():
    pos = Position(
        stance="More weekly volume gives more growth",
        supporting_findings=["sq1:PMID:1"],
        evidence_grade="high",
        best_counterargument="Protein, not volume, may matter above 1.62 g/kg/day.",  # F2, cited by the answer
    )
    assert check(answer(positions=[pos])).ok


def test_problems_are_capped_for_the_rewrite_prompt():
    a = answer(practical_protocol=" ".join(f"Do {n} sets." for n in range(30, 50)))
    r = check(a)
    assert len(r.issues) == 20
    assert len(r.problems(limit=12)) == 13 and "8 more" in r.problems(limit=12)[-1]


# --- The gate in the graph ---------------------------------------------------------------

from backend import config  # noqa: E402
from backend.tests.graph_fakes import fake_llm, writer_response  # noqa: E402
from backend.tests.test_graph import nodes, run  # noqa: E402


def test_gate_passes_a_cited_answer_with_numbers(search_tool):
    r = run(fake_llm(search_tool), search_tool)
    assert r.outcome.status == "answered", r.outcome
    report = r.citation_reports[-1]
    assert report.ok and report.numbers_checked >= 2 and report.numbers_from_user == 1  # "your 4 training days"
    assert "[1]" in r.answer.recommendation  # inline ids became numbered markers
    gate = [s for s in r.trace if s.node == "validate_citations"]
    assert len(gate) == 1 and gate[0].status == "ok" and gate[0].agent == "citation_validator"


def test_gate_failure_gets_one_escalated_rewrite(search_tool):
    llm = fake_llm(search_tool, writer=[writer_response(unbacked_number="17"), writer_response()])
    r = run(llm, search_tool)
    assert r.outcome.status == "answered", r.outcome
    assert [w.model for w in llm.calls_for("writer")] == [config.SONNET, config.OPUS]
    feedback = llm.calls_for("writer")[1].user
    assert "rejected by review" in feedback and "states 17" in feedback and "citation check" in feedback
    assert [c.ok for c in r.citation_reports] == [False, True]
    assert [a.decision for a in r.answer_reviews] == ["rewrite", "accept"]
    assert "17" not in r.answer.practical_protocol
    assert nodes(r)[-7:] == ["writer", "validate_citations", "check_answer", "writer", "validate_citations", "check_answer", "finalize"]


def test_gate_failing_twice_fails_cleanly_without_an_answer(search_tool):
    llm = fake_llm(search_tool, writer=writer_response(unbacked_number="17"))
    r = run(llm, search_tool)
    assert len(llm.calls_for("writer")) == 2
    assert r.answer is None and r.findings
    assert r.outcome.status == "failed"
    assert r.outcome.failure.reason == "citations_failed" and r.outcome.failure.node == "validate_citations"
    assert "17" in r.outcome.failure.message


def test_gate_after_research_send_back_fails_without_rewrite(search_tool):
    llm = fake_llm(search_tool, ("volume", "frequency", "empty"), writer=writer_response(unbacked_number="17"))
    r = run(llm, search_tool)
    assert r.reviews[0].decision == "send_back"  # the single send-back went to research
    assert len(llm.calls_for("writer")) == 1
    assert r.outcome.failure.reason == "citations_failed" and r.answer is None

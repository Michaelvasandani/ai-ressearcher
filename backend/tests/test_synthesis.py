"""Position mapping, debate and judgment: deterministic rules and each agent alone (FakeLLM)."""

from __future__ import annotations

import pytest

from backend.agents.advocate import AdvocateAgent, ArgumentDraft, vet_claims
from backend.agents.evidence import confidence_cap, decide_status, grade_body
from backend.agents.judge import JudgeAgent, JudgeDraft, JudgeError, build_verdict
from backend.agents.mapper import MapDraft, MapperAgent, MapperError, build_position_map
from backend.graph.supervisor import downweight_reasons
from backend.llm import FakeLLM
from backend.schemas import Argument, ArgumentClaim, Position, PositionMap, UserProfile
from backend.tests.conftest import make_finding

TRAINED_WOMAN = UserProfile(sex="female", training_status="intermediate", age=28)

META = make_finding("sq1:META", study_type="meta_analysis", grade="high")
RCT = make_finding("sq1:RCT", study_type="rct", grade="moderate")
RCT2 = make_finding("sq1:RCT2", study_type="rct", grade="moderate", source_id="RCT2-src", source_url="https://example.org/r2")
RCT3 = make_finding("sq1:RCT3", study_type="rct", grade="moderate", source_id="RCT3-src", source_url="https://example.org/r3")
CASE = make_finding("sq2:CASE", study_type="case_report", grade="very_low", sample_size=1)
RAT = make_finding("sq2:RAT", study_type="animal", grade="very_low", population="rats")
PRE = make_finding("sq1:PRE", study_type="preprint", grade="low")
ALL = [META, RCT, RCT2, RCT3, CASE, RAT, PRE]
BY_ID = {f.id: f for f in ALL}


def pos(stance: str, *findings, grade: str | None = None) -> Position:
    g = grade or grade_body(list(findings), {})[0]
    return Position(stance=stance, supporting_findings=[f.id for f in findings], evidence_grade=g, best_counterargument="c")


# --- Body-of-evidence grades (code, not the model) ------------------------------


def test_grade_body_is_gradestyle():
    assert grade_body([META, CASE], {})[0] == "high"
    assert grade_body([RCT], {})[0] == "moderate"  # one trial is not a body of trials
    grade, reasons = grade_body([RCT, RCT2, RCT3], {})
    assert grade == "high" and any("consistent" in r for r in reasons)
    grade, reasons = grade_body([PRE, RAT], {})
    assert grade == "low" and "only preprint, animal or in-vitro" in reasons[-1]
    assert grade_body([CASE], {})[0] == "very_low"


def test_grade_body_downgrades_indirect_evidence_and_counts_a_paper_once():
    grade, reasons = grade_body([META], {META.id: "studied men only; user is female"})
    assert grade == "moderate" and "indirect" in reasons[0]
    # The same paper found under two sub-questions (two finding ids) does not make three consistent trials.
    dup = RCT.model_copy(update={"id": "sq2:RCT"})
    assert grade_body([RCT, dup, RCT2], {})[0] == "moderate"


# --- False balance ------------------------------------------------------------------


def test_weak_dissent_is_set_aside_as_consensus():
    """'Is creatine safe?': a meta-analysis + RCT against a case report + rat study is not a debate."""
    safe, harm = pos("safe", META, RCT), pos("harmful", CASE, RAT)
    d = decide_status([harm, safe], BY_ID, {}, proposed="genuinely_contested")
    assert d.status == "consensus"
    assert d.positions == [safe] and d.set_aside == [harm]
    assert any("false balance" in r for r in d.reasons)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((META, RCT), (make_finding("sq1:M2", source_id="M2", grade="high"), RCT2), "genuinely_contested"),
        ((META,), (RCT2,), "leaning"),  # credible (moderate) dissent against a high lead
        ((PRE,), (CASE,), "consensus"),  # low lead vs very-low dissent: still not credible dissent
    ],
)
def test_status_from_grades(a, b, expected):
    pa, pb = pos("a", *a), pos("b", *b)
    by_id = BY_ID | {f.id: f for f in (*a, *b)}
    assert decide_status([pa, pb], by_id, {}, proposed="genuinely_contested").status == expected


def test_mapper_can_only_move_towards_consensus():
    m2 = make_finding("sq1:M2", source_id="M2", grade="high")
    d = decide_status([pos("a", META), pos("b", m2)], BY_ID | {m2.id: m2}, {}, proposed="consensus")
    assert d.status == "consensus" and len(d.positions) == 1 and len(d.set_aside) == 1
    assert decide_status([pos("a", META), pos("b", RCT)], BY_ID, {}, proposed="leaning").status == "leaning"


def test_build_position_map_cleans_ids_and_grades_in_code():
    draft = MapDraft(
        positions=[
            {"stance": "harmful", "supporting_findings": [CASE.id, RAT.id, "sq9:INVENTED"], "best_counterargument": "c"},
            {"stance": "safe", "supporting_findings": [META.id, RCT.id, CASE.id], "best_counterargument": "c"},
            {"stance": "ghost", "supporting_findings": ["nope"], "best_counterargument": "c"},
        ],
        status="genuinely_contested",
        rationale="r",
    )
    pm = build_position_map(draft, ALL, {})
    assert pm.status == "consensus" and pm.proposed_status == "genuinely_contested"
    assert [p.stance for p in pm.positions] == ["safe"] and pm.positions[0].evidence_grade == "high"
    assert pm.positions[0].supporting_findings == [META.id, RCT.id]  # CASE already used by "harmful"
    assert [p.stance for p in pm.set_aside] == ["harmful"] and pm.set_aside[0].evidence_grade == "very_low"
    notes = " ".join(pm.status_reasons)
    assert "sq9:INVENTED" in notes and "dropped position 'ghost'" in notes
    assert set(pm.unassigned) == {RCT2.id, RCT3.id, PRE.id}


def test_mapper_agent_prompt_and_failure(sets_intake):
    draft = MapDraft(positions=[{"stance": "s", "supporting_findings": ["x"], "best_counterargument": "c"}],
                     status="consensus", rationale="r")
    llm = FakeLLM({"position_mapper": draft})
    with pytest.raises(MapperError):
        MapperAgent(llm).run(sets_intake, [META, RAT], {RAT.id: "non-human study (animal)"})
    user = llm.requests[0].user
    assert f"[{META.id}] (meta_analysis, grade high" in user
    assert "down-weighted: non-human study (animal)" in user and "cannot carry a recommendation alone" in user


# --- Advocates ----------------------------------------------------------------------


def test_advocate_rejects_uncited_and_invented_claims(sets_intake):
    draft = {
        "claims": [
            {"text": "A meta-analysis supports it.", "finding_ids": [META.id]},
            {"text": "Everyone knows this.", "finding_ids": []},
            {"text": "A trial I made up.", "finding_ids": [RCT.id, "sq9:INVENTED"]},
        ]
    }
    llm = FakeLLM({"advocate": draft})
    turn = AdvocateAgent(llm).run(sets_intake, 1, pos("safe", META, RCT), ALL)
    assert turn.round == 1 and turn.argument is not None and turn.error is None
    assert [c.text for c in turn.argument.claims] == ["A meta-analysis supports it."]
    assert {(r.text, r.reason.split(":")[0]) for r in turn.rejected_claims} == {
        ("Everyone knows this.", "cites no finding id"),
        ("A trial I made up.", "cites unknown finding ids"),
    }
    assert "Write your opening argument." in llm.requests[0].user


def test_advocate_with_no_valid_claim_fails_the_turn(sets_intake):
    llm = FakeLLM({"advocate": {"claims": [{"text": "Trust me.", "finding_ids": []}]}})
    turn = AdvocateAgent(llm).run(sets_intake, 2, pos("harmful", CASE), ALL)
    assert turn.argument is None and "every claim was rejected" in turn.error


def test_argument_schema_and_citation_check_reject_bad_ids():
    with pytest.raises(ValueError):
        ArgumentClaim(text="no id", finding_ids=[])
    arg = Argument(stance="s", claims=[ArgumentClaim(text="t", finding_ids=["sq9:INVENTED"])])
    with pytest.raises(ValueError, match="unknown finding ids"):
        arg.check_citations({META.id})
    kept, rejected = vet_claims(ArgumentDraft(claims=[{"text": "t", "finding_ids": ["  "]}]), {META.id})
    assert kept == [] and rejected[0].reason == "cites no finding id"


def test_rebuttal_sees_opponents_openings(sets_intake):
    opening = AdvocateAgent(FakeLLM({"advocate": {"claims": [{"text": "Harm seen.", "finding_ids": [CASE.id]}]}})).run(
        sets_intake, 2, pos("harmful", CASE), ALL
    )
    llm = FakeLLM({"advocate": {"claims": [{"text": "A single case report.", "finding_ids": [CASE.id]}]}})
    rebuttal = AdvocateAgent(llm).run(sets_intake, 1, pos("safe", META), ALL, opponents=[opening])
    assert rebuttal.round == 2 and rebuttal.argument.rebuttal_of == "P2: harmful"
    assert "Harm seen. [sq2:CASE]" in llm.requests[0].user and "Write your rebuttal." in llm.requests[0].user


# --- Judge --------------------------------------------------------------------------


def _map(*positions: Position, status: str = "leaning") -> PositionMap:
    return PositionMap(status=status, positions=list(positions))


def _draft(**kw) -> JudgeDraft:
    data = dict(recommended_position=1, status="consensus", confidence="high", recommendation="r", rationale="r")
    return JudgeDraft(**(data | kw))


def test_judge_confidence_is_capped_by_the_best_supporting_grade():
    pm = _map(pos("a", RCT), pos("b", META))
    v = build_verdict(_draft(status="leaning"), pm, ALL, {})
    assert v.confidence == "moderate" and v.confidence_cap == "moderate"
    assert "capped at moderate (best supporting grade is moderate)" in v.adjustments[0]
    assert [p.stance for p in v.positions] == ["a", "b"]
    # Indirect evidence lowers the cap: a down-weighted moderate trial allows only low confidence.
    v = build_verdict(_draft(status="leaning"), pm, ALL, {RCT.id: "studied men only; user is female"})
    assert v.confidence == "low" and v.downweighted_findings == {RCT.id: "studied men only; user is female"}


def test_judge_cap_contested_and_status_rules():
    pm = _map(pos("a", RCT), pos("b", META), status="genuinely_contested")
    v = build_verdict(_draft(recommended_position=2, status="genuinely_contested"), pm, ALL, {})
    assert v.confidence == "moderate" and v.positions[0].stance == "b"  # recommended position first
    # The judge cannot make the answer more contested than the map...
    v = build_verdict(_draft(status="genuinely_contested", confidence="low"), _map(pos("a", META), pos("b", RCT)), ALL, {})
    assert v.status == "leaning" and "cannot be more contested" in v.adjustments[0]
    # ...but may settle it: a consensus verdict shows one position.
    v = build_verdict(_draft(status="consensus"), _map(pos("a", META), pos("b", RCT)), ALL, {})
    assert v.status == "consensus" and v.confidence == "high" and len(v.positions) == 1


def test_judge_rejects_recommendations_carried_by_preprint_or_animal_work():
    pm = _map(pos("a", META), pos("b", PRE, RAT))
    with pytest.raises(JudgeError, match="cannot carry"):
        build_verdict(_draft(recommended_position=2), pm, ALL, {})
    with pytest.raises(JudgeError, match="not one of"):
        build_verdict(_draft(recommended_position=3), pm, ALL, {})


def test_judge_agent_sees_findings_grades_and_debate(sets_intake):
    llm = FakeLLM({"judge": _draft()})
    v = JudgeAgent(llm).run(sets_intake, _map(pos("safe", META), status="consensus"), ALL, [], {RAT.id: "non-human study (animal)"})
    user = llm.requests[0].user
    assert v.status == "consensus" and v.confidence == "high"
    assert "P1: safe\n  evidence grade: high" in user and f"[{RAT.id}] (animal, grade very_low" in user
    assert "No debate was held" in user


def test_confidence_cap_without_carrying_evidence():
    assert confidence_cap([PRE, RAT], {}, "consensus")[0] == "low"


# --- Personalization --------------------------------------------------------------


def test_downweight_reasons():
    assert downweight_reasons(TRAINED_WOMAN, make_finding("a", population="34 resistance-trained men")) == [
        "studied men only; user is female"
    ]
    assert downweight_reasons(None, RAT) == ["non-human study (animal)"]
    assert downweight_reasons(TRAINED_WOMAN, make_finding("b", population="40 trained adults (22 men, 18 women)")) == []

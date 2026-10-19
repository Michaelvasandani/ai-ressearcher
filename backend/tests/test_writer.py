import pytest

from backend.agents.writer import WriterAgent, WriterError
from backend.llm import FakeLLM
from backend.schemas import DISCLAIMER, Position, PositionMap, Verdict
from backend.tests.conftest import load_fixture


def test_writer_builds_answer_and_strips_unknown_ids(sets_intake, sets_findings):
    llm = FakeLLM({"writer": load_fixture("writer_sets.json")})
    answer = WriterAgent(llm).run(sets_intake, sets_findings)
    known = {f.id for f in sets_findings}

    assert answer.status == "leaning" and answer.confidence == "moderate"
    assert answer.disclaimer == DISCLAIMER
    assert len(answer.positions) == 2  # invented position dropped
    assert answer.all_cited_ids() <= known
    assert "sq1:FIXTURE-9999" not in answer.positions[0].supporting_findings
    assert answer.whats_new is None  # empty string normalized
    prompt = llm.requests[0].user
    assert all(f"[{f.id}]" in prompt for f in sets_findings)
    assert "intermediate" in prompt


def test_contested_with_one_surviving_position_downgrades(sets_intake, sets_findings):
    draft = load_fixture("writer_sets.json") | {"status": "genuinely_contested"}
    draft["positions"] = draft["positions"][:1] + draft["positions"][2:]  # one real, one invented
    answer = WriterAgent(FakeLLM({"writer": draft})).run(sets_intake, sets_findings)
    assert answer.status == "leaning" and len(answer.positions) == 1


def test_no_supported_position_fails(sets_intake, sets_findings):
    draft = load_fixture("writer_sets.json")
    draft["positions"] = draft["positions"][2:]
    with pytest.raises(WriterError):
        WriterAgent(FakeLLM({"writer": draft})).run(sets_intake, sets_findings)


def test_verdict_overrides_model_status(sets_intake, sets_findings):
    verdict = Verdict(
        status="consensus",
        recommendation="10-20 sets",
        confidence="high",
        rationale="r",
        positions=[Position(stance="more helps", supporting_findings=["sq1:FIXTURE-0001"], evidence_grade="high", best_counterargument="c")],
    )
    llm = FakeLLM({"writer": load_fixture("writer_sets.json")})
    answer = WriterAgent(llm).run(sets_intake, sets_findings, verdict=verdict)
    assert answer.status == "consensus" and answer.confidence == "high"
    assert answer.positions == verdict.positions
    assert "judge's verdict is final" in llm.requests[0].user


def test_position_map_overrides_status(sets_intake, sets_findings):
    pm = PositionMap(
        status="genuinely_contested",
        positions=[
            Position(stance="a", supporting_findings=["sq1:FIXTURE-0001"], evidence_grade="high", best_counterargument="c"),
            Position(stance="b", supporting_findings=["sq1:FIXTURE-0004"], evidence_grade="moderate", best_counterargument="c"),
        ],
    )
    answer = WriterAgent(FakeLLM({"writer": load_fixture("writer_sets.json")})).run(sets_intake, sets_findings, positions=pm)
    assert answer.status == "genuinely_contested" and answer.positions == pm.positions


def test_writer_guards(sets_intake, sets_findings, fake_llm):
    with pytest.raises(WriterError):
        WriterAgent(fake_llm).run(sets_intake, [])
    oos = sets_intake.model_copy(update={"in_scope": False, "scope_category": "injury", "referral_message": "x"})
    with pytest.raises(WriterError):
        WriterAgent(fake_llm).run(oos, sets_findings)

"""Citation gate repairs: deterministic autofix, the citation-only repair pass, and how they sit in the graph."""

from __future__ import annotations

from typing import Any

from backend import config
from backend.agents.citation_repair import CitationRepairAgent, RepairDraft, can_repair, repairable_fields
from backend.citations import (
    answer_id_texts,
    autofix_citations,
    unnumber_citations,
    validate_answer,
    with_id_texts,
)
from backend.llm import FakeLLM, LLMRequest
from backend.schemas import Intake, Position, UserProfile
from backend.tests.graph_fakes import fake_llm, snippet_number, writer_response
from backend.tests.test_citations import F1, F2, F3, FINDINGS, PROFILE, answer, check, kinds
from backend.tests.test_graph import nodes, run

# --- Markers <-> ids -----------------------------------------------------------------


def test_unnumber_round_trips_through_with_id_texts():
    a = answer()
    texts = answer_id_texts(a)
    assert texts["recommendation"] == "Do 10 or more hard sets per muscle each week [sq1:PMID:1]."
    assert unnumber_citations("x [1, 2] y [9]", ["a", "b"]) == "x [a; b] y [9]"  # [9] points nowhere: kept
    same, invented = with_id_texts(a, {}, {f.id for f in FINDINGS})
    assert invented == [] and same.recommendation == a.recommendation and same.cited_finding_ids == a.cited_finding_ids


def test_with_id_texts_renumbers_and_drops_invented_ids():
    a = answer()
    new, invented = with_id_texts(
        a, {"recommendation": "Train each muscle twice a week [sq3:PMID:3]. Also [sq9:PMID:999]."}, {f.id for f in FINDINGS}
    )
    assert invented == ["sq9:PMID:999"]
    assert new.cited_finding_ids[0] == "sq3:PMID:3" and new.recommendation.startswith("Train each muscle twice a week [1].")
    assert "[2]" in new.why  # sq1 moved from [1] to [2] everywhere


# --- Deterministic autofix -------------------------------------------------------------


def test_autofix_cites_the_one_cited_study_that_holds_an_advice_number():
    a = answer(practical_protocol="Aim for 10 or more hard sets per muscle. Eat about 1.6 g/kg of protein a day [2].")
    assert kinds(check(a)) == ["uncited_number"]
    fixed, notes = autofix_citations(a, FINDINGS, question="How many sets per week?", profile=PROFILE)
    assert notes == ["practical_protocol: cited sq1:PMID:1 for 10"]
    assert fixed.practical_protocol.startswith("Aim for 10 or more hard sets per muscle [1].")
    assert check(fixed).ok


def test_autofix_leaves_evidence_fields_ambiguous_numbers_and_uncited_studies_alone():
    findings = FINDINGS
    # Evidence field: a claim needs a semantic citation, not a number match.
    a = answer(why="Ten or more weekly sets beat fewer than 5 sets. Protein above 1.62 g/kg/day added nothing more [2].")
    assert autofix_citations(a, findings, profile=PROFILE)[1] == []
    # The number is in a study the answer does not cite (F3 holds "2"): no completion.
    b = answer(practical_protocol="Train each muscle 2 times a week. Eat about 1.6 g/kg of protein a day [2].")
    assert autofix_citations(b, findings, profile=PROFILE)[1] == []
    # Two different cited studies hold the number: ambiguous, left alone.
    c = answer(
        practical_protocol="Use studies from 2020 on. Eat about 1.6 g/kg of protein a day [2].",  # every year is 2020
        cited_finding_ids=["sq1:PMID:1", "sq2:PMID:2", "sq3:PMID:3"],
    )
    assert autofix_citations(c, findings, profile=PROFILE)[1] == []


def test_autofix_ignores_numbers_from_the_user():
    a = answer(practical_protocol="Use your 4 training days for 10 hard sets per muscle. Eat about 1.6 g/kg [2].")
    fixed, notes = autofix_citations(a, FINDINGS, question="How many sets?", profile=PROFILE)
    assert notes and check(fixed).ok


# --- Repair agent ----------------------------------------------------------------------

INTAKE = Intake(
    original_question="How many sets?", scoped_question="How many sets per week for hypertrophy?", in_scope=True,
    scope_category="in_scope", scope_reason="training", classified_by="llm", profile=PROFILE,
)


def test_can_repair_text_and_position_fields_but_not_unknown_ids():
    bad_text = check(answer(why="An uncited claim. Protein above 1.62 g/kg/day added nothing more [2]."))
    assert can_repair(bad_text) and repairable_fields(bad_text) == ["why"]
    stance = Position(stance="Do 99 sets", supporting_findings=["sq1:PMID:1"], evidence_grade="high", best_counterargument="x")
    bad_stance = check(answer(positions=[stance]))
    assert can_repair(bad_stance) and repairable_fields(bad_stance) == ["positions[1].stance"]
    unknown = check(answer(cited_finding_ids=["sq1:PMID:1", "sq2:PMID:2", "sq9:PMID:404"]))
    assert not can_repair(unknown)  # an unknown finding id is not a wording problem


def test_repair_rewrites_a_position_counterargument_without_citations():
    pos = Position(
        stance="More weekly volume gives more growth", supporting_findings=["sq1:PMID:1"], evidence_grade="high",
        best_counterargument="Gains stop above 30 sets.",
    )
    a = answer(positions=[pos])
    report = check(a)
    assert repairable_fields(report) == ["positions[1].best_counterargument"]
    llm = FakeLLM(
        {
            "citation_repair": RepairDraft(
                fields=[{"field": "positions[1].best_counterargument", "text": "Gains per set shrink at high volumes [sq1:PMID:1]."}]
            )
        }
    )
    fixed = CitationRepairAgent(llm).run(INTAKE, a, FINDINGS, report)
    assert fixed.positions[0].best_counterargument == "Gains per set shrink at high volumes."
    assert fixed.positions[0].supporting_findings == ["sq1:PMID:1"] and check(fixed).ok


def test_repair_agent_rewrites_only_the_requested_fields():
    a = answer(why="An uncited claim. Protein above 1.62 g/kg/day added nothing more [2].")
    report = check(a)
    llm = FakeLLM(
        {
            "citation_repair": RepairDraft(
                fields=[
                    {"field": "why", "text": "Each extra weekly set added about 0.37% muscle size [sq1:PMID:1]."},
                    {"field": "recommendation", "text": "Not requested: ignored [sq3:PMID:3]."},
                ]
            )
        }
    )
    fixed = CitationRepairAgent(llm, model=config.SONNET).run(INTAKE, a, FINDINGS, report)
    req = llm.calls_for("citation_repair")[0]
    assert req.model == config.SONNET and "Fields to fix: why" in req.user and "[sq2:PMID:2]" in req.user
    assert fixed.recommendation == a.recommendation
    assert check(fixed).ok


# --- In the graph -------------------------------------------------------------------------


def repair_response(protocol: str = "Spread your sets for each muscle across your 4 training days."):
    """Drop the unbacked sentence that writer_response(unbacked_number=...) adds."""

    def respond(request: LLMRequest[Any]) -> RepairDraft:
        return RepairDraft(fields=[{"field": "practical_protocol", "text": protocol}])

    return respond


def test_repair_fixes_the_draft_without_using_the_send_back(search_tool):
    llm = fake_llm(search_tool, writer=writer_response(unbacked_number="17"), citation_repair=repair_response())
    r = run(llm, search_tool)
    assert r.outcome.status == "answered", r.outcome
    assert len(llm.calls_for("writer")) == 1 and len(llm.calls_for("citation_repair")) == 1
    assert llm.calls_for("citation_repair")[0].model == config.SONNET
    assert [(c.stage, c.ok) for c in r.citation_reports] == [("draft", False), ("repaired", True)]
    assert [a.decision for a in r.answer_reviews] == ["accept"]
    assert "17" not in r.answer.practical_protocol
    assert nodes(r)[-6:-1] == ["writer", "validate_citations", "repair_citations", "validate_citations", "check_answer"]
    repair_step = next(s for s in r.trace if s.node == "repair_citations")
    assert repair_step.agent == "citation_repair" and repair_step.cost_usd > 0 and repair_step.status == "ok"


def test_repair_rescues_a_run_whose_send_back_went_to_research(search_tool):
    # Phase 6's failure mode: research used the single send-back, so a failed draft had no rewrite left.
    llm = fake_llm(
        search_tool, ("volume", "frequency", "empty"),
        writer=writer_response(unbacked_number="17"), citation_repair=repair_response(),
    )
    r = run(llm, search_tool)
    assert r.reviews[0].decision == "send_back"
    assert r.outcome.status in ("answered", "degraded"), r.outcome
    assert r.answer is not None and len(llm.calls_for("writer")) == 1


def test_a_repair_that_does_not_fix_it_falls_back_to_the_rewrite(search_tool):
    llm = fake_llm(
        search_tool,
        writer=[writer_response(unbacked_number="17"), writer_response()],
        citation_repair=repair_response("Do 17 sets per muscle per week [sq9:PMID:404]."),  # invented id: dropped
    )
    r = run(llm, search_tool)
    assert r.outcome.status == "answered"
    assert [w.model for w in llm.calls_for("writer")] == [config.SONNET, config.OPUS]
    assert len(llm.calls_for("citation_repair")) == 1  # the rewrite passed: no second repair
    assert [(c.attempt, c.stage, c.ok) for c in r.citation_reports] == [(1, "draft", False), (1, "repaired", False), (2, "draft", True)]


def test_one_repair_per_draft_and_the_run_still_fails_cleanly(search_tool):
    bad = repair_response("Do 17 sets per muscle per week.")
    llm = fake_llm(search_tool, writer=writer_response(unbacked_number="17"), citation_repair=bad)
    r = run(llm, search_tool)
    assert len(llm.calls_for("writer")) == 2 and len(llm.calls_for("citation_repair")) == 2  # one per draft
    assert r.answer is None and r.outcome.failure.reason == "citations_failed"


def test_no_repair_when_the_draft_passes(search_tool):
    llm = fake_llm(search_tool, citation_repair=repair_response())
    r = run(llm, search_tool)
    assert r.outcome.status == "answered" and llm.calls_for("citation_repair") == []
    assert [c.stage for c in r.citation_reports] == ["draft"]


def test_autofix_runs_inside_the_gate(search_tool):
    # The writer forgets the bracket on an advice sentence whose number its cited study states.
    def writer(request: LLMRequest[Any]):
        draft = writer_response()(request)
        n = next(filter(None, (snippet_number(request, fid) for fid in draft.cited_finding_ids)))
        return draft.model_copy(update={"practical_protocol": f"Keep the figure of {n} in mind."})

    llm = fake_llm(search_tool, writer=writer, citation_repair=repair_response())
    r = run(llm, search_tool)
    assert r.outcome.status == "answered", r.outcome
    assert llm.calls_for("citation_repair") == []
    stages = [(c.stage, c.ok) for c in r.citation_reports]
    assert stages in ([("draft", False), ("autofix", True)], [("draft", True)])


def test_validate_answer_records_the_stage():
    r = validate_answer(answer(), [F1, F2, F3], profile=UserProfile(), stage="repaired")
    assert r.stage == "repaired"

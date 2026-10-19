"""Researcher, supervisor and graph on top of the live search tool (mocked HTTP, offline)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import httpx

from backend.agents.researcher import ExtractionBatch, ResearcherAgent, StudyExtraction
from backend.graph import GraphDeps, memory_checkpointer, run_research
from backend.graph import nodes as graph_nodes
from backend.graph.nodes import Nodes
from backend.graph.state import PassRecord, ResearchRecord, all_findings
from backend.graph.supervisor import provenance_gate, review_sub_question
from backend.llm import FakeLLM
from backend.schemas import SearchStatus
from backend.tests.conftest import make_finding
from backend.tests.graph_fakes import fake_llm
from backend.tests.live_fakes import FakeAPIs, live_tool

QUERY = "resistance training volume hypertrophy"


def extraction(request) -> ExtractionBatch:
    """Model stand-in: claims carry the numbers; snippets are quoted from the abstract."""
    studies = []
    for sid in re.findall(r'<record source_id="([^"]+)">', request.user):
        block = request.user.split(f'<record source_id="{sid}">', 1)[1].split("</record>", 1)[0]
        abstract = block.split("Abstract: ", 1)[1].split("\n", 1)[0]
        claim = abstract.split(". ")[0]
        if sid == "PMID:39000002":
            claim = "Muscle thickness grew 5.2% with 20 weekly sets versus 3.9% with 10."
        studies.append(
            StudyExtraction(
                source_id=sid, relevant=True, claim=claim, snippet=abstract.split(". ")[0],
                study_type="rct", population="40 trained women", sample_size=40, effect=None, funding_conflict=False,
            )
        )
    return ExtractionBatch(studies=studies)


def run_researcher(apis: FakeAPIs, sub_question):
    llm = FakeLLM({"researcher": extraction})
    sq = sub_question.model_copy(update={"search_terms": [QUERY]})
    return ResearcherAgent(llm, live_tool(apis)).run(sq), llm


def test_researcher_findings_carry_provenance_and_full_text_snippets(volume_sub_question):
    result, llm = run_researcher(FakeAPIs(), volume_sub_question)
    status = result.status
    assert status.state == "ok"
    by_source = {f.source_id: f for f in result.findings}

    trial = by_source["PMID:39000002"]
    p = trial.provenance
    assert p.source == "pubmed" and p.endpoint == "pubmed:efetch" and p.url == "https://pubmed.ncbi.nlm.nih.gov/39000002/"
    assert p.text_source == "full_text" and p.full_text_url.endswith("/PMC9000002/fullTextXML")
    assert p.also_found_in == ["europepmc:search"] and "crossref" in p.retraction_checked_by
    assert trial.retrieved_at == p.retrieved_at
    # The model quoted the abstract; the snippet comes from the Results section instead.
    assert trial.snippet.startswith("Results: Muscle thickness increased by 5.2% in the 20-set group")

    meta = by_source["PMID:39000001"]
    assert meta.provenance.text_source == "abstract" and meta.study_type == "meta_analysis"
    assert "PMID:39000001" in status.abstract_only and "PMID:39000002" not in status.abstract_only

    dropped = {(d.source_id, d.reason) for d in status.dropped}
    assert {("PMID:39000004", "retracted"), ("PMID:39000006", "retracted"), ("PMID:39000003", "author_cap")} <= dropped
    assert ("PMID:39000002", "duplicate") in dropped
    rw = next(d for d in status.dropped if d.source_id == "PMID:39000006")
    assert "retraction-watch" in rw.detail

    prompt = llm.requests[0].user
    assert "39000004" not in prompt and "39000006" not in prompt  # retracted never reach the model
    assert "Text available: open-access full text" in prompt and "Text available: abstract only" in prompt
    assert "MeSH: Resistance Training" in prompt
    assert by_source["DOI:10.31236/osf.io/ec008"].study_type == "preprint"


def test_researcher_paywalled_status(volume_sub_question):
    result, _ = run_researcher(FakeAPIs({"fulltext": [httpx.Response(404)]}), volume_sub_question)
    assert result.status.state == "paywalled" and result.findings
    assert all(f.provenance.text_source == "abstract" for f in result.findings)
    assert len(result.status.abstract_only) == len(result.findings)


def test_researcher_retracted_status(volume_sub_question):
    only_retracted = httpx.Response(200, json={"esearchresult": {"count": "1", "idlist": ["39000004"]}})
    empty = httpx.Response(200, json={"hitCount": 0, "resultList": {"result": []}})
    result, llm = run_researcher(FakeAPIs({"esearch": [only_retracted], "epmc_search": [empty]}), volume_sub_question)
    assert result.status.state == "retracted" and result.findings == []
    assert llm.requests == []


def test_researcher_rate_limited_and_timed_out_statuses(volume_sub_question):
    limited = FakeAPIs({"esearch": [httpx.Response(429)], "epmc_search": [httpx.Response(429)]})
    result, llm = run_researcher(limited, volume_sub_question)
    assert result.status.state == "rate_limited" and llm.requests == []
    assert any("rate_limited" in e for e in result.status.errors)

    slow = FakeAPIs({"esearch": [httpx.ReadTimeout("x")], "epmc_search": [httpx.ReadTimeout("x")]})
    assert run_researcher(slow, volume_sub_question)[0].status.state == "timed_out"


# --- Supervisor -------------------------------------------------------------------


def _record(sq, state, findings=()):
    status = SearchStatus(sub_question_id=sq.id, state=state, records_seen=3, findings_kept=len(findings))
    return ResearchRecord(
        sub_question_id=sq.id, findings=list(findings), searches_used=1, queries_run=list(sq.search_terms[:1]),
        passes=[PassRecord(pass_no=1, model="m", escalated=False, queries=sq.search_terms[:1], status=status, finding_ids=[f.id for f in findings])],
    )


def test_supervisor_paywalled_is_answered_but_flagged(volume_sub_question):
    rec = _record(volume_sub_question, "paywalled", [make_finding("sq1:PMID:1", population="trained women")])
    review = review_sub_question(volume_sub_question, rec, None, searches_left=3, send_back_available=True, budget_block=None)
    assert review.answered and "abstract_only" in review.issues and "search_failed" not in review.issues
    assert not review.another_pass
    report = SimpleNamespace(sub_questions=[review], pass_no=1)
    kinds = {d.kind for d in Nodes._review_degradations(report)}
    assert kinds == {"abstract_only_evidence"}


def test_supervisor_retracted_only_searches_again_without_escalating(volume_sub_question):
    rec = _record(volume_sub_question, "retracted")
    review = review_sub_question(volume_sub_question, rec, None, searches_left=3, send_back_available=True, budget_block=None)
    assert {"no_findings", "retracted_only"} <= set(review.issues)
    assert review.another_pass and not review.escalate
    assert review.refined_queries[0].endswith("meta-analysis meta-analysis")  # look for a synthesis instead
    assert "weekly sets muscle thickness trained" in review.refined_queries  # plus untried planned terms
    assert volume_sub_question.search_terms[0] not in review.refined_queries  # never repeat a query that ran


# --- Provenance gate ----------------------------------------------------------------


def test_provenance_gate_rejects_findings_without_provenance():
    good = make_finding("sq1:PMID:1")
    missing = make_finding("sq1:PMID:2", provenance=None)
    mismatched = make_finding("sq1:PMID:3", provenance=good.provenance)
    kept, rejected = provenance_gate([good, missing, mismatched])
    assert kept == [good]
    assert rejected == {"sq1:PMID:2": "no provenance record", "sq1:PMID:3": "provenance is for 1, finding is 3"}

    sq = SimpleNamespace(id="sq1")
    state = {"sub_questions": [sq], "research": {"sq1": ResearchRecord(sub_question_id="sq1", findings=[good, missing])}}
    assert all_findings(state) == [good]  # never reaches the writer


def test_graph_rejects_unprovenanced_findings_before_the_writer(search_tool, monkeypatch):
    original = graph_nodes.ResearcherAgent.run

    def strip_one(self, sq, profile=None):
        result = original(self, sq, profile)
        if result.findings:
            result.findings[0] = result.findings[0].model_copy(update={"provenance": None})
        return result

    monkeypatch.setattr(graph_nodes.ResearcherAgent, "run", strip_one)
    llm = fake_llm(search_tool)
    r = run_research("How many sets per week for hypertrophy?", deps=GraphDeps(llm=llm, search_tool=search_tool), checkpointer=memory_checkpointer())
    rejected = [d for d in r.outcome.degradations if d.kind == "provenance_rejected"]
    assert len(rejected) == 3
    assert all(f.provenance is not None for f in r.findings)
    writer_prompt = llm.calls_for("writer")[0].user
    for step in (s for s in r.trace if s.node == "researcher"):
        bad = [d for d in step.output["status"]["dropped"] if d["reason"] == "no_provenance"]
        assert len(bad) == 1
        assert f"[{step.sub_question_id}:{bad[0]['source_id']}] (" not in writer_prompt
    assert re.search(r"^\[sq\d:FIXTURE-\d{4}\] \(", writer_prompt, re.M)  # the prompt format the check relies on
    assert r.answer is not None


# --- Graph end to end on the live tool --------------------------------------------------


def test_graph_runs_on_live_tool_with_mocked_apis():
    apis = FakeAPIs()
    tool = live_tool(apis)
    records = tool.search(QUERY).records
    llm = fake_llm(SimpleNamespace(records=records))
    r = run_research(
        "How many sets per week for hypertrophy?", deps=GraphDeps(llm=llm, search_tool=tool), checkpointer=memory_checkpointer()
    )
    assert r.answer is not None and r.outcome.status in ("answered", "degraded")
    assert r.findings and all(f.provenance and f.provenance.source in ("pubmed", "europepmc") for f in r.findings)
    assert not any(f.retracted or f.source_id in ("PMID:39000004", "PMID:39000006") for f in r.findings)
    calls = [c for s in r.trace if s.node == "researcher" for c in s.tool_calls]
    assert calls and all(c.tool == "search:live" and c.state == "ok" for c in calls)

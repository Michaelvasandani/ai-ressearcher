from backend.agents.researcher import ResearcherAgent
from backend.llm import FakeLLM, LLMError
from backend.tools.search import FixtureSearchTool
from backend.tests.conftest import load_fixture


def _run(llm, tool, sq, **kw):
    return ResearcherAgent(llm, tool, **kw).run(sq)


def test_researcher_returns_graded_findings_with_provenance(search_tool, volume_sub_question):
    llm = FakeLLM({"researcher": load_fixture("extraction_sets.json")})
    result = _run(llm, search_tool, volume_sub_question)
    by_source = {f.source_id: f for f in result.findings}

    assert set(by_source) == {"FIXTURE-0001", "FIXTURE-0002", "FIXTURE-0004", "FIXTURE-0005", "FIXTURE-0007"}
    status = result.status
    assert status.state == "ok" and status.sub_question_id == "sq1"
    assert status.searches_run == 2 and status.findings_kept == 5
    dropped = {(d.source_id, d.reason) for d in status.dropped}
    assert ("FIXTURE-0006", "retracted") in dropped  # retracted paper never reaches the model
    assert ("FIXTURE-0010", "irrelevant") in dropped
    assert ("FIXTURE-0008", "irrelevant") in dropped
    assert ("FIXTURE-0003", "extraction_failed") in dropped  # model skipped it
    assert "FIXTURE-0006" not in llm.requests[0].user

    meta = by_source["FIXTURE-0001"]
    assert meta.id == "sq1:FIXTURE-0001" and meta.sub_question_id == "sq1"
    assert meta.source_url == "https://example.org/fixtures/FIXTURE-0001"  # provenance from the record
    assert meta.published.year == 2017 and meta.title.startswith("Dose-response")
    assert meta.grade == "high" and meta.study_type == "meta_analysis"
    assert meta.snippet in search_tool.records[0].full_text

    # Source metadata overrides the model's study-design label.
    assert by_source["FIXTURE-0004"].study_type == "preprint"
    assert by_source["FIXTURE-0004"].grade == "low"
    assert by_source["FIXTURE-0007"].grade == "very_low"
    assert by_source["FIXTURE-0002"].grade == "moderate"
    # A paraphrased snippet is replaced by the source sentence that carries the claim's numbers.
    assert by_source["FIXTURE-0002"].snippet.startswith("Results: Quadriceps muscle thickness increased by 6.8%")


def test_partial_when_one_search_fails(search_tool, volume_sub_question):
    tool = FixtureSearchTool(search_tool.records, failures={"weekly sets": "rate_limited"})
    llm = FakeLLM({"researcher": load_fixture("extraction_sets.json")})
    result = _run(llm, tool, volume_sub_question)
    assert result.status.state == "partial"
    assert result.findings and any("rate_limited" in e for e in result.status.errors)


def test_all_searches_fail_returns_status_not_exception(search_tool, volume_sub_question, fake_llm):
    tool = FixtureSearchTool(search_tool.records, failures={"": "timed_out"})
    result = _run(fake_llm, tool, volume_sub_question)
    assert result.findings == [] and result.status.state == "timed_out"
    assert fake_llm.requests == []  # nothing to extract, no model call


def test_no_results(search_tool, volume_sub_question, fake_llm):
    sq = volume_sub_question.model_copy(update={"search_terms": ["creatine kidney function"]})
    result = _run(fake_llm, search_tool, sq)
    assert result.status.state == "no_results" and fake_llm.requests == []


def test_search_tool_exception_is_contained(volume_sub_question, fake_llm):
    class Broken:
        name = "broken"

        def search(self, query, max_results=10):
            raise RuntimeError("socket closed")

    result = _run(fake_llm, Broken(), volume_sub_question)
    assert result.status.state == "error" and "socket closed" in result.status.errors[0]


def test_extraction_failure_is_contained(search_tool, volume_sub_question):
    llm = FakeLLM({"researcher": LLMError("overloaded")})
    result = _run(llm, search_tool, volume_sub_question)
    assert result.findings == [] and result.status.state == "error"
    assert all(d.reason in ("retracted", "extraction_failed") for d in result.status.dropped)


def test_budget_caps(search_tool, volume_sub_question):
    llm = FakeLLM({"researcher": load_fixture("extraction_sets.json")})
    result = _run(llm, search_tool, volume_sub_question, max_searches=1, max_records=3)
    assert result.status.searches_run == 1 and search_tool.queries == volume_sub_question.search_terms[:1]
    assert llm.requests[0].user.count("<record ") == 3


def test_snippet_without_the_claims_numbers_is_replaced(search_tool):
    from backend.agents.researcher import _choose_snippet

    rec = search_tool.records[0]  # FIXTURE-0001
    claim = "Each extra weekly set added 0.37% muscle size."
    assert _choose_snippet("No study was industry funded.", claim, rec).startswith(
        "Each additional weekly set per muscle was associated with a 0.37% increase"
    )
    good = "Each additional weekly set per muscle was associated with a 0.37% increase in muscle size."
    assert _choose_snippet(good, claim, rec) == good
    # No numbers in the claim: any verbatim snippet is accepted.
    assert _choose_snippet("No study was industry funded.", "Volume helps.", rec) == "No study was industry funded."

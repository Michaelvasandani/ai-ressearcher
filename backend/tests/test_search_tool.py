from backend.tools.search import FixtureSearchTool


def test_fixture_search_matches_by_keywords(search_tool):
    resp = search_tool.search("protein intake energy deficit lean mass")
    assert resp.state == "ok"
    assert resp.records[0].source_id == "FIXTURE-0009"
    assert search_tool.queries == ["protein intake energy deficit lean mass"]


def test_fixture_search_no_results_and_injected_failure(search_tool):
    assert search_tool.search("creatine kidney function").state == "no_results"
    tool = FixtureSearchTool(search_tool.records, failures={"protein": "rate_limited"})
    resp = tool.search("protein timing")
    assert resp.state == "rate_limited" and resp.records == []


def test_max_results(search_tool):
    assert len(search_tool.search("resistance training volume hypertrophy", max_results=2).records) == 2

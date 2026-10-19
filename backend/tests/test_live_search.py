"""Live search tools against recorded-format fixtures and mocked HTTP (offline)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import httpx
import pytest

from backend.config import SearchSettings
from backend.schemas import StudyRecord
from backend.tests.live_fakes import TODAY, FakeAPIs, Sleeps, fixture_bytes, http_client, json_file, live_tool
from backend.tools.composite import CompositeSearchTool
from backend.tools.europepmc import EuropePMCSearch, build_query, parse_hit
from backend.tools.factory import build_search_tool
from backend.tools.http import FetchError, HttpClient, RateLimiter
from backend.tools.jats import parse_jats
from backend.tools.pubmed import PubMedSearch, build_term, parse_efetch
from backend.tools.ranking import (
    DiversityCaps,
    RecencyPolicy,
    apply_caps,
    author_group,
    dedupe,
    normalize_doi,
    rank,
)
from backend.tools.retraction import CrossrefRetractionChecker, retraction_in_updates
from backend.tools.search import FixtureSearchTool
from backend.tools.sportrxiv import SportRxivSearch

QUERY = "resistance training volume hypertrophy"
NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)


def bare_http(apis: FakeAPIs, sleeps: Sleeps | None = None, max_retries: int = 2) -> HttpClient:
    """No rate limiters, so recorded sleeps are only backoff."""
    return HttpClient(
        httpx.Client(transport=httpx.MockTransport(apis)), max_retries=max_retries,
        sleep=sleeps if sleeps is not None else Sleeps(), jitter=lambda: 0.0,
    )


def rec(source_id: str, **kw) -> StudyRecord:
    data = dict(
        source_id=source_id, source_url=f"https://example.org/{source_id}", title=f"Study {source_id}",
        abstract="Abstract.", published=date(2022, 1, 1), retrieved_at=NOW, source="pubmed", endpoint="pubmed:efetch",
    )
    data.update(kw)
    return StudyRecord(**data)


# --- PubMed ------------------------------------------------------------------


def test_parse_efetch_metadata_types_mesh_and_retraction():
    records = {r.pmid: r for r in parse_efetch(fixture_bytes("pubmed_efetch.xml"), retrieved_at=NOW)}
    assert set(records) == {"39000001", "39000002", "39000003", "39000004", "39000005", "39000006"}

    meta = records["39000001"]
    assert meta.source_id == "PMID:39000001" and meta.source_url == "https://pubmed.ncbi.nlm.nih.gov/39000001/"
    assert meta.doi == "10.5555/ec.2017.001" and meta.pmcid == "PMC9000001"
    assert {"Meta-Analysis", "Systematic Review"} <= set(meta.publication_types)
    assert meta.mesh_terms == ["Resistance Training", "Muscle, Skeletal", "Hypertrophy"]
    assert meta.published == date(2016, 12, 14)  # electronic date beats the issue date
    assert meta.authors == ["Alder A", "Birch B", "Cedar C"] and meta.journal == "Journal of sports sciences"
    assert meta.abstract.startswith("Objective: To quantify") and "Results: Each additional weekly set" in meta.abstract
    assert meta.source == "pubmed" and meta.endpoint == "pubmed:efetch" and meta.retrieved_at == NOW
    assert not meta.retracted and meta.text_source == "abstract"

    retracted = records["39000004"]
    assert retracted.retracted and "PMID 39000099" in retracted.retraction_notice
    assert records["39000005"].published == date(2012, 11, 1)  # MedlineDate "2012 Nov-Dec"
    assert records["39000005"].authors == ["Juniper J"]  # ValidYN="N" author skipped
    assert records["39000002"].published == date(2022, 3, 1)


def test_parse_efetch_rejects_malformed_xml():
    with pytest.raises(ValueError):
        parse_efetch(b"<PubmedArticleSet><PubmedArticle>")


def test_pubmed_term_keeps_older_syntheses_inside_recency_window():
    term = build_term("creatine strength", TODAY, RecencyPolicy(max_years=10))
    assert term == '(creatine strength) AND ("2016/01/01"[dp] : "3000"[dp] OR meta-analysis[pt] OR systematic review[pt])'
    plain = build_term("x", TODAY, RecencyPolicy(max_years=5, keep_older_syntheses=False))
    assert plain == '(x) AND "2021/01/01"[dp] : "3000"[dp]'


def test_pubmed_search_sends_api_key_and_keeps_esearch_order():
    apis = FakeAPIs()
    tool = PubMedSearch(bare_http(apis), api_key="KEY", email="me@example.org", today=lambda: TODAY)
    resp = tool.search(QUERY, max_results=6)
    assert resp.state == "ok"
    assert [r.pmid for r in resp.records] == ["39000002", "39000001", "39000006", "39000003", "39000004", "39000005"]
    esearch, efetch = apis.requests
    assert esearch.url.params["api_key"] == "KEY" and esearch.url.params["tool"] == "evidence-coach"
    assert esearch.url.params["email"] == "me@example.org" and esearch.url.params["retmax"] == "6"
    assert '"2016/01/01"[dp]' in esearch.url.params["term"]
    assert efetch.url.params["id"].split(",")[0] == "39000002" and efetch.url.params["retmode"] == "xml"


def test_pubmed_no_results_skips_efetch():
    apis = FakeAPIs({"esearch": [json_file("pubmed_esearch_empty.json")]})
    resp = PubMedSearch(bare_http(apis)).search("zzqx")
    assert resp.state == "no_results" and resp.records == []
    assert len(apis.requests) == 1


def test_pubmed_429_backs_off_honours_retry_after_then_succeeds():
    sleeps = Sleeps()
    apis = FakeAPIs({"esearch": [httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(429), json_file("pubmed_esearch.json")]})
    resp = PubMedSearch(bare_http(apis, sleeps, max_retries=3), today=lambda: TODAY).search(QUERY)
    assert resp.state == "ok" and len(resp.records) == 6
    assert sleeps == [2.0, 1.0]  # Retry-After honoured, then exponential backoff (0.5 * 2**1)
    assert len(apis.calls("esearch")) == 3


def test_pubmed_rate_limited_after_retry_cap():
    sleeps = Sleeps()
    apis = FakeAPIs({"esearch": [httpx.Response(429)]})
    resp = PubMedSearch(bare_http(apis, sleeps, max_retries=2)).search(QUERY)
    assert resp.state == "rate_limited" and "after 3 attempts" in resp.message
    assert len(apis.calls("esearch")) == 3 and sleeps == [0.5, 1.0]


def test_backoff_is_capped():
    sleeps = Sleeps()
    apis = FakeAPIs({"esearch": [httpx.Response(429, headers={"Retry-After": "120"})]})
    PubMedSearch(bare_http(apis, sleeps, max_retries=1)).search(QUERY)
    assert sleeps == [8.0]


def test_pubmed_timeout_and_server_error_and_bad_xml_are_states():
    timeout = FakeAPIs({"efetch": [httpx.ReadTimeout("slow")]})
    assert PubMedSearch(bare_http(timeout, max_retries=1)).search(QUERY).state == "timed_out"
    assert len(timeout.calls("efetch")) == 2  # timeouts are retried too

    server = FakeAPIs({"esearch": [httpx.Response(503)]})
    assert PubMedSearch(bare_http(server, max_retries=0)).search(QUERY).state == "error"

    bad = FakeAPIs({"efetch": [httpx.Response(200, content=b"<PubmedArticleSet><oops")]})
    resp = PubMedSearch(bare_http(bad)).search(QUERY)
    assert resp.state == "error" and "malformed" in resp.message

    client_error = FakeAPIs({"esearch": [httpx.Response(400)]})
    assert PubMedSearch(bare_http(client_error)).search(QUERY).state == "error"
    assert len(client_error.requests) == 1  # a 400 is not retried

    connect = FakeAPIs({"esearch": [httpx.ConnectError("refused")]})
    assert PubMedSearch(bare_http(connect, max_retries=0)).search(QUERY).state == "error"


# --- Europe PMC ----------------------------------------------------------------


def test_parse_europepmc_hits_label_preprints_and_open_access():
    data = json_file("europepmc_search.json").json()
    hits = [parse_hit(h, NOW) for h in data["resultList"]["result"]]
    by_id = {h.source_id: h for h in hits if h}
    trial = by_id["PMID:39000002"]
    assert trial.pmcid == "PMC9000002" and trial.access == "open_access" and trial.endpoint == "europepmc:search"
    assert trial.source_url == "https://europepmc.org/article/MED/39000002"
    pmc_only = by_id["DOI:10.5555/ec.2025.007"]
    assert pmc_only.title.endswith("a randomized crossover trial") and "<h4>" not in pmc_only.abstract
    assert pmc_only.authors == ["Larch L", "Maple M"] and pmc_only.published == date(2025, 1, 15)
    preprint = by_id["DOI:10.31236/osf.io/ec008"]
    assert preprint.is_preprint and preprint.journal == "SportRxiv" and preprint.access == "paywalled"
    assert preprint.authors == ["Oak O", "Pine P"]


def test_europepmc_query_and_no_results():
    assert build_query("x", TODAY, RecencyPolicy()) == (
        '(x) AND (PUB_YEAR:[2016 TO 2026] OR PUB_TYPE:"meta-analysis" OR PUB_TYPE:"systematic review")'
    )
    apis = FakeAPIs({"epmc_search": [json_file("europepmc_search_empty.json")]})
    resp = EuropePMCSearch(bare_http(apis)).search("zzqx")
    assert resp.state == "no_results"
    assert apis.requests[0].url.params["resultType"] == "core"


def test_full_text_fetch_paths():
    epmc = EuropePMCSearch(bare_http(FakeAPIs()))
    sections, why = epmc.fetch_full_text("PMC9000002")
    assert why == "ok" and sections.results.startswith("Muscle thickness increased by 5.2%")
    assert epmc.fetch_full_text("PMC1234567") == (None, "not_open_access")  # 404: no OA full text
    review = EuropePMCSearch(bare_http(FakeAPIs({"fulltext": [httpx.Response(200, content=fixture_bytes("europepmc_fulltext_noresults.xml"))]})))
    assert review.fetch_full_text("PMC9000003") == (None, "no_results_section")
    slow = EuropePMCSearch(bare_http(FakeAPIs({"fulltext": [httpx.ReadTimeout("slow")]}), max_retries=0))
    assert slow.fetch_full_text("PMC9000002") == (None, "timed_out")


# --- JATS ----------------------------------------------------------------------


def test_jats_keeps_methods_results_and_funding_only():
    s = parse_jats(fixture_bytes("europepmc_fulltext_PMC9000002.xml"))
    assert s.methods.startswith("Forty resistance-trained women") and "ultrasound" in s.methods
    assert "Participants" not in s.methods  # sub-section titles are separators, not text
    assert s.results == (
        "Muscle thickness increased by 5.2% in the 20-set group and 3.9% in the 10-set group "
        "(difference 1.3%, 95% CI −0.4 to 3.0). Strength gains did not differ between groups."
    )
    assert "999.9" not in s.results  # table cells dropped
    assert s.notes == ["This study received no external funding.", "The authors declare no competing interests."]
    rendered = s.render()
    assert "Introduction" not in rendered and "Doubling volume" not in rendered  # discussion dropped
    assert rendered.index("Methods:") < rendered.index("Results:") < rendered.index("Funding and conflicts:")


def test_jats_titles_without_sec_type_and_nested_studies():
    s = parse_jats(fixture_bytes("europepmc_fulltext_PMC9000007.xml"))
    assert s.methods.startswith("Twenty-four recreationally trained adults")
    assert s.results.startswith("Quadriceps cross-sectional area increased 7.4%")  # "Results and Discussion"
    assert "Beyond 8 weekly sets" not in s.render()  # conclusions dropped
    assert s.notes == ["Larch L is a consultant for a supplement company."]


def test_jats_malformed_and_clipped():
    with pytest.raises(ValueError):
        parse_jats("<article><body>")
    long_results = "<article><body><sec sec-type='results'><p>" + "The effect was 1.5%. " * 400 + "</p></sec></body></article>"
    s = parse_jats(long_results)
    assert len(s.results) < 3_600 and s.results.endswith("[...]")


# --- Crossref / Retraction Watch -------------------------------------------------


def test_crossref_updated_by_from_retraction_watch():
    apis = FakeAPIs()
    checker = CrossrefRetractionChecker(bare_http(apis), mailto="me@example.org")
    result = checker.check("https://doi.org/10.5555/EC.2024.006")
    assert result.checked and result.retracted
    assert result.notice == "retraction notice 10.5555/ec.2025.retraction.006 (retraction-watch)"
    assert apis.requests[0].url.params["mailto"] == "me@example.org"
    assert len(apis.requests) == 1  # updated-by present: no second query
    checker.check("10.5555/ec.2024.006")
    assert len(apis.requests) == 1  # cached


def test_crossref_update_to_notice_and_clean_record():
    apis = FakeAPIs()
    checker = CrossrefRetractionChecker(bare_http(apis))
    notice = checker.check("10.5555/ec.2023.003")
    assert notice.retracted and notice.notice == "retraction notice 10.5555/ec.2023.retraction.003 (publisher)"
    assert apis.calls("crossref_updates")[0].url.params["filter"] == "updates:10.5555/ec.2023.003"
    clean = checker.check("10.5555/ec.2022.002")
    assert clean.checked and not clean.retracted
    assert retraction_in_updates(json_file("crossref_updates_retraction.json").json()["message"]["items"], "10.5555/other") is None


def test_crossref_relation_and_failure_paths():
    relation = FakeAPIs({"crossref_work": [httpx.Response(200, json={"message": {"relation": {"is-retracted-by": [{"id": "10.5555/n"}]}}})]})
    assert CrossrefRetractionChecker(bare_http(relation)).check("10.5555/x").retracted

    missing = CrossrefRetractionChecker(bare_http(FakeAPIs({"crossref_work": [httpx.Response(404)]})))
    result = missing.check("10.5555/x")
    assert result.checked and not result.retracted and "not registered" in result.error

    limited = CrossrefRetractionChecker(bare_http(FakeAPIs({"crossref_work": [httpx.Response(429)]}), max_retries=0))
    result = limited.check("10.5555/x")
    assert not result.checked and result.error.startswith("rate_limited")
    assert "10.5555/x" not in limited._cache  # failures are not cached


# --- Ranking, dedupe, caps -------------------------------------------------------


def test_dedupe_merges_across_sources_by_doi_and_pmid():
    pm = rec("PMID:1", pmid="1", doi="10.1/A", publication_types=["Randomized Controlled Trial"], mesh_terms=["Humans"])
    ep = rec("PMID:1", pmid="1", pmcid="PMC5", source="europepmc", endpoint="europepmc:search", access="open_access", abstract="A longer abstract.")
    by_doi = rec("DOI:10.1/a", doi="https://doi.org/10.1/a", source="europepmc", endpoint="europepmc:search")
    other = rec("PMID:2", pmid="2")
    merged, dropped = dedupe([ep, other, pm, by_doi])
    assert [m.source_id for m in merged] == ["PMID:1", "PMID:2"]
    first = merged[0]
    assert first.source == "pubmed" and first.pmcid == "PMC5" and first.access == "open_access"
    assert first.publication_types == ["Randomized Controlled Trial"] and first.abstract == "A longer abstract."
    assert first.also_found_in == ["europepmc:search"]
    assert [d.reason for d in dropped] == ["duplicate", "duplicate"]
    assert normalize_doi("doi:10.1/ABC") == "10.1/abc"


def test_dedupe_keeps_retraction_from_any_source():
    merged, _ = dedupe([rec("PMID:1", pmid="1"), rec("X", pmid="1", source="europepmc", retracted=True, retraction_notice="n")])
    assert merged[0].retracted and merged[0].retraction_notice == "n"


def test_rank_prefers_recent_and_keeps_old_meta_analyses():
    old_primary = rec("old", published=date(2008, 1, 1))
    old_meta = rec("meta", published=date(2009, 1, 1), publication_types=["Meta-Analysis"])
    recent = rec("recent", published=date(2024, 1, 1), publication_types=["Randomized Controlled Trial"])
    mid = rec("mid", published=date(2018, 1, 1))
    # The old primary study is the most "relevant" hit but still ranks last.
    ranked = rank({"pubmed": [old_primary, old_meta, mid, recent]}, [old_primary, old_meta, mid, recent], TODAY, RecencyPolicy())
    assert [r.source_id for r in ranked][-1] == "old"
    assert set([r.source_id for r in ranked][:2]) == {"meta", "recent"}  # old synthesis kept above mid-aged primary


def test_caps_per_journal_and_author_group():
    records = [
        rec(f"j{i}", journal="Sports Med", authors=[f"First{i} A", f"Senior{i} B"]) for i in range(4)
    ] + [rec(f"lab{i}", journal=f"J{i}", authors=["Lead L", "Schoenfeld BJ"]) for i in range(3)]
    kept, dropped = apply_caps(records, DiversityCaps(per_journal=3, per_author_group=2))
    assert [r.source_id for r in kept] == ["j0", "j1", "j2", "lab0", "lab1"]
    assert {(d.source_id, d.reason) for d in dropped} == {("j3", "journal_cap"), ("lab2", "author_cap")}
    assert author_group(rec("x", authors=["Müller H"])) == "muller h"


# --- Composite -------------------------------------------------------------------


def test_composite_merges_checks_retractions_and_reads_full_text():
    apis = FakeAPIs()
    resp = live_tool(apis).search(QUERY, max_results=10)
    assert resp.state == "ok" and resp.source_states == {"pubmed": "ok", "europepmc": "ok"}
    by_id = {r.source_id: r for r in resp.records}

    # Deduplicated across sources (DOI case differs between the sources for 39000001).
    assert sum(1 for r in resp.records if r.pmid == "39000002") == 1
    trial = by_id["PMID:39000002"]
    assert trial.also_found_in == ["europepmc:search"] and trial.source == "pubmed"
    assert {(d.source_id, d.reason) for d in resp.dropped} >= {("PMID:39000002", "duplicate"), ("PMID:39000001", "duplicate")}

    # Full text from Europe PMC: methods and results sections, with provenance of where it came from.
    assert trial.text_source == "full_text" and trial.access == "open_access"
    assert trial.full_text.startswith("Methods: Forty resistance-trained women")
    assert trial.full_text_url == "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9000002/fullTextXML"
    assert by_id["DOI:10.5555/ec.2025.007"].full_text.startswith("Methods: Twenty-four")
    # 404 on fullTextXML: paywalled, abstract kept and labeled.
    meta = by_id["PMID:39000001"]
    assert meta.access == "paywalled" and meta.full_text is None and meta.text_source == "abstract"

    # Retractions: PubMed publication type, and Crossref / Retraction Watch for one PubMed missed.
    assert by_id["PMID:39000004"].retracted
    rw = by_id["PMID:39000006"]
    assert rw.retracted and "retraction-watch" in rw.retraction_notice and "crossref" in rw.retraction_checked_by
    assert not any(r.pmcid == "PMC9000006" for r in resp.records)
    assert not any("/PMC" in str(q.url) and "39000004" in str(q.url) for q in apis.requests)

    # One lab cannot dominate: the third Cedar paper is capped.
    assert ("PMID:39000003", "author_cap") in {(d.source_id, d.reason) for d in resp.dropped}
    # Preprints are labeled; the old primary study ranks last.
    assert by_id["DOI:10.31236/osf.io/ec008"].is_preprint
    assert resp.records[-1].source_id == "PMID:39000005"
    # Every record carries provenance.
    assert all(r.has_provenance() and r.endpoint for r in resp.records)


def test_composite_one_source_down_is_still_ok():
    apis = FakeAPIs({"esearch": [httpx.Response(429)]})
    resp = live_tool(apis).search(QUERY)
    assert resp.state == "ok" and resp.source_states == {"pubmed": "rate_limited", "europepmc": "ok"}
    assert "pubmed: rate_limited" in resp.message
    assert {r.source for r in resp.records} == {"europepmc"}


@pytest.mark.parametrize(
    "failure,state",
    [(httpx.Response(429), "rate_limited"), (httpx.ReadTimeout("slow"), "timed_out"), (httpx.Response(500), "error")],
)
def test_composite_all_sources_failing_returns_state(failure, state):
    apis = FakeAPIs({"esearch": [failure], "epmc_search": [failure]})
    resp = live_tool(apis).search(QUERY)
    assert resp.state == state and resp.records == []


def test_composite_no_results():
    apis = FakeAPIs({"esearch": [json_file("pubmed_esearch_empty.json")], "epmc_search": [json_file("europepmc_search_empty.json")]})
    resp = live_tool(apis).search("zzqx")
    assert resp.state == "no_results" and resp.source_states == {"pubmed": "no_results", "europepmc": "no_results"}


def test_composite_paywalled_when_no_open_access_full_text():
    apis = FakeAPIs({"fulltext": [httpx.Response(404)]})
    resp = live_tool(apis).search(QUERY)
    assert resp.state == "paywalled" and resp.records
    assert all(r.full_text is None for r in resp.records)
    assert "findings rest on abstracts" in resp.message


def test_composite_full_text_failure_falls_back_to_abstract():
    apis = FakeAPIs({"fulltext": [httpx.ReadTimeout("slow")]})
    resp = live_tool(apis).search(QUERY)
    trial = next(r for r in resp.records if r.pmid == "39000002")
    assert trial.access == "unavailable" and trial.full_text is None
    assert "full text unavailable, abstract used" in resp.message


def test_composite_crossref_down_keeps_records_unverified():
    apis = FakeAPIs({"crossref_work": [httpx.Response(503)]})
    resp = live_tool(apis).search(QUERY)
    assert resp.state == "ok" and "crossref retraction check unavailable" in resp.message
    trial = next(r for r in resp.records if r.pmid == "39000002")
    assert "crossref" not in trial.retraction_checked_by


def test_composite_time_budget_skips_enrichment():
    ticks = iter([0.0] + [100.0] * 100)
    apis = FakeAPIs()
    tool = live_tool(apis, clock=lambda: next(ticks), time_budget_s=10.0)
    resp = tool.search(QUERY)
    assert resp.records and not apis.calls("fulltext") and not apis.calls("crossref_work")
    assert all(r.full_text is None for r in resp.records)


def test_composite_survives_a_crashing_source():
    class Boom:
        name = "boom"

        def search(self, query, max_results=10):
            raise RuntimeError("bug")

    tool = CompositeSearchTool([Boom(), FixtureSearchTool.from_file()], today=lambda: TODAY)
    resp = tool.search("protein intake energy deficit lean mass")
    assert resp.state == "ok" and resp.source_states["boom"] == "error"


def test_sportrxiv_preprints_are_labeled_and_withdrawn_is_retracted():
    apis = FakeAPIs()
    resp = SportRxivSearch(bare_http(apis)).search("weekly volume hypertrophy")
    assert resp.state == "ok"
    ok, withdrawn = resp.records
    assert ok.is_preprint and ok.source_id == "DOI:10.51224/SRXIV.009" and ok.journal == "SportRxiv (preprint)"
    assert withdrawn.retracted and "data error" in withdrawn.retraction_notice
    assert apis.requests[0].url.params["filter[provider]"] == "sportrxiv"


def test_live_tool_includes_sportrxiv_when_enabled():
    apis = FakeAPIs()
    tool = live_tool(apis, settings=SearchSettings(backend="live", include_preprints=True, max_retries=0))
    resp = tool.search(QUERY)
    assert resp.source_states.get("sportrxiv") == "ok"
    assert any(r.source == "sportrxiv" and r.is_preprint for r in resp.records)


# --- HTTP plumbing and config ------------------------------------------------------


def test_rate_limiter_spaces_requests():
    now = [0.0]
    sleeps: list[float] = []
    limiter = RateLimiter(2.0, clock=lambda: now[0], sleep=sleeps.append)
    for _ in range(3):
        limiter.wait()
    assert sleeps == [0.5, 1.0]  # slots at 0, 0.5, 1.0


def test_live_http_applies_ncbi_limit_by_api_key():
    with_key = http_client(FakeAPIs(), SearchSettings(backend="live", ncbi_api_key="k"))
    without = http_client(FakeAPIs(), SearchSettings(backend="live"))
    assert with_key.limiters["eutils.ncbi.nlm.nih.gov"].interval < without.limiters["eutils.ncbi.nlm.nih.gov"].interval
    assert without.limiters["eutils.ncbi.nlm.nih.gov"].interval >= 1 / 3


def test_fetch_error_404_is_not_retried():
    apis = FakeAPIs({"fulltext": [httpx.Response(404)]})
    with pytest.raises(FetchError) as e:
        bare_http(apis).get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML")
    assert e.value.kind == "not_found" and len(apis.requests) == 1


def test_search_settings_from_env():
    assert SearchSettings.from_env({}).backend == "fixture"
    s = SearchSettings.from_env({"SEARCH_BACKEND": "LIVE", "NCBI_API_KEY": "k", "SEARCH_INCLUDE_PREPRINTS": "1"})
    assert s.backend == "live" and s.ncbi_api_key == "k" and s.include_preprints
    with pytest.raises(ValueError):
        SearchSettings.from_env({"SEARCH_BACKEND": "tavily"})


def test_build_search_tool_selects_backend(monkeypatch):
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    assert isinstance(build_search_tool(), FixtureSearchTool)
    live = build_search_tool(SearchSettings(backend="live"))
    assert isinstance(live, CompositeSearchTool) and [s.name for s in live.sources] == ["pubmed", "europepmc"]

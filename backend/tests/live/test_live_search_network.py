"""Real PubMed / Europe PMC / Crossref calls. Skipped unless RUN_LIVE_NETWORK=1 (no LLM spend).

    RUN_LIVE_NETWORK=1 NCBI_API_KEY=... SEARCH_CONTACT_EMAIL=you@example.org \
        .venv/bin/python -m pytest -m live_network -s tests/live/test_live_search_network.py
"""

from __future__ import annotations

import os

import pytest

from backend.config import SearchSettings
from backend.tools.factory import build_live_search_tool, live_http
from backend.tools.retraction import CrossrefRetractionChecker

pytestmark = pytest.mark.live_network


@pytest.fixture(scope="module")
def settings() -> SearchSettings:
    return SearchSettings.from_env({**os.environ, "SEARCH_BACKEND": "live"})


def test_live_search_returns_provenanced_records(settings):
    tool = build_live_search_tool(settings)
    resp = tool.search("resistance training volume hypertrophy", max_results=8)
    print(resp.state, resp.source_states, resp.message)
    for r in resp.records:
        print(f"  {r.source_id} {r.published} {r.access} {r.text_source} retracted={r.retracted} {r.title[:70]}")
    assert resp.state in ("ok", "paywalled"), resp.message
    assert resp.records and all(r.has_provenance() and r.endpoint for r in resp.records)
    assert {"pubmed", "europepmc"} & {r.source for r in resp.records}
    assert any(r.full_text and "Results:" in r.full_text for r in resp.records), "expected at least one OA full text"


def test_live_crossref_knows_a_retraction(settings):
    # Wakefield et al., Lancet 1998 (retracted 2010): a stable, well-known retraction.
    checker = CrossrefRetractionChecker(live_http(settings), mailto=settings.contact_email)
    result = checker.check("10.1016/S0140-6736(97)11096-0")
    print(result)
    assert result.checked and result.retracted

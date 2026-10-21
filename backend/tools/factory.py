"""Build the configured SearchTool: `SEARCH_BACKEND=fixture` (default) or `live`."""

from __future__ import annotations

import httpx

from backend.config import SearchSettings
from backend.tools import europepmc, pubmed, retraction, sportrxiv
from backend.tools.composite import CompositeSearchTool
from backend.tools.europepmc import EuropePMCSearch
from backend.tools.http import USER_AGENT, HttpClient, RateLimiter
from backend.tools.pubmed import PubMedSearch
from backend.tools.retraction import CrossrefRetractionChecker
from backend.tools.search import FixtureSearchTool, SearchTool
from backend.tools.sportrxiv import SportRxivSearch


def live_http(settings: SearchSettings, transport: httpx.BaseTransport | None = None, **kw) -> HttpClient:
    """One client for all sources, with per-host rate limits shared by every researcher thread."""
    client = httpx.Client(
        transport=transport, timeout=settings.timeout_s, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    )
    return HttpClient(
        client,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        rate_limits={
            pubmed.HOST: pubmed.rate_limit(settings.ncbi_api_key),
            europepmc.HOST: 8.0,
            retraction.HOST: 10.0,  # Crossref polite pool
            sportrxiv.HOST: 5.0,
        },
        concurrency={retraction.HOST: 3},  # Crossref polite pool: X-Concurrency-Limit 3
        **kw,
    )


def build_live_search_tool(settings: SearchSettings, http: HttpClient | None = None) -> CompositeSearchTool:
    http = http or live_http(settings)
    epmc = EuropePMCSearch(http, email=settings.contact_email)
    sources: list[SearchTool] = [PubMedSearch(http, api_key=settings.ncbi_api_key, email=settings.contact_email), epmc]
    if settings.include_preprints:
        sources.append(SportRxivSearch(http))
    return CompositeSearchTool(
        sources,
        full_text=epmc,
        retraction=CrossrefRetractionChecker(
            http, mailto=settings.contact_email, list_limiter=RateLimiter(retraction.LIST_QUERIES_PER_SECOND, sleep=http.sleep)
        ),
        time_budget_s=settings.time_budget_s,
    )


def build_search_tool(settings: SearchSettings | None = None) -> SearchTool:
    settings = settings or SearchSettings.from_env()
    if settings.backend == "live":
        return build_live_search_tool(settings)
    return FixtureSearchTool.from_file()

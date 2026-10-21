"""Mocked HTTP for the live search tools (httpx.MockTransport), offline.

Fixtures in tests/fixtures/live/ are handwritten in the documented response
formats of PubMed E-utilities (esearch JSON, efetch PubmedArticleSet XML),
Europe PMC REST (search JSON with resultType=core, fullTextXML JATS), Crossref
REST (/works/{doi}, /works?filter=updates:{doi}) and the OSF preprints API
(SportRxiv). They were written by hand because the sandbox cannot reach these
APIs. The papers are SYNTHETIC: PMIDs 39000001-39000006, PMCIDs
PMC9000001-PMC9000007 and DOIs under the 10.5555 test prefix are not real.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import httpx

from backend.config import SearchSettings
from backend.tools.factory import build_live_search_tool, live_http
from backend.tools.http import HttpClient

LIVE = Path(__file__).parent / "fixtures" / "live"
TODAY = date(2026, 10, 19)

Handler = Callable[[httpx.Request], httpx.Response]


def fixture_bytes(name: str) -> bytes:
    return (LIVE / name).read_bytes()


def _file(name: str, content_type: str) -> httpx.Response:
    return httpx.Response(200, content=fixture_bytes(name), headers={"Content-Type": content_type})


def json_file(name: str) -> httpx.Response:
    return _file(name, "application/json")


def xml_file(name: str) -> httpx.Response:
    return _file(name, "text/xml")


class FakeAPIs:
    """Routes requests to fixtures. `overrides` maps a route key ("esearch", "efetch",
    "epmc_search", "fulltext", "crossref_work", "crossref_updates", "osf") to a list of
    responses / exceptions consumed in order (the last one repeats)."""

    FULL_TEXT = {"PMC9000002": "europepmc_fulltext_PMC9000002.xml", "PMC9000007": "europepmc_fulltext_PMC9000007.xml"}
    RETRACTED_WORKS = {"10.5555/ec.2024.006": "crossref_work_retracted.json"}
    RETRACTION_NOTICES = {"10.5555/ec.2023.003": "crossref_updates_retraction.json"}

    def __init__(self, overrides: dict[str, list[httpx.Response | Exception]] | None = None):
        self.overrides = {k: list(v) for k, v in (overrides or {}).items()}
        self.requests: list[httpx.Request] = []

    def calls(self, route: str) -> list[httpx.Request]:
        return [r for r in self.requests if self.route(r) == route]

    @staticmethod
    def route(request: httpx.Request) -> str:
        host, path = request.url.host, request.url.path
        if host == "eutils.ncbi.nlm.nih.gov":
            return "esearch" if path.endswith("esearch.fcgi") else "efetch"
        if host == "www.ebi.ac.uk":
            return "fulltext" if path.endswith("/fullTextXML") else "epmc_search"
        if host == "api.crossref.org":
            return "crossref_updates" if path.rstrip("/") == "/works" else "crossref_work"
        if host == "api.osf.io":
            return "osf"
        return "unknown"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = self.route(request)
        queue = self.overrides.get(key)
        if queue:
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item
        return self.default(key, request)

    def default(self, key: str, request: httpx.Request) -> httpx.Response:
        if key == "esearch":
            return json_file("pubmed_esearch.json")
        if key == "efetch":
            return xml_file("pubmed_efetch.xml")
        if key == "epmc_search":
            return json_file("europepmc_search.json")
        if key == "fulltext":
            pmcid = request.url.path.split("/")[-2]
            return xml_file(self.FULL_TEXT[pmcid]) if pmcid in self.FULL_TEXT else httpx.Response(404)
        if key == "crossref_work":
            doi = unquote(request.url.path.split("/works/", 1)[1]).lower()
            if doi in self.RETRACTED_WORKS:
                return json_file(self.RETRACTED_WORKS[doi])
            return httpx.Response(200, json={"status": "ok", "message": {"DOI": doi, "type": "journal-article"}})
        if key == "crossref_updates":  # filters of one name are ORed: updates:a,updates:b
            dois = [f.split("updates:", 1)[1] for f in request.url.params["filter"].split(",")]
            items = [i for d in dois if d in self.RETRACTION_NOTICES for i in json_file(self.RETRACTION_NOTICES[d]).json()["message"]["items"]]
            return httpx.Response(200, json={"status": "ok", "message": {"total-results": len(items), "items": items}})
        if key == "osf":
            return json_file("osf_sportrxiv_preprints.json")
        return httpx.Response(404)


class Sleeps(list):
    """Records requested sleeps instead of sleeping."""

    def __call__(self, seconds: float) -> None:
        self.append(round(seconds, 3))


def http_client(apis: FakeAPIs, settings: SearchSettings | None = None, sleeps: Sleeps | None = None) -> HttpClient:
    settings = settings or SearchSettings(backend="live", max_retries=2)
    return live_http(settings, transport=httpx.MockTransport(apis), sleep=sleeps if sleeps is not None else Sleeps(), jitter=lambda: 0.0)


def live_tool(apis: FakeAPIs, settings: SearchSettings | None = None, sleeps: Sleeps | None = None, **kw):
    settings = settings or SearchSettings(backend="live", max_retries=2)
    tool = build_live_search_tool(settings, http=http_client(apis, settings, sleeps))
    tool.today = lambda: TODAY
    for s in tool.sources:
        s.today = lambda: TODAY  # type: ignore[attr-defined]
    for k, v in kw.items():
        setattr(tool, k, v)
    return tool

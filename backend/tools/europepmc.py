"""Europe PMC REST: search (JSON, resultType=core) and open-access full text (JATS XML).

Search adds PMCIDs and open-access flags PubMed lacks, and indexes preprints
(source "PPR"), which are labeled as such. `fetch_full_text` pulls the
methods and results sections from `/{PMCID}/fullTextXML`.

Docs: https://europepmc.org/RestfulWebService
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from backend.schemas import SearchResponse, StudyRecord, utcnow
from backend.tools.http import FetchError, HttpClient
from backend.tools.jats import FullTextSections, parse_jats
from backend.tools.ranking import RecencyPolicy

REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SEARCH = f"{REST}/search"
HOST = "www.ebi.ac.uk"


def build_query(query: str, today: date, policy: RecencyPolicy) -> str:
    window = f"PUB_YEAR:[{policy.min_year(today)} TO {today.year}]"
    if policy.keep_older_syntheses:
        window = f'({window} OR PUB_TYPE:"meta-analysis" OR PUB_TYPE:"systematic review")'
    return f"({query}) AND {window}"


def full_text_url(pmcid: str) -> str:
    return f"{REST}/{pmcid}/fullTextXML"


def _date(hit: dict[str, Any]) -> date | None:
    for key in ("firstPublicationDate", "electronicPublicationDate"):
        value = hit.get(key)
        if value:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                pass
    year = hit.get("pubYear")
    return date(int(year), 1, 1) if year and str(year).isdigit() else None


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _authors(hit: dict[str, Any]) -> list[str]:
    authors = (hit.get("authorList") or {}).get("author") or []
    out = []
    for a in authors:
        if a.get("lastName"):
            out.append(f"{a['lastName']} {a.get('initials', '')}".strip())
        elif a.get("collectiveName"):
            out.append(a["collectiveName"])
    if not out and hit.get("authorString"):
        out = [s.strip().rstrip(".") for s in hit["authorString"].split(",") if s.strip()]
    return out


def parse_hit(hit: dict[str, Any], retrieved_at: datetime) -> StudyRecord | None:
    published = _date(hit)
    title = _strip_tags(hit.get("title", ""))
    if not published or not title:
        return None
    src, ext_id = hit.get("source"), hit.get("id")
    pmid, doi, pmcid = hit.get("pmid"), hit.get("doi"), hit.get("pmcid")
    pub_types = list((hit.get("pubTypeList") or {}).get("pubType") or [])
    is_preprint = src == "PPR" or "preprint" in {t.lower() for t in pub_types}
    if pmid:
        source_id, url = f"PMID:{pmid}", f"https://europepmc.org/article/MED/{pmid}"
    elif doi:
        source_id, url = f"DOI:{doi}", f"https://doi.org/{doi}"
    else:
        source_id, url = f"EPMC:{src}/{ext_id}", f"https://europepmc.org/article/{src}/{ext_id}"
    journal = ((hit.get("journalInfo") or {}).get("journal") or {}).get("title")
    if is_preprint:
        journal = journal or (hit.get("bookOrReportDetails") or {}).get("publisher") or "preprint server"
    open_access = hit.get("isOpenAccess") == "Y" and bool(pmcid)
    retracted = "retracted publication" in {t.lower() for t in pub_types}
    return StudyRecord(
        source_id=source_id,
        source_url=url,
        title=title,
        abstract=_strip_tags(hit.get("abstractText", "")),
        journal=journal,
        authors=_authors(hit),
        published=published,
        publication_types=pub_types,
        retracted=retracted,
        is_preprint=is_preprint,
        pmid=pmid,
        doi=doi,
        pmcid=pmcid,
        mesh_terms=[m.get("descriptorName", "") for m in (hit.get("meshHeadingList") or {}).get("meshHeading") or []],
        source="europepmc",
        endpoint="europepmc:search",
        retrieved_at=retrieved_at,
        access="open_access" if open_access else "paywalled",
        retraction_checked_by=["europepmc:pub_type"],
        retraction_notice="Europe PMC publication type: Retracted Publication" if retracted else None,
    )

FULL_TEXT_CACHE_SIZE = 500  # parsed articles kept in memory per process


class EuropePMCSearch:
    """`SearchTool` over Europe PMC search, plus the full-text fetcher the composite uses."""

    name = "europepmc"

    def __init__(
        self,
        http: HttpClient,
        email: str | None = None,
        policy: RecencyPolicy = RecencyPolicy(),
        today: Callable[[], date] = date.today,
    ):
        self.http = http
        self.email = email
        self.policy = policy
        self.today = today
        # Full text by PMCID, shared by parallel researchers; transient failures are not cached.
        self._full_text: dict[str, tuple[FullTextSections | None, str]] = {}
        self._lock = threading.Lock()

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        params: dict[str, Any] = {
            "query": build_query(query, self.today(), self.policy),
            "format": "json",
            "resultType": "core",
            "pageSize": max_results,
        }
        if self.email:
            params["email"] = self.email
        try:
            data = self.http.get(SEARCH, params).json()
        except FetchError as e:
            return SearchResponse(query=query, state="no_results" if e.kind == "not_found" else e.kind, message=str(e))
        except ValueError as e:
            return SearchResponse(query=query, state="error", message=f"europepmc: {e}")
        retrieved_at = utcnow()
        hits = (data.get("resultList") or {}).get("result") or []
        records = [r for h in hits if (r := parse_hit(h, retrieved_at)) is not None]
        if not records:
            return SearchResponse(query=query, state="no_results", message=f"europepmc: hitCount={data.get('hitCount', 0)}")
        return SearchResponse(query=query, state="ok", records=records[:max_results])

    def fetch_full_text(self, pmcid: str) -> tuple[FullTextSections | None, str]:
        """Methods/results for an open-access article. Returns (sections or None, why).

        `why` is "ok", "not_open_access" (404: no OA full text, i.e. paywalled for
        us), "no_results_section", or a FetchError kind.
        """
        with self._lock:
            if pmcid in self._full_text:
                return self._full_text[pmcid]
        result = self._fetch_full_text(pmcid)
        if result[1] in ("ok", "not_open_access", "no_results_section"):
            with self._lock:
                self._full_text[pmcid] = result
                if len(self._full_text) > FULL_TEXT_CACHE_SIZE:
                    self._full_text.pop(next(iter(self._full_text)))  # drop the oldest
        return result

    def _fetch_full_text(self, pmcid: str) -> tuple[FullTextSections | None, str]:
        try:
            resp = self.http.get(full_text_url(pmcid))
        except FetchError as e:
            return None, "not_open_access" if e.kind == "not_found" else e.kind
        try:
            sections = parse_jats(resp.content)
        except ValueError:
            return None, "error"
        if not sections.usable:
            return None, "no_results_section"
        return sections, "ok"

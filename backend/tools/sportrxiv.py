"""SportRxiv preprints via the OSF API (SportRxiv is hosted on OSF Preprints).

Optional source: off unless SEARCH_INCLUDE_PREPRINTS=1. Every record is
labeled `is_preprint=True`, so the Researcher types it "preprint" and the
supervisor never lets it carry a recommendation alone. Withdrawn preprints are
marked retracted.

Docs: https://developer.osf.io/#tag/Preprints
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from backend.schemas import SearchResponse, StudyRecord, utcnow
from backend.tools.http import FetchError, HttpClient

PREPRINTS = "https://api.osf.io/v2/preprints/"
HOST = "api.osf.io"
PROVIDER = "sportrxiv"


def _keywords(query: str) -> str:
    """OSF's title filter is a substring match, so search on the longest content word pair."""
    words = [w for w in re.findall(r"[A-Za-z0-9-]+", query) if len(w) > 3]
    return " ".join(sorted(words, key=len, reverse=True)[:1]) or query


def parse_preprint(item: dict[str, Any], retrieved_at) -> StudyRecord | None:
    a = item.get("attributes") or {}
    title = (a.get("title") or "").strip()
    raw_date = a.get("date_published") or a.get("date_created")
    if not title or not raw_date:
        return None
    pid = item.get("id", "")
    links = item.get("links") or {}
    doi = None
    preprint_doi = links.get("preprint_doi") or ""
    if "doi.org/" in preprint_doi:
        doi = preprint_doi.split("doi.org/", 1)[1]
    withdrawn = bool(a.get("date_withdrawn"))
    return StudyRecord(
        source_id=f"DOI:{doi}" if doi else f"SPORTRXIV:{pid}",
        source_url=links.get("html") or f"https://osf.io/preprints/sportrxiv/{pid}",
        title=title,
        abstract=(a.get("description") or "").strip(),
        journal="SportRxiv (preprint)",
        published=date.fromisoformat(raw_date[:10]),
        publication_types=["Preprint"],
        retracted=withdrawn,
        is_preprint=True,
        doi=doi,
        source="sportrxiv",
        endpoint="sportrxiv:osf_preprints",
        retrieved_at=retrieved_at,
        access="unknown",
        retraction_checked_by=["sportrxiv:withdrawn"],
        retraction_notice=f"withdrawn: {a.get('withdrawal_justification') or 'no reason given'}" if withdrawn else None,
    )


class SportRxivSearch:
    name = "sportrxiv"

    def __init__(self, http: HttpClient):
        self.http = http

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        params = {"filter[provider]": PROVIDER, "filter[title]": _keywords(query), "page[size]": max_results}
        try:
            data = self.http.get(PREPRINTS, params).json()
        except FetchError as e:
            return SearchResponse(query=query, state="no_results" if e.kind == "not_found" else e.kind, message=str(e))
        except ValueError as e:
            return SearchResponse(query=query, state="error", message=f"sportrxiv: {e}")
        retrieved_at = utcnow()
        records = [r for item in data.get("data") or [] if (r := parse_preprint(item, retrieved_at))]
        if not records:
            return SearchResponse(query=query, state="no_results")
        return SearchResponse(query=query, state="ok", records=records[:max_results])

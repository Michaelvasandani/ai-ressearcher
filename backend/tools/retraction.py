"""Retraction checks via Crossref (which carries the Retraction Watch database).

A DOI is retracted when:
- its Crossref work lists an `updated-by` entry of a retraction type (Crossref
  adds these for publisher notices and for Retraction Watch records), or a
  `relation` that points at a retraction notice; or, failing that,
- some notice's `update-to` points at it: `GET /works?filter=updates:{doi}`.

PubMed's "Retracted Publication" publication type and RetractionIn links are
read by the PubMed parser; this adds the Crossref / Retraction Watch check for
everything with a DOI (including Europe PMC-only and preprint records).

Docs: https://api.crossref.org/swagger-ui/index.html ,
https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from backend.tools.http import FetchError, HttpClient
from backend.tools.ranking import normalize_doi

WORKS = "https://api.crossref.org/works"
HOST = "api.crossref.org"
RETRACTION_TYPES = frozenset({"retraction", "withdrawal", "removal"})
RELATION_KEYS = frozenset({"is-retracted-by", "has-retraction"})


@dataclass(frozen=True)
class RetractionCheck:
    doi: str
    checked: bool  # False when Crossref could not be reached; the record is kept but unverified
    retracted: bool = False
    notice: str | None = None  # notice DOI and source, e.g. "10.1/x (retraction-watch)"
    error: str | None = None


def _is_retraction(entry: dict[str, Any]) -> bool:
    return str(entry.get("type", "")).lower().replace("-", "_") in RETRACTION_TYPES


def _describe(entry: dict[str, Any], doi_key: str = "DOI") -> str:
    source = entry.get("source") or "publisher"
    return f"{entry.get('type', 'retraction')} notice {entry.get(doi_key) or entry.get('id') or ''} ({source})".strip()


def retraction_in_work(message: dict[str, Any]) -> str | None:
    """Notice description if a Crossref work record says it was retracted, else None."""
    for entry in message.get("updated-by") or []:
        if _is_retraction(entry):
            return _describe(entry)
    for key, targets in (message.get("relation") or {}).items():
        if key.lower() in RELATION_KEYS and targets:
            return f"relation {key}: {targets[0].get('id', '')}".strip()
    return None


def retraction_in_updates(items: list[dict[str, Any]], doi: str) -> str | None:
    """Notice description if any item's `update-to` retracts `doi`."""
    for item in items:
        for entry in item.get("update-to") or []:
            if normalize_doi(entry.get("DOI")) == doi and _is_retraction(entry):
                return f"{entry.get('type')} notice {item.get('DOI', '')} ({entry.get('source') or 'publisher'})"
    return None


class CrossrefRetractionChecker:
    """Thread-safe, cached per DOI."""

    name = "crossref"

    def __init__(self, http: HttpClient, mailto: str | None = None):
        self.http = http
        self.mailto = mailto
        self._cache: dict[str, RetractionCheck] = {}
        self._lock = threading.Lock()

    def _params(self, **kw: Any) -> dict[str, Any]:
        return {**kw, **({"mailto": self.mailto} if self.mailto else {})}

    def check(self, raw_doi: str) -> RetractionCheck:
        doi = normalize_doi(raw_doi) or raw_doi
        with self._lock:
            if doi in self._cache:
                return self._cache[doi]
        result = self._check(doi)
        if result.checked:
            with self._lock:
                self._cache[doi] = result
        return result

    def _check(self, doi: str) -> RetractionCheck:
        try:
            work = self.http.get(f"{WORKS}/{quote(doi, safe='/')}", self._params()).json().get("message") or {}
            notice = retraction_in_work(work)
            if notice:
                return RetractionCheck(doi=doi, checked=True, retracted=True, notice=notice)
            if "updated-by" in work:  # the work lists its updates: absence of a retraction is conclusive
                return RetractionCheck(doi=doi, checked=True)
            data = self.http.get(WORKS, self._params(filter=f"updates:{doi}", rows=20)).json()
            notice = retraction_in_updates((data.get("message") or {}).get("items") or [], doi)
            return RetractionCheck(doi=doi, checked=True, retracted=bool(notice), notice=notice)
        except FetchError as e:
            if e.kind == "not_found":  # DOI unknown to Crossref (e.g. DataCite): nothing to find
                return RetractionCheck(doi=doi, checked=True, error="not registered with Crossref")
            return RetractionCheck(doi=doi, checked=False, error=f"{e.kind}: {e}")
        except ValueError as e:
            return RetractionCheck(doi=doi, checked=False, error=f"malformed Crossref response: {e}")

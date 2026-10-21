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
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from backend.tools.http import FetchError, HttpClient, RateLimiter
from backend.tools.ranking import normalize_doi

WORKS = "https://api.crossref.org/works"
HOST = "api.crossref.org"
RETRACTION_TYPES = frozenset({"retraction", "withdrawal", "removal"})
RELATION_KEYS = frozenset({"is-retracted-by", "has-retraction"})
# Polite pool: 10 requests/second for /works/{doi}, but list queries (?filter=) are refused unless sent
# one at a time at ~3/second (measured 2026-10: parallel list queries get 429 even under 3/second).
LIST_QUERIES_PER_SECOND = 3.0
WORK_LOOKUP_WORKERS = 3  # Crossref polite pool: X-Concurrency-Limit 3
UPDATES_BATCH = 20  # DOIs per batched notices query
UPDATES_ROWS = 1000  # Crossref's maximum page; notices for 20 DOIs fit easily
# A failed check is reused this long, so parallel researchers do not retry a rate-limited DOI.
FAILURE_TTL_S = 120.0


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
    """Thread-safe, cached per DOI (failed checks only for FAILURE_TTL_S)."""

    name = "crossref"

    def __init__(
        self,
        http: HttpClient,
        mailto: str | None = None,
        failure_ttl_s: float = FAILURE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        list_limiter: RateLimiter | None = None,  # spacing for the filter=updates query
    ):
        self.http = http
        self.list_limiter = list_limiter
        self.mailto = mailto
        self.failure_ttl_s = failure_ttl_s
        self.clock = clock
        self._cache: dict[str, RetractionCheck] = {}
        self._failures: dict[str, tuple[float, RetractionCheck]] = {}  # doi -> (expires at, result)
        self._lock = threading.Lock()
        self._list_lock = threading.Lock()  # list queries go one at a time

    def _params(self, **kw: Any) -> dict[str, Any]:
        return {**kw, **({"mailto": self.mailto} if self.mailto else {})}

    def check(self, raw_doi: str) -> RetractionCheck:
        return self.check_many([raw_doi])[raw_doi]

    def check_many(self, raw_dois: list[str]) -> dict[str, RetractionCheck]:
        """Results keyed by the DOIs as given. Notices for all DOIs that need one come from a
        single batched list query (Crossref ORs repeated filters), which is the scarce request."""
        dois = {raw: normalize_doi(raw) or raw for raw in raw_dois}
        found: dict[str, RetractionCheck] = {}
        with self._lock:
            for doi in set(dois.values()):
                failed = self._failures.get(doi)
                if doi in self._cache:
                    found[doi] = self._cache[doi]
                elif failed and failed[0] > self.clock():
                    found[doi] = failed[1]
        pending = [d for d in dict.fromkeys(dois.values()) if d not in found]
        need_updates: list[str] = []
        with ThreadPoolExecutor(max_workers=WORK_LOOKUP_WORKERS) as pool:
            for doi, result in zip(pending, pool.map(self._check_work, pending)):
                if result is None:
                    need_updates.append(doi)
                else:
                    found[doi] = result
        for i in range(0, len(need_updates), UPDATES_BATCH):
            found |= self._check_updates(need_updates[i : i + UPDATES_BATCH])
        with self._lock:
            for doi in pending:
                result = found[doi]
                if result.checked:
                    self._cache[doi] = result
                    self._failures.pop(doi, None)
                else:
                    self._failures[doi] = (self.clock() + self.failure_ttl_s, result)
        return {raw: found[doi] for raw, doi in dois.items()}

    def _check_work(self, doi: str) -> RetractionCheck | None:
        """The work record's verdict, or None when only a notices query can tell."""
        try:
            work = self.http.get(f"{WORKS}/{quote(doi, safe='/')}", self._params()).json().get("message") or {}
        except FetchError as e:
            return self._failed(doi, e)
        except ValueError as e:
            return RetractionCheck(doi=doi, checked=False, error=f"malformed Crossref response: {e}")
        notice = retraction_in_work(work)
        if notice:
            return RetractionCheck(doi=doi, checked=True, retracted=True, notice=notice)
        if "updated-by" in work:  # the work lists its updates: absence of a retraction is conclusive
            return RetractionCheck(doi=doi, checked=True)
        return None

    def _check_updates(self, dois: list[str]) -> dict[str, RetractionCheck]:
        """One list query for notices that update any of `dois`."""
        params = self._params(filter=",".join(f"updates:{d}" for d in dois), rows=UPDATES_ROWS)
        try:
            with self._list_lock:
                if self.list_limiter:
                    self.list_limiter.wait()
                data = self.http.get(WORKS, params).json()
        except FetchError as e:
            return {d: self._failed(d, e) for d in dois}
        except ValueError as e:
            return {d: RetractionCheck(doi=d, checked=False, error=f"malformed Crossref response: {e}") for d in dois}
        items = (data.get("message") or {}).get("items") or []
        out = {}
        for doi in dois:
            notice = retraction_in_updates(items, doi)
            out[doi] = RetractionCheck(doi=doi, checked=True, retracted=bool(notice), notice=notice)
        return out

    @staticmethod
    def _failed(doi: str, e: FetchError) -> RetractionCheck:
        if e.kind == "not_found":  # DOI unknown to Crossref (e.g. DataCite): nothing to find
            return RetractionCheck(doi=doi, checked=True, error="not registered with Crossref")
        return RetractionCheck(doi=doi, checked=False, error=f"{e.kind}: {e}")

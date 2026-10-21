"""The live `SearchTool`: query every source, merge, check, and read the full text.

One `search(query)` call:
1. Queries each source (PubMed, Europe PMC, optionally SportRxiv) in parallel.
2. Deduplicates by DOI / PMID / PMCID, merging what each source knows.
3. Ranks by fused relevance, recency and design; caps per journal and per
   author group.
4. Checks the top records for retractions in Crossref / Retraction Watch.
5. Fetches open-access full text from Europe PMC and keeps the methods and
   results sections; records without it fall back to the abstract, labeled.

It never raises. The response state is:
- "ok" when any source returned records (other sources' failures are in
  `source_states` and `message`);
- "paywalled" when records came back but none has open-access full text;
- otherwise the most actionable failure across sources (rate_limited >
  timed_out > error), or "no_results".
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from backend.schemas import SearchResponse, SearchState, StudyRecord
from backend.tools.europepmc import EuropePMCSearch, full_text_url
from backend.tools.ranking import DiversityCaps, RecencyPolicy, apply_caps, dedupe, rank
from backend.tools.retraction import CrossrefRetractionChecker
from backend.tools.search import SearchTool

FAILURE_PRIORITY: tuple[SearchState, ...] = ("rate_limited", "timed_out", "error")


def _safe_search(tool: SearchTool, query: str, max_results: int) -> SearchResponse:
    try:
        return tool.search(query, max_results=max_results)
    except Exception as e:  # a source bug must not take the other sources down
        return SearchResponse(query=query, state="error", message=f"{type(e).__name__}: {e}")


class CompositeSearchTool:
    name = "live"

    def __init__(
        self,
        sources: Sequence[SearchTool],
        *,
        full_text: EuropePMCSearch | None = None,
        retraction: CrossrefRetractionChecker | None = None,
        policy: RecencyPolicy = RecencyPolicy(),
        caps: DiversityCaps = DiversityCaps(),
        per_source_results: int | None = None,
        max_full_text: int = 8,
        time_budget_s: float = 60.0,
        workers: int = 4,
        today: Callable[[], date] = date.today,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.sources = list(sources)
        self.full_text = full_text
        self.retraction = retraction
        self.policy = policy
        self.caps = caps
        self.per_source_results = per_source_results
        self.max_full_text = max_full_text
        self.time_budget_s = time_budget_s
        self.workers = workers
        self.today = today
        self.clock = clock

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        deadline = self.clock() + self.time_budget_s
        per_source = self.per_source_results or max_results * 2  # headroom for dedupe and caps
        with ThreadPoolExecutor(max_workers=max(len(self.sources), 1)) as pool:
            responses = list(pool.map(lambda s: _safe_search(s, query, per_source), self.sources))
        source_states = {s.name: r.state for s, r in zip(self.sources, responses)}
        notes = [f"{s.name}: {r.state}" + (f" ({r.message})" if r.message else "") for s, r in zip(self.sources, responses) if r.state != "ok"]
        by_source = {s.name: r.records for s, r in zip(self.sources, responses)}
        all_records = [rec for r in responses for rec in r.records]

        if not all_records:
            states = {r.state for r in responses}
            state: SearchState = next((s for s in FAILURE_PRIORITY if s in states), "no_results")
            return SearchResponse(query=query, state=state, message="; ".join(notes) or None, source_states=source_states)

        merged, dropped = dedupe(all_records)
        ranked = rank(by_source, merged, self.today(), self.policy)
        capped, cap_drops = apply_caps(ranked, self.caps)
        top = capped[:max_results]
        dropped = [*dropped, *cap_drops]

        top = self._check_retractions(top, deadline, notes)
        top = self._attach_full_text(top, deadline, notes)

        live = [r for r in top if not r.retracted]
        state = "ok"
        if live and not any(r.full_text for r in live) and any(r.access == "paywalled" for r in live):
            state = "paywalled"
            notes.append("no open-access full text: findings rest on abstracts")
        return SearchResponse(
            query=query,
            state=state,
            records=top,
            message="; ".join(notes) or None,
            source_states=source_states,
            dropped=dropped,
        )

    # --- enrichment ----------------------------------------------------------

    def _map(self, fn: Callable[[StudyRecord], StudyRecord], records: list[StudyRecord]) -> list[StudyRecord]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(fn, records))

    def _check_retractions(self, records: list[StudyRecord], deadline: float, notes: list[str]) -> list[StudyRecord]:
        checker = self.retraction
        if checker is None:
            return records
        unverified: list[str] = []
        to_check = [r.doi for r in records if r.doi and not r.retracted]
        results = checker.check_many(to_check) if to_check and self.clock() < deadline else {}

        def check(r: StudyRecord) -> StudyRecord:
            if r.retracted or not r.doi:
                return r
            result = results.get(r.doi)
            if result is None:  # out of time before the check
                unverified.append(r.source_id)
                return r
            if not result.checked:
                unverified.append(r.source_id)
                return r
            update: dict = {"retraction_checked_by": [*r.retraction_checked_by, "crossref"]}
            if result.retracted:
                update |= {"retracted": True, "retraction_notice": result.notice}
            return r.model_copy(update=update)

        out = [check(r) for r in records]
        if unverified:
            notes.append(f"crossref retraction check unavailable for {len(unverified)} record(s)")
        return out

    def _attach_full_text(self, records: list[StudyRecord], deadline: float, notes: list[str]) -> list[StudyRecord]:
        fetcher = self.full_text
        failed: list[str] = []
        lock = threading.Lock()
        budget = {"left": self.max_full_text}

        def take_slot() -> bool:
            with lock:
                if budget["left"] <= 0:
                    return False
                budget["left"] -= 1
                return True

        def fetch(r: StudyRecord) -> StudyRecord:
            if r.full_text or r.retracted:
                return r
            if not r.pmcid or r.access == "paywalled" or fetcher is None:
                return r.model_copy(update={"access": "paywalled" if r.access in ("unknown", "paywalled") else r.access})
            if self.clock() >= deadline or not take_slot():
                return r.model_copy(update={"access": "unavailable"})
            sections, why = fetcher.fetch_full_text(r.pmcid)
            if sections is not None:
                return r.model_copy(
                    update={"full_text": sections.render(), "full_text_url": full_text_url(r.pmcid), "access": "open_access"}
                )
            if why == "not_open_access":
                return r.model_copy(update={"access": "paywalled"})
            failed.append(f"{r.source_id} ({why})")
            return r.model_copy(update={"access": "unavailable"})

        out = self._map(fetch, records)
        if failed:
            notes.append(f"full text unavailable, abstract used: {', '.join(failed[:5])}")
        return out


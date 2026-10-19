"""Literature search interface.

`SearchTool` is what the Researcher depends on. Implementations:
- `FixtureSearchTool` (here): handwritten records from JSON, for offline tests.
- `CompositeSearchTool` (composite.py): PubMed + Europe PMC (+ SportRxiv),
  Crossref retraction checks and open-access full text. Build either with
  `backend.tools.factory.build_search_tool()` (SEARCH_BACKEND=fixture|live).

Contract: `search` never raises for expected failures (no results, rate
limit, paywall, timeout). It returns a `SearchResponse` with a `state`.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Protocol

from backend.schemas import SearchResponse, SearchState, StudyRecord, utcnow

DEFAULT_FIXTURES = Path(__file__).parent / "fixtures" / "studies.json"

_STOPWORDS = frozenset(
    "a an and or the of in on for to with by vs versus is are does do how what which per".split()
)


class SearchTool(Protocol):
    name: str

    def search(self, query: str, max_results: int = 10) -> SearchResponse: ...


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 1}


class FixtureSearchTool:
    """Keyword search over a fixed list of StudyRecords.

    A record matches when at least `min_fraction` of the query's keywords (half
    by default; min 2) occur in its title, abstract or full text. `failures`
    maps a query substring to a failure state, to exercise the Researcher's
    failure paths. `from_file(only=...)` scopes the search to some records,
    e.g. one topic's studies for a live test whose planner writes its own queries.
    """

    name = "fixture"

    def __init__(
        self,
        records: list[StudyRecord],
        failures: dict[str, SearchState] | None = None,
        min_fraction: float = 0.5,
    ):
        self.records = records
        self.failures = failures or {}
        self.min_fraction = min_fraction
        self.queries: list[str] = []
        self._index = [(r, _tokens(f"{r.title} {r.abstract} {r.full_text or ''}")) for r in records]

    @classmethod
    def from_file(cls, path: Path = DEFAULT_FIXTURES, only: set[str] | None = None, **kw) -> "FixtureSearchTool":
        data = json.loads(Path(path).read_text())
        records = [StudyRecord.model_validate(r) for r in data["records"]]
        if only is not None:
            records = [r for r in records if r.source_id in only]
        return cls(records, **kw)

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        self.queries.append(query)
        for pattern, state in self.failures.items():
            if pattern.lower() in query.lower():
                return SearchResponse(query=query, state=state, message=f"injected {state}")
        q = _tokens(query)
        if not q:
            return SearchResponse(query=query, state="no_results", message="empty query")
        needed = max(2, math.ceil(len(q) * self.min_fraction)) if len(q) > 1 else 1
        scored = [(len(q & toks), r) for r, toks in self._index]
        hits = [r for score, r in sorted(scored, key=lambda x: -x[0]) if score >= needed]
        if not hits:
            return SearchResponse(query=query, state="no_results")
        now = utcnow()  # stamp provenance the way a live source would
        records = [
            r.model_copy(update={"retrieved_at": r.retrieved_at or now, "endpoint": r.endpoint or "fixture:studies.json"})
            for r in hits[:max_results]
        ]
        return SearchResponse(query=query, state="ok", records=records)

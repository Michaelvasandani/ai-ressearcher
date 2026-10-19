"""Golden set: questions with reference answers from position stands and meta-analyses.

`golden/golden.json` holds the items; this module validates and selects them.
Each item says whether it is in scope (out-of-scope items expect a referral),
which evidence status a well-calibrated answer reports, the reference
recommendation and key numbers (for the agreement judge), and the key studies
(for coverage), each with how sure we are of its identifiers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.schemas import EvidenceStatus, ScopeCategory, UserProfile

GOLDEN_PATH = Path(__file__).parent / "golden" / "golden.json"

IdConfidence = Literal["verified", "high", "medium", "unverified"]
# Identifiers we trust enough to score coverage against.
COVERAGE_CONFIDENCE: frozenset[str] = frozenset({"verified", "high", "medium"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KeyNumber(_Strict):
    value: str  # "1.6", "10-20", "-0.09"
    unit: str
    context: str = ""


class Reference(_Strict):
    recommendation: str
    key_points: list[str] = Field(default_factory=list)
    key_numbers: list[KeyNumber] = Field(default_factory=list)


class KeyStudy(_Strict):
    citation: str
    pmid: str | None = None
    doi: str | None = None
    id_confidence: IdConfidence

    def identifiers(self) -> set[str]:
        out: set[str] = set()
        if self.pmid:
            out.add(self.pmid.strip())
        if self.doi:
            out.add(self.doi.strip().lower())
        return out


class GoldenItem(_Strict):
    id: str
    question: str
    profile: UserProfile | None = None
    topic: str
    tags: list[str] = Field(default_factory=list)
    in_scope: bool
    expected_scope_category: ScopeCategory
    acceptable_scope_categories: list[ScopeCategory] = Field(default_factory=list)
    expected_status: EvidenceStatus | None
    acceptable_statuses: list[EvidenceStatus] = Field(default_factory=list)
    reference: Reference
    key_studies: list[KeyStudy] = Field(default_factory=list)
    source: str
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "GoldenItem":
        if self.in_scope != (self.expected_scope_category == "in_scope"):
            raise ValueError(f"{self.id}: in_scope must match expected_scope_category")
        if self.in_scope and self.expected_status is None:
            raise ValueError(f"{self.id}: in-scope items need an expected_status")
        if self.expected_status and self.expected_status not in self.acceptable_statuses:
            self.acceptable_statuses = [self.expected_status, *self.acceptable_statuses]
        if self.expected_scope_category not in self.acceptable_scope_categories:
            self.acceptable_scope_categories = [self.expected_scope_category, *self.acceptable_scope_categories]
        return self

    def scored_studies(self) -> list[KeyStudy]:
        """Key studies whose identifiers count for coverage."""
        return [s for s in self.key_studies if s.id_confidence in COVERAGE_CONFIDENCE and s.identifiers()]


class GoldenSet(_Strict):
    version: int
    description: str = ""
    items: list[GoldenItem]

    @model_validator(mode="after")
    def _unique(self) -> "GoldenSet":
        ids = [i.id for i in self.items]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate golden ids: {sorted(dupes)}")
        return self


def load_golden(path: Path | str = GOLDEN_PATH) -> list[GoldenItem]:
    return GoldenSet.model_validate(json.loads(Path(path).read_text())).items


def select(
    items: list[GoldenItem],
    *,
    ids: list[str] | None = None,
    tags: list[str] | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> list[GoldenItem]:
    """Filter by ids and/or tags (an item matches any tag), then take a seeded sample."""
    out = list(items)
    if ids:
        known = {i.id for i in items}
        missing = [i for i in ids if i not in known]
        if missing:
            raise KeyError(f"unknown golden ids: {missing}")
        out = [i for i in out if i.id in set(ids)]
    if tags:
        out = [i for i in out if set(tags) & set(i.tags)]
    if sample is not None and sample < len(out):
        out = sorted(random.Random(seed).sample(out, sample), key=lambda i: items.index(i))
    return out

"""Merge, deduplicate, rank and diversify search records from several sources.

- Deduplicate by DOI and PMID across sources, merging what each source knows
  (PubMed's publication types and MeSH, Europe PMC's PMCID and open-access flag).
- Rank by fused relevance (reciprocal rank across sources), recency and design:
  the last 5-10 years are preferred, and older meta-analyses / systematic
  reviews stay in as standard references.
- Cap results per journal and per author group (same last author) so one lab
  or journal cannot dominate the evidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from backend.schemas import DroppedRecord, StudyRecord

SOURCE_PRIORITY = {"pubmed": 0, "europepmc": 1, "sportrxiv": 2, "crossref": 3, "fixture": 4}
SYNTHESIS_TYPES = ("meta-analysis", "systematic review")


@dataclass(frozen=True)
class RecencyPolicy:
    """Prefer the last `preferred_years`; accept up to `max_years`; keep older syntheses."""

    preferred_years: int = 5
    max_years: int = 10
    keep_older_syntheses: bool = True

    def min_year(self, today: date) -> int:
        return today.year - self.max_years


@dataclass(frozen=True)
class DiversityCaps:
    per_journal: int = 3
    per_author_group: int = 2


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d)
    return d or None


def is_synthesis(record: StudyRecord) -> bool:
    types = " ".join(t.lower() for t in record.publication_types)
    return any(t in types for t in SYNTHESIS_TYPES) or bool(
        re.search(r"\b(meta-analys[ie]s|systematic review)\b", record.title, re.I)
    )


def record_keys(r: StudyRecord) -> list[str]:
    keys = []
    if (doi := normalize_doi(r.doi)):
        keys.append(f"doi:{doi}")
    if r.pmid:
        keys.append(f"pmid:{r.pmid}")
    if r.pmcid:
        keys.append(f"pmcid:{r.pmcid.upper()}")
    if not keys:
        keys.append(f"id:{r.source_id}")
    return keys


def _origin(r: StudyRecord) -> str:
    return r.endpoint or r.source


def _merge(primary: StudyRecord, other: StudyRecord) -> StudyRecord:
    """Fill gaps in `primary` from `other`; retraction from either source wins."""
    update: dict = {}
    for f in ("pmid", "doi", "pmcid", "journal", "full_text", "full_text_url", "retraction_notice"):
        if getattr(primary, f) is None and getattr(other, f) is not None:
            update[f] = getattr(other, f)
    if len(other.abstract) > len(primary.abstract):
        update["abstract"] = other.abstract
    for f in ("publication_types", "mesh_terms", "authors"):
        if not getattr(primary, f) and getattr(other, f):
            update[f] = getattr(other, f)
    if other.retracted:
        update["retracted"] = True
    if primary.access in ("unknown", "unavailable") and other.access != "unknown":
        update["access"] = other.access
    update["is_preprint"] = primary.is_preprint and other.is_preprint
    seen = [*primary.also_found_in, _origin(other), *other.also_found_in]
    update["also_found_in"] = [o for o in dict.fromkeys(seen) if o != _origin(primary)]
    update["retraction_checked_by"] = list(dict.fromkeys([*primary.retraction_checked_by, *other.retraction_checked_by]))
    return primary.model_copy(update=update)


def dedupe(records: list[StudyRecord]) -> tuple[list[StudyRecord], list[DroppedRecord]]:
    """Merge records that share a DOI, PMID or PMCID. Keeps first-seen order; the
    higher-priority source (PubMed) supplies the base record and its source id."""
    groups: list[list[StudyRecord]] = []
    index: dict[str, int] = {}
    for r in records:
        hit = next((index[k] for k in record_keys(r) if k in index), None)
        if hit is None:
            hit = len(groups)
            groups.append([])
        groups[hit].append(r)
        for k in record_keys(r):
            index.setdefault(k, hit)
    out: list[StudyRecord] = []
    dropped: list[DroppedRecord] = []
    for group in groups:
        ordered = sorted(group, key=lambda r: SOURCE_PRIORITY.get(r.source, 9))
        merged = ordered[0]
        for other in ordered[1:]:
            merged = _merge(merged, other)
            dropped.append(
                DroppedRecord(
                    source_id=other.source_id, reason="duplicate", detail=f"{_origin(other)} copy merged into {merged.source_id}"
                )
            )
        out.append(merged)
    return out, dropped


def score(record: StudyRecord, relevance: float, today: date, policy: RecencyPolicy) -> float:
    """Higher is better. `relevance` is the fused reciprocal rank across sources (0-1)."""
    age = max(today.year - record.published.year, 0)
    s = relevance * 10
    if age <= policy.preferred_years:
        s += 3
    elif age <= policy.max_years:
        s += 1.5
    elif not (policy.keep_older_syntheses and is_synthesis(record)):
        s -= 4  # old primary study: only used when nothing newer is available
    types = {t.lower() for t in record.publication_types}
    if is_synthesis(record):
        s += 2
    elif "randomized controlled trial" in types:
        s += 1
    if record.is_preprint:
        s -= 1.5
    if record.full_text:
        s += 0.5  # methods and results available to check the claim
    return s


def rank(
    by_source: dict[str, list[StudyRecord]], merged: list[StudyRecord], today: date, policy: RecencyPolicy
) -> list[StudyRecord]:
    """Order merged records by reciprocal-rank fusion across sources plus recency and design."""
    rrf: dict[str, float] = {}
    for records in by_source.values():
        for pos, r in enumerate(records):
            for k in record_keys(r):
                rrf[k] = rrf.get(k, 0.0) + 1.0 / (pos + 2)
    def relevance(r: StudyRecord) -> float:
        return max((rrf.get(k, 0.0) for k in record_keys(r)), default=0.0)

    return sorted(merged, key=lambda r: score(r, relevance(r), today, policy), reverse=True)


def author_group(record: StudyRecord) -> str | None:
    """The lab, approximated by the last (senior) author's normalized surname + first initial."""
    last = record.last_author
    if not last:
        return None
    ascii_ = unicodedata.normalize("NFKD", last).encode("ascii", "ignore").decode().lower()
    parts = ascii_.replace(",", " ").split()
    if not parts:
        return None
    initial = parts[1][0] if len(parts) > 1 else ""
    return f"{parts[0]} {initial}".strip()


def apply_caps(records: list[StudyRecord], caps: DiversityCaps) -> tuple[list[StudyRecord], list[DroppedRecord]]:
    """Walk records best-first; drop any that would exceed a journal or author-group cap."""
    journals: dict[str, int] = {}
    groups: dict[str, int] = {}
    kept: list[StudyRecord] = []
    dropped: list[DroppedRecord] = []
    for r in records:
        journal = (r.journal or "").strip().lower() or None
        group = author_group(r)
        if journal and journals.get(journal, 0) >= caps.per_journal:
            dropped.append(DroppedRecord(source_id=r.source_id, reason="journal_cap", detail=r.journal))
            continue
        if group and groups.get(group, 0) >= caps.per_author_group:
            dropped.append(DroppedRecord(source_id=r.source_id, reason="author_cap", detail=f"last author {r.last_author}"))
            continue
        if journal:
            journals[journal] = journals.get(journal, 0) + 1
        if group:
            groups[group] = groups.get(group, 0) + 1
        kept.append(r)
    return kept, dropped

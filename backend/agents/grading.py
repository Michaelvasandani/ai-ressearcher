"""Deterministic, GRADE-inspired certainty grade for a single finding.

Grading is done in code, not by the model, so it is reproducible and every
downgrade has a recorded reason. Phase 4's Position Mapper grades bodies of
evidence; this grades one study.
"""

from __future__ import annotations

from backend.schemas import EvidenceGrade, StudyType

_ORDER: list[EvidenceGrade] = ["very_low", "low", "moderate", "high"]

BASELINE: dict[StudyType, EvidenceGrade] = {
    "meta_analysis": "high",
    "rct": "moderate",  # a single trial, not a body of trials
    "cohort": "low",
    "case_control": "low",
    "review": "low",  # narrative review: no pooled estimate
    "preprint": "low",  # not peer reviewed
    "case_report": "very_low",
    "animal": "very_low",
    "in_vitro": "very_low",
}

SMALL_TRIAL_N = 20


def grade_finding(
    study_type: StudyType, sample_size: int | None, funding_conflict: bool | None
) -> tuple[EvidenceGrade, list[str]]:
    grade = BASELINE[study_type]
    reasons = [f"baseline {grade} for {study_type.replace('_', ' ')}"]
    level = _ORDER.index(grade)

    if study_type == "rct" and sample_size is not None and sample_size < SMALL_TRIAL_N:
        level -= 1
        reasons.append(f"downgraded: small trial (n={sample_size} < {SMALL_TRIAL_N})")
    if funding_conflict:
        level -= 1
        reasons.append("downgraded: funding conflict of interest")

    return _ORDER[max(level, 0)], reasons

"""Deterministic rules for synthesis: body-of-evidence grades, the false-balance
rule, the confidence cap, and how findings are shown to the synthesis agents.

The Position Mapper, Advocates and Judge are models; everything here is code,
so the parts of a verdict that must not depend on rhetoric (how strong a
position's evidence is, whether a disagreement is real, how confident the
answer may be) are reproducible and every adjustment has a recorded reason.

Down-weighting: a finding whose study population does not match the user (see
graph.supervisor.downweight_reasons) counts as indirect evidence. GRADE
downgrades indirect evidence one level, and its weight in support counts is
halved. The finding is never dropped: it still appears, labelled, to the
mapper, the advocates and the judge.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from backend.schemas import Confidence, EvidenceGrade, EvidenceStatus, Finding, Position

GRADE_ORDER: list[EvidenceGrade] = ["very_low", "low", "moderate", "high"]
STATUS_ORDER: list[EvidenceStatus] = ["consensus", "leaning", "genuinely_contested"]
CONFIDENCE_ORDER: list[Confidence] = ["low", "moderate", "high"]

CANNOT_CARRY_ALONE = frozenset({"animal", "in_vitro", "preprint"})
DOWNWEIGHT = 0.5  # weight of a finding from a population unlike the user's
CONSISTENT_STUDIES = 3  # independent moderate-or-better human studies that make a body "high"
CONTESTED_SUPPORT_RATIO = 0.5  # an equal-grade rival with at least this share of the support is a real split


def grade_rank(g: EvidenceGrade) -> int:
    return GRADE_ORDER.index(g)


def lower(g: EvidenceGrade, levels: int = 1) -> EvidenceGrade:
    return GRADE_ORDER[max(grade_rank(g) - levels, 0)]


def least_contested(*statuses: EvidenceStatus) -> EvidenceStatus:
    return min(statuses, key=STATUS_ORDER.index)


def can_carry(f: Finding) -> bool:
    """Preprints, animal and in-vitro work may support a point but never carry a recommendation alone."""
    return f.study_type not in CANNOT_CARRY_ALONE


def effective_grade(f: Finding, downweighted: Mapping[str, str]) -> EvidenceGrade:
    """The finding's grade, one level lower if its population does not match the user (indirectness)."""
    return lower(f.grade) if f.id in downweighted else f.grade


def _one_per_source(findings: Iterable[Finding], downweighted: Mapping[str, str]) -> list[Finding]:
    """The same paper found under two sub-questions counts once (its best effective grade)."""
    best: dict[str, Finding] = {}
    for f in findings:
        cur = best.get(f.source_id)
        if cur is None or grade_rank(effective_grade(f, downweighted)) > grade_rank(effective_grade(cur, downweighted)):
            best[f.source_id] = f
    return list(best.values())


def support_weight(findings: Iterable[Finding], downweighted: Mapping[str, str]) -> float:
    """How much evidence backs a position: grade-weighted, one vote per paper, down-weighted findings at half."""
    return sum(
        (grade_rank(f.grade) + 1) * (DOWNWEIGHT if f.id in downweighted else 1.0)
        for f in _one_per_source(findings, downweighted)
    )


def grade_body(findings: list[Finding], downweighted: Mapping[str, str]) -> tuple[EvidenceGrade, list[str]]:
    """GRADE-style certainty of the body of evidence behind one position.

    - Start from the best effective grade among findings that can carry a
      recommendation (peer-reviewed human studies).
    - If none can, the body is at most `low`: preprint / animal / in-vitro only.
    - Several consistent moderate-or-better human studies (independent papers)
      raise a `moderate` body to `high`: one trial is moderate, a body of
      consistent trials is not.
    """
    if not findings:
        return "very_low", ["no supporting findings"]
    papers = _one_per_source(findings, downweighted)
    carrying = [f for f in papers if can_carry(f)]
    reasons: list[str] = []
    indirect = sorted(f.id for f in papers if f.id in downweighted)
    if indirect:
        reasons.append(f"indirect (population differs from the user), one level lower: {', '.join(indirect)}")
    if not carrying:
        best = max((effective_grade(f, downweighted) for f in papers), key=grade_rank)
        grade = best if grade_rank(best) <= grade_rank("low") else "low"
        reasons.append(f"{grade}: only preprint, animal or in-vitro evidence")
        return grade, reasons
    top = max(carrying, key=lambda f: grade_rank(effective_grade(f, downweighted)))
    grade = effective_grade(top, downweighted)
    reasons.append(f"best supporting evidence {grade}: {top.id} ({top.study_type.replace('_', ' ')})")
    solid = [f for f in carrying if grade_rank(effective_grade(f, downweighted)) >= grade_rank("moderate")]
    if grade == "moderate" and len(solid) >= CONSISTENT_STUDIES:
        grade = "high"
        reasons.append(f"raised to high: {len(solid)} consistent moderate-grade studies")
    return grade, reasons


def position_findings(position: Position, by_id: Mapping[str, Finding]) -> list[Finding]:
    return [by_id[i] for i in position.supporting_findings if i in by_id]


@dataclass(frozen=True)
class StatusDecision:
    status: EvidenceStatus
    positions: list[Position]  # leading position first; only those worth debating
    set_aside: list[Position]  # weak dissent: not a credible side, never debated
    reasons: list[str]


def decide_status(
    positions: list[Position],
    by_id: Mapping[str, Finding],
    downweighted: Mapping[str, str],
    proposed: EvidenceStatus,
) -> StatusDecision:
    """Avoid false balance, in code.

    Positions are ranked by evidence grade, then grade-weighted support. A
    dissenting position is a credible side only if its evidence is at least
    moderate or as strong as the leading position's. Low or very-low dissent
    against a stronger lead is set aside: the answer is `consensus`, not a
    50/50 debate. With credible dissent the status is `genuinely_contested`
    when a rival matches the lead's grade with comparable support, otherwise
    `leaning`. The mapper's proposed status can move the answer towards
    consensus, never away from it.
    """
    weights = {id(p): support_weight(position_findings(p, by_id), downweighted) for p in positions}
    ordered = sorted(positions, key=lambda p: (grade_rank(p.evidence_grade), weights[id(p)]), reverse=True)
    lead, rest = ordered[0], ordered[1:]
    reasons = [f"leading position: {lead.stance!r} ({lead.evidence_grade}, support {weights[id(lead)]:.1f})"]
    if not rest:
        reasons.append("one position: consensus")
        return StatusDecision("consensus", [lead], [], reasons)

    credible: list[Position] = []
    weak: list[Position] = []
    for p in rest:
        if grade_rank(p.evidence_grade) >= grade_rank("moderate") or grade_rank(p.evidence_grade) >= grade_rank(lead.evidence_grade):
            credible.append(p)
        else:
            weak.append(p)
            reasons.append(
                f"set aside {p.stance!r}: only {p.evidence_grade} evidence against a {lead.evidence_grade} lead "
                "(not a credible side; avoids false balance)"
            )
    if not credible:
        code_status: EvidenceStatus = "consensus"
    elif any(
        p.evidence_grade == lead.evidence_grade and weights[id(p)] >= CONTESTED_SUPPORT_RATIO * weights[id(lead)]
        for p in credible
    ):
        code_status = "genuinely_contested"
        reasons.append("a rival position has evidence of the same grade and comparable support: contested")
    else:
        code_status = "leaning"
        reasons.append("credible dissent, but weaker than the lead: leaning")

    status = least_contested(code_status, proposed)
    if status != code_status:
        reasons.append(f"mapper proposed {proposed}; kept the less contested status")
    if status == "consensus":
        return StatusDecision("consensus", [lead], credible + weak, reasons)
    return StatusDecision(status, [lead, *credible], weak, reasons)


def confidence_cap(
    findings: list[Finding], downweighted: Mapping[str, str], status: EvidenceStatus
) -> tuple[Confidence, str]:
    """Highest confidence the recommended position's evidence allows.

    The best effective grade among the findings that can carry a
    recommendation sets the cap (high -> high, moderate -> moderate, low or
    very low -> low). A genuinely contested answer is at most moderate.
    """
    carrying = [f for f in findings if can_carry(f)]
    if not carrying:
        return "low", "no peer-reviewed human finding supports the recommendation"
    best = max((effective_grade(f, downweighted) for f in carrying), key=grade_rank)
    cap: Confidence = "high" if best == "high" else "moderate" if best == "moderate" else "low"
    reason = f"best supporting grade is {best}"
    if status == "genuinely_contested" and cap == "high":
        cap, reason = "moderate", reason + "; contested evidence caps confidence at moderate"
    return cap, reason


def min_confidence(a: Confidence, b: Confidence) -> Confidence:
    return min(a, b, key=CONFIDENCE_ORDER.index)


# --- Prompt rendering ---------------------------------------------------------


def render_finding(f: Finding, downweighted: Mapping[str, str]) -> str:
    """One finding as the synthesis agents see it. Starts like the writer's format: `[id] (type, grade g`."""
    n = f.sample_size if f.sample_size is not None else "?"
    lines = [
        f"[{f.id}] ({f.study_type}, grade {f.grade}, n={n}, {f.published.year}) {f.claim}",
        f"  population: {f.population}",
        f"  effect: {f.effect or 'not reported'}",
        f'  snippet: "{f.snippet}"',
    ]
    if f.study_type in CANNOT_CARRY_ALONE:
        lines.append(f"  note: {f.study_type.replace('_', ' ')}; cannot carry a recommendation alone")
    if f.id in downweighted:
        lines.append(f"  down-weighted: {downweighted[f.id]}")
    return "\n".join(lines)


def render_findings(findings: list[Finding], downweighted: Mapping[str, str]) -> str:
    return "\n".join(render_finding(f, downweighted) for f in findings)


def render_positions(positions: list[Position]) -> str:
    return "\n".join(
        f"P{i}: {p.stance}\n  evidence grade: {p.evidence_grade}\n  supporting findings: {', '.join(p.supporting_findings)}"
        f"\n  best counterargument: {p.best_counterargument}"
        for i, p in enumerate(positions, 1)
    )

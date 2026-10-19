"""Eval metrics: per item, then aggregated.

Per item (`ItemMetrics`):
- scope: out-of-scope items must be referred (with an acceptable category); in-scope items must not be.
- agreement: the judge's score of the recommendation against the golden reference (no answer = 0
  in the strict mean; `agreement_answered` averages answered items only).
- calibration: expected vs actual evidence status (exact, or within the item's acceptable statuses).
- faithfulness: share of (claim, cited finding) pairs where the finding supports the claim
  (supports 1, partially 0.5), and the share of claims supported by at least one cited finding.
- coverage: golden key studies (trusted identifiers only) whose PMID / DOI is among the run's findings.
- cost and latency: what the run paid (state spend) and its wall-clock time; plus judge cost.
- citation gate: whether the writer's first draft passed the gate, the issue kinds it had, and
  whether a repair or the escalated rewrite was needed.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.graph.builder import RunResult
from backend.schemas import Finding
from evals.golden import GoldenItem
from evals.harness import RunCase
from evals.judges import Judge

STATUSES = ("consensus", "leaning", "genuinely_contested")
NO_ANSWER = "no_answer"


class GateStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drafts: int = 0  # citation reports (one per validated draft)
    first_draft_ok: bool | None = None  # None: no draft reached the gate
    first_draft_issues: dict[str, int] = Field(default_factory=dict)
    repaired: bool = False  # a citation-only repair ran
    repair_ok: bool | None = None
    rewrite_used: bool = False  # the escalated (Opus) rewrite ran
    final_ok: bool | None = None
    failed: bool = False  # the run failed with citations_failed


class ItemMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    topic: str
    in_scope_expected: bool
    outcome: str  # answered / degraded / referred / failed / error
    failure_reason: str | None = None
    message: str = ""
    answered: bool = False
    referred: bool = False
    scope_category: str | None = None
    scope_correct: bool
    expected_status: str | None = None
    actual_status: str | None = None
    status_exact: bool | None = None
    status_acceptable: bool | None = None
    confidence: str | None = None
    recommendation: str | None = None
    agreement: float | None = None
    agreement_rating: str | None = None
    agreement_detail: dict[str, Any] = Field(default_factory=dict)
    faithfulness: float | None = None
    claim_support: float | None = None
    faithfulness_pairs: int = 0
    unsupported_pairs: list[dict[str, Any]] = Field(default_factory=list)
    coverage_found: list[str] = Field(default_factory=list)
    coverage_total: int = 0
    findings: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    judge_cost_usd: float = 0.0
    gate: GateStats = Field(default_factory=GateStats)


# --- Per item ---------------------------------------------------------------------


def _norm_ids(f: Finding) -> set[str]:
    ids = {f.source_id.strip(), f.source_id.strip().lower()}
    url = f.source_url.lower()
    for marker in ("doi.org/", "pubmed.ncbi.nlm.nih.gov/"):
        if marker in url:
            ids.add(url.split(marker, 1)[1].strip("/"))
    if f.provenance is not None:
        ids.add(f.provenance.source_id.strip().lower())
        ids.update(x.strip().lower() for x in f.provenance.also_found_in)
    return {i.removeprefix("doi:").removeprefix("pmid:") for i in ids}


def coverage(item: GoldenItem, findings: list[Finding]) -> tuple[list[str], int]:
    """(key studies found, key studies scored). A study counts once if any of its ids matches."""
    have: set[str] = set()
    for f in findings:
        have |= _norm_ids(f)
    studies = item.scored_studies()
    found = [s.pmid or s.doi or "" for s in studies if s.identifiers() & have]
    return found, len(studies)


def gate_stats(result: RunResult) -> GateStats:
    reports = result.citation_reports
    if not reports:
        return GateStats(failed=bool(result.outcome and result.outcome.failure and result.outcome.failure.reason == "citations_failed"))
    first = reports[0]
    repaired = [r for r in reports if getattr(r, "stage", "draft") == "repaired"]
    kinds = Counter(i.kind for i in first.issues)
    failure = result.outcome.failure if result.outcome else None
    return GateStats(
        drafts=len(reports),
        first_draft_ok=first.ok,
        first_draft_issues=dict(kinds),
        repaired=bool(repaired),
        repair_ok=repaired[0].ok if repaired else None,
        rewrite_used=any(r.attempt > 1 for r in reports),
        final_ok=reports[-1].ok,
        failed=bool(failure and failure.reason == "citations_failed"),
    )


def item_metrics(item: GoldenItem, case: RunCase, judge: Judge | None) -> ItemMetrics:
    if case.result is None or case.result.outcome is None:
        return ItemMetrics(
            item_id=item.id, topic=item.topic, in_scope_expected=item.in_scope, outcome="error",
            message=case.error or "no result", scope_correct=False, latency_s=case.wall_s,
            expected_status=item.expected_status, coverage_total=len(item.scored_studies()),
        )
    r = case.result
    outcome = r.outcome
    assert outcome is not None
    referred = outcome.status == "referred"
    category = r.intake.scope_category if r.intake else None
    if item.in_scope:
        scope_correct = not referred
    else:
        scope_correct = referred and category in item.acceptable_scope_categories
    answer = r.answer
    found, total = coverage(item, r.findings)
    m = ItemMetrics(
        item_id=item.id,
        topic=item.topic,
        in_scope_expected=item.in_scope,
        outcome=outcome.status,
        failure_reason=outcome.failure.reason if outcome.failure else None,
        message=outcome.message[:300],
        answered=answer is not None,
        referred=referred,
        scope_category=category,
        scope_correct=scope_correct,
        expected_status=item.expected_status,
        actual_status=answer.status if answer else None,
        confidence=answer.confidence if answer else None,
        recommendation=answer.recommendation if answer else None,
        coverage_found=found,
        coverage_total=total,
        findings=len(r.findings),
        cost_usd=r.spend.cost_usd,
        latency_s=case.wall_s,
        llm_calls=r.spend.llm_calls,
        tokens=r.spend.total_tokens,
        gate=gate_stats(r),
    )
    if answer is not None and item.expected_status is not None:
        m.status_exact = answer.status == item.expected_status
        m.status_acceptable = answer.status in item.acceptable_statuses
    if answer is not None and item.in_scope and judge is not None:
        agr = judge.agreement(item, answer)
        faith = judge.faithfulness(answer, r.findings)
        m.agreement, m.agreement_rating, m.agreement_detail = agr.score, agr.rating, agr.detail
        m.faithfulness, m.claim_support = faith.score, faith.claim_support
        m.faithfulness_pairs = len(faith.pairs)
        m.unsupported_pairs = [p for p in faith.pairs if p["label"] != "supports"][:8]
        m.judge_cost_usd = agr.cost_usd + faith.cost_usd
    return m


# --- Aggregates ---------------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 4) if xs else None


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return round(s[k], 4)


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def confusion_matrix(rows: list[ItemMetrics]) -> dict[str, dict[str, int]]:
    """expected status -> actual status (or no_answer) -> count, over in-scope items."""
    matrix = {e: {a: 0 for a in (*STATUSES, NO_ANSWER)} for e in STATUSES}
    for m in rows:
        if m.in_scope_expected and m.expected_status in matrix:
            matrix[m.expected_status][m.actual_status or NO_ANSWER] += 1
    return matrix


def aggregate(rows: list[ItemMetrics]) -> dict[str, Any]:
    ins = [m for m in rows if m.in_scope_expected]
    oos = [m for m in rows if not m.in_scope_expected]
    answered = [m for m in ins if m.answered]
    judged = [m for m in answered if m.agreement is not None]
    pairs = sum(m.faithfulness_pairs for m in answered)
    faith_weighted = (
        round(sum((m.faithfulness or 0.0) * m.faithfulness_pairs for m in answered) / pairs, 4) if pairs else None
    )
    cov_total = sum(m.coverage_total for m in ins)
    cov_found = sum(len(m.coverage_found) for m in ins)
    with_status = [m for m in answered if m.status_exact is not None]
    matrix = confusion_matrix(rows)
    drafted = [m for m in rows if m.gate.first_draft_ok is not None]
    issue_kinds: Counter[str] = Counter()
    for m in drafted:
        issue_kinds.update(m.gate.first_draft_issues)
    costs = [m.cost_usd for m in rows if m.outcome != "error"]
    answered_costs = [m.cost_usd for m in answered]
    lat = [m.latency_s for m in rows if m.outcome != "error"]
    return {
        "items": len(rows),
        "in_scope": len(ins),
        "out_of_scope": len(oos),
        "errors": sum(m.outcome == "error" for m in rows),
        "answer_rate": _rate([m.answered for m in ins]),
        "outcomes": dict(Counter(m.outcome for m in rows)),
        "failure_reasons": dict(Counter(m.failure_reason for m in rows if m.failure_reason)),
        "agreement": {
            # Strict: an in-scope item without a judged answer scores 0.
            # None when answers exist but were not judged (no judge configured).
            "mean": None if (answered and not judged) else _mean([m.agreement or 0.0 for m in ins]),
            "answered_mean": _mean([m.agreement for m in judged if m.agreement is not None]),
            "ratings": dict(Counter(m.agreement_rating for m in judged)),
            "judged": len(judged),
        },
        "faithfulness": {
            "pair_mean": faith_weighted,
            "claim_support_mean": _mean([m.claim_support for m in answered if m.claim_support is not None]),
            "pairs": pairs,
        },
        "calibration": {
            "confusion_matrix": matrix,
            "exact_accuracy": _rate([bool(m.status_exact) for m in with_status]),
            "acceptable_accuracy": _rate([bool(m.status_acceptable) for m in with_status]),
            # A debate where the literature agrees, and consensus where it truly disagrees:
            "false_debate": matrix["consensus"]["genuinely_contested"],
            "missed_debate": matrix["genuinely_contested"]["consensus"],
            "n": len(with_status),
        },
        "coverage": {
            "micro": round(cov_found / cov_total, 4) if cov_total else None,
            "found": cov_found,
            "total": cov_total,
        },
        "scope": {
            "accuracy": _rate([m.scope_correct for m in rows if m.outcome != "error"]),
            "out_of_scope_referred": _rate([m.scope_correct for m in oos]),
            "in_scope_false_referrals": sum(m.referred for m in ins),
        },
        "cost_usd": {
            "mean": _mean(costs),
            "mean_in_scope": _mean([m.cost_usd for m in ins if m.outcome != "error"]),
            "mean_answered": _mean(answered_costs),
            "p50": _pct(costs, 0.5),
            "p90": _pct(costs, 0.9),
            "total": round(sum(costs), 4),
            "judge_total": round(sum(m.judge_cost_usd for m in rows), 4),
        },
        "latency_s": {"mean": _mean(lat), "p50": _pct(lat, 0.5), "p90": _pct(lat, 0.9)},
        "citation_gate": {
            "drafted": len(drafted),
            "first_draft_pass_rate": _rate([bool(m.gate.first_draft_ok) for m in drafted]),
            "repair_rate": _rate([m.gate.repaired for m in drafted]),
            "repair_pass_rate": _rate([bool(m.gate.repair_ok) for m in drafted if m.gate.repaired]),
            "rewrite_rate": _rate([m.gate.rewrite_used for m in drafted]),
            "final_pass_rate": _rate([bool(m.gate.final_ok) for m in drafted]),
            "citations_failed": sum(m.gate.failed for m in rows),
            "first_draft_issue_kinds": dict(issue_kinds),
        },
    }

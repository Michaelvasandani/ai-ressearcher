"""Judges for the answer-quality metrics.

Two implementations behind one interface:

- `LLMJudge`: agreement on Sonnet (the coach's recommendation vs the golden
  reference, scored with a fixed rubric) and faithfulness on Haiku (does each
  cited finding support the sentence that cites it). Structured output through
  the backend's `LLMClient`, so `FakeLLM` drives it in tests. Results are
  cached on disk by a hash of the request, so recomputing a report is free.
- `HeuristicJudge`: deterministic proxies with no model (reference numbers
  found in the answer; numbers and content words shared by a sentence and its
  finding). Used by the offline eval mode and the CI gate's own tests. It is
  a smoke signal, not a quality measure.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from backend import config
from backend.citations import (
    MARKER,
    extract_numbers,
    finding_values,
    marker_numbers,
    matches,
    sentences,
    strip_markers,
)
from backend.llm import CallRecord, LLMClient, LLMRequest
from backend.schemas import CoachAnswer, Finding
from evals.golden import GoldenItem

AGREEMENT_AGENT = "eval_agreement"
FAITHFULNESS_AGENT = "eval_faithfulness"
AGREEMENT_MODEL = config.SONNET
FAITHFULNESS_MODEL = config.HAIKU
MAX_PAIRS = 40  # claim-finding pairs judged per answer

AgreementRating = Literal["agrees", "mostly_agrees", "partially_agrees", "disagrees"]
AGREEMENT_SCORES: dict[str, float] = {"agrees": 1.0, "mostly_agrees": 0.75, "partially_agrees": 0.25, "disagrees": 0.0}
SupportLabel = Literal["supports", "partially_supports", "does_not_support"]
SUPPORT_SCORES: dict[str, float] = {"supports": 1.0, "partially_supports": 0.5, "does_not_support": 0.0}


# --- Output schemas ---------------------------------------------------------------


class AgreementJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: AgreementRating
    numbers_consistent: Literal["yes", "no", "not_applicable"]
    contradictions: list[str]
    missing_key_points: list[str]
    rationale: str

    @property
    def score(self) -> float:
        return AGREEMENT_SCORES[self.rating]


class PairJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: int
    label: SupportLabel
    reason: str


class FaithfulnessJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgments: list[PairJudgment]


# --- Claim-finding pairs ------------------------------------------------------------


class ClaimPair(BaseModel):
    pair_id: int
    field: str
    claim: str  # the sentence, markers stripped
    finding_id: str


def claim_pairs(answer: CoachAnswer, findings: list[Finding], limit: int = MAX_PAIRS) -> tuple[list[ClaimPair], int]:
    """(sentence, cited finding) pairs from the answer's text fields and position stances.

    Returns the pairs (at most `limit`) and how many there were in all."""
    by_id = {f.id: f for f in findings}
    refs = answer.cited_finding_ids
    pairs: list[ClaimPair] = []
    fields = ["recommendation", "why", "whats_new", "practical_protocol", "applies_to_you", "what_would_change_this"]
    for name in fields:
        for s in sentences(getattr(answer, name) or ""):
            ids: list[str] = []
            for m in MARKER.finditer(s):
                for n in marker_numbers(m.group(1)):
                    if 1 <= n <= len(refs) and refs[n - 1] in by_id:
                        ids.append(refs[n - 1])
            claim = " ".join(strip_markers(s).split())
            for fid in dict.fromkeys(ids):
                pairs.append(ClaimPair(pair_id=len(pairs) + 1, field=name, claim=claim, finding_id=fid))
    for k, p in enumerate(answer.positions, start=1):
        for fid in [i for i in p.supporting_findings if i in by_id][:4]:
            pairs.append(ClaimPair(pair_id=len(pairs) + 1, field=f"positions[{k}].stance", claim=p.stance, finding_id=fid))
    return pairs[:limit], len(pairs)


# --- Interface ----------------------------------------------------------------------


@dataclass
class AgreementResult:
    score: float | None  # None: nothing to judge (no answer)
    rating: str | None
    detail: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0


@dataclass
class FaithfulnessResult:
    pairs: list[dict[str, Any]]  # pair + label + reason
    total_pairs: int  # before the MAX_PAIRS cap
    cost_usd: float = 0.0

    @property
    def score(self) -> float | None:
        if not self.pairs:
            return None
        return sum(SUPPORT_SCORES[p["label"]] for p in self.pairs) / len(self.pairs)

    @property
    def claim_support(self) -> float | None:
        """Share of claims supported by at least one of the findings they cite."""
        claims: dict[tuple[str, str], bool] = {}
        for p in self.pairs:
            key = (p["field"], p["claim"])
            claims[key] = claims.get(key, False) or p["label"] == "supports"
        return sum(claims.values()) / len(claims) if claims else None


class Judge(Protocol):
    name: str

    def agreement(self, item: GoldenItem, answer: CoachAnswer) -> AgreementResult: ...

    def faithfulness(self, answer: CoachAnswer, findings: list[Finding]) -> FaithfulnessResult: ...


# --- LLM judge ----------------------------------------------------------------------

AGREEMENT_SYSTEM = """\
You grade a coaching answer against a reference answer taken from position stands and \
meta-analyses. Judge substance, not wording or length.

Rubric:
- agrees: the recommendation points the same way as the reference, and any numbers it gives \
(doses, sets, grams per kg, percentages) are consistent with the reference's key numbers \
(the same value, an overlapping range, or a reasonable rounding).
- mostly_agrees: same direction, with a minor gap: a missing nuance, a somewhat wider or \
narrower range, or a missing key number.
- partially_agrees: some overlap, but a materially different number, an important missing \
caveat that changes what the user would do, or hedging where the reference is clear.
- disagrees: a different or opposite recommendation, or numbers that contradict the reference.

The answer may add details that the reference does not mention; do not penalize them unless \
they contradict it. numbers_consistent is not_applicable when neither side gives numbers. \
List contradictions and missing key points briefly (empty lists when there are none)."""

FAITHFULNESS_SYSTEM = """\
You check citations. Each pair has a claim from an answer and one study finding it cites \
(the finding's claim, effect, population and a verbatim snippet). For every pair decide \
whether the finding supports the claim:
- supports: the finding states or directly implies the claim, including every number in it.
- partially_supports: the finding supports part of the claim, or the claim generalizes it \
somewhat (e.g. to a broader population) without contradicting it.
- does_not_support: the finding is about something else, contradicts the claim, or does not \
contain a number the claim attributes to it.
Advice phrased as an instruction ("do 10 sets") is supported when the finding supports the \
evidence behind it. Return one judgment per pair_id, with a short reason."""


def _render_answer(answer: CoachAnswer) -> str:
    parts = [
        f"Status: {answer.status} (confidence {answer.confidence})",
        f"Recommendation: {strip_markers(answer.recommendation)}",
        f"Why: {strip_markers(answer.why)}",
        f"Practical protocol: {strip_markers(answer.practical_protocol)}",
    ]
    for k, p in enumerate(answer.positions, start=1):
        parts.append(f"Position {k} ({p.evidence_grade}): {p.stance}")
    return "\n".join(" ".join(x.split()) for x in parts)


def _render_reference(item: GoldenItem) -> str:
    ref = item.reference
    lines = [f"Reference recommendation: {ref.recommendation}"]
    if ref.key_points:
        lines.append("Key points:\n" + "\n".join(f"- {p}" for p in ref.key_points))
    if ref.key_numbers:
        lines.append(
            "Key numbers:\n" + "\n".join(f"- {n.value} {n.unit} ({n.context})" for n in ref.key_numbers)
        )
    lines.append(f"Source: {item.source}")
    if item.notes:
        lines.append(f"Grading notes: {item.notes}")
    return "\n".join(lines)


def _render_finding(f: Finding) -> str:
    return (
        f"claim: {f.claim}\n  effect: {f.effect or 'not reported'}\n  population: {f.population}\n"
        f"  study: {f.study_type}, n={f.sample_size if f.sample_size is not None else '?'}, {f.published.year}\n"
        f'  snippet: "{f.snippet}"'
    )


class JudgeCache:
    """JSON files keyed by a hash of (model, system, user). Thread-safe enough for one process."""

    def __init__(self, root: Path | None):
        self.root = root
        self._lock = threading.Lock()

    def key(self, request: LLMRequest[Any]) -> str:
        blob = json.dumps([request.agent, request.model, request.system, request.user], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def get(self, key: str) -> dict[str, Any] | None:
        if self.root is None:
            return None
        path = self.root / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.root is None:
            return
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / f"{key}.json").write_text(json.dumps(value))


class LLMJudge:
    name = "llm"

    def __init__(
        self,
        llm: LLMClient,
        cache_dir: Path | None = None,
        agreement_model: str = AGREEMENT_MODEL,
        faithfulness_model: str = FAITHFULNESS_MODEL,
    ):
        self.llm = llm
        self.cache = JudgeCache(cache_dir)
        self.agreement_model = agreement_model
        self.faithfulness_model = faithfulness_model

    def _complete(self, request: LLMRequest[Any]) -> tuple[Any, float]:
        """(output, cost). A cached judgment reports what it cost when it was made (paid once)."""
        key = self.cache.key(request)
        hit = self.cache.get(key)
        if hit is not None:
            return request.output_type.model_validate(hit["output"]), float(hit.get("cost_usd") or 0.0)
        result = self.llm.complete(request)
        record: CallRecord = result.record
        self.cache.put(key, {"output": result.output.model_dump(mode="json"), "cost_usd": record.cost_usd})
        return result.output, record.cost_usd

    def agreement(self, item: GoldenItem, answer: CoachAnswer) -> AgreementResult:
        profile = item.profile.model_dump_json(exclude_none=True) if item.profile else "{}"
        user = (
            f"Question: {item.question}\nUser profile: {profile}\n\n{_render_reference(item)}\n\n"
            f"Coach answer:\n{_render_answer(answer)}"
        )
        request = LLMRequest(
            agent=AGREEMENT_AGENT, system=AGREEMENT_SYSTEM, user=user, output_type=AgreementJudgment,
            model=self.agreement_model, max_tokens=2_000, effort="low",
        )
        out, cost = self._complete(request)
        return AgreementResult(score=out.score, rating=out.rating, detail=out.model_dump(), cost_usd=cost)

    def faithfulness(self, answer: CoachAnswer, findings: list[Finding]) -> FaithfulnessResult:
        pairs, total = claim_pairs(answer, findings)
        if not pairs:
            return FaithfulnessResult(pairs=[], total_pairs=total)
        by_id = {f.id: f for f in findings}
        blocks = [
            f"<pair id=\"{p.pair_id}\">\nclaim: {p.claim}\nfinding [{p.finding_id}]:\n  {_render_finding(by_id[p.finding_id])}\n</pair>"
            for p in pairs
        ]
        request = LLMRequest(
            agent=FAITHFULNESS_AGENT, system=FAITHFULNESS_SYSTEM, user="\n\n".join(blocks),
            output_type=FaithfulnessJudgment, model=self.faithfulness_model,
            max_tokens=min(16_000, 300 + 120 * len(pairs)), effort="low",
        )
        out, cost = self._complete(request)
        labels = {j.pair_id: j for j in out.judgments}
        rows = []
        for p in pairs:
            j = labels.get(p.pair_id)
            # A pair the judge skipped counts as unsupported: missing judgments must not inflate the score.
            rows.append(
                p.model_dump()
                | {"label": j.label if j else "does_not_support", "reason": j.reason if j else "no judgment returned"}
            )
        return FaithfulnessResult(pairs=rows, total_pairs=total, cost_usd=cost)


# --- Heuristic judge ------------------------------------------------------------------

_WORD = re.compile(r"[a-z][a-z-]{3,}")
_STOP = frozenset(
    "that this with from were more than have been into your their they them when what which also about "
    "should would could does each other most some such only very much many over under between".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _number_hits(spec: str, text_values: list[Any]) -> bool:
    nums = extract_numbers(spec, spelled="none")
    return bool(nums) and all(matches(n, text_values) for n in nums)


class HeuristicJudge:
    """Deterministic, model-free proxies. See the module docstring."""

    name = "heuristic"

    def agreement(self, item: GoldenItem, answer: CoachAnswer) -> AgreementResult:
        text = " ".join(
            strip_markers(t or "") for t in (answer.recommendation, answer.why, answer.practical_protocol)
        )
        values = [(n.value, n.places) for n in extract_numbers(text, spelled="all")]
        nums = item.reference.key_numbers
        if nums:
            hit = sum(_number_hits(n.value, values) for n in nums) / len(nums)
        else:
            ref = _content_words(item.reference.recommendation)
            hit = len(ref & _content_words(text)) / len(ref) if ref else 0.0
        rating: AgreementRating = (
            "agrees" if hit >= 0.75 else "mostly_agrees" if hit >= 0.5 else "partially_agrees" if hit > 0 else "disagrees"
        )
        return AgreementResult(score=AGREEMENT_SCORES[rating], rating=rating, detail={"overlap": round(hit, 3)})

    def faithfulness(self, answer: CoachAnswer, findings: list[Finding]) -> FaithfulnessResult:
        pairs, total = claim_pairs(answer, findings)
        by_id = {f.id: f for f in findings}
        rows = []
        for p in pairs:
            f = by_id[p.finding_id]
            ftext = " ".join([f.claim, f.effect or "", f.snippet, f.population, f.title or ""])
            numbers_ok = all(matches(n, finding_values(f)) for n in extract_numbers(p.claim))
            words = _content_words(p.claim)
            overlap = len(words & _content_words(ftext)) / len(words) if words else 0.0
            label: SupportLabel = (
                "supports" if numbers_ok and overlap >= 0.25 else "partially_supports" if overlap >= 0.1 else "does_not_support"
            )
            rows.append(p.model_dump() | {"label": label, "reason": f"numbers_ok={numbers_ok}, overlap={overlap:.2f}"})
        return FaithfulnessResult(pairs=rows, total_pairs=total)

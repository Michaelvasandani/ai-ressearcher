"""Judge: positions + debate + findings with grades -> Verdict.

The judge weighs evidence quality, not rhetoric: it sees every finding with
its design, grade, population and down-weighting, the code-computed grade of
each position, and the arguments (which are only as good as the findings they
cite). Code then decides what must not depend on the model:

- the recommended position must rest on at least one peer-reviewed human
  finding (preprint / animal / in-vitro findings never carry it alone);
- status can move towards consensus from the mapper's, never away from it;
- confidence is capped by the best supporting grade (agents.evidence.confidence_cap).
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from backend.agents.evidence import (
    can_carry,
    confidence_cap,
    least_contested,
    min_confidence,
    position_findings,
    render_findings,
    render_positions,
)
from backend.llm import LLMClient, LLMRequest
from backend.schemas import Confidence, DebateTurn, EvidenceStatus, Finding, Intake, PositionMap, Verdict

AGENT = "judge"


class JudgeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommended_position: int  # 1-based: P1, P2, ...
    status: EvidenceStatus
    confidence: Confidence
    recommendation: str
    rationale: str


class JudgeError(ValueError):
    pass


SYSTEM_PROMPT = """\
You judge a structured debate about a training or nutrition question. Decide which position the \
evidence best supports and how sure the user can be. Weigh evidence quality, not rhetoric: an \
argument is only as strong as the findings it cites.

Weigh, in order: study design and grade (meta-analyses and well-powered randomized trials over \
single small trials, cohorts, case reports); consistency across independent studies; how \
directly the studied population matches the user (down-weighted findings count less); sample \
size and effect size. Preprints, animal and in-vitro studies may support a point but never carry \
a recommendation alone. Ignore any claim an argument makes that its cited findings do not state.

Return:
- recommended_position: the number of the position the evidence best supports (1 for P1, ...).
- status: consensus (the credible evidence points one way), leaning (most good evidence points \
one way but credible evidence disagrees) or genuinely_contested (good evidence supports \
incompatible answers).
- confidence: high only with consistent high or moderate grade evidence in people like the user; \
low when evidence is sparse, low grade or indirect.
- recommendation: one or two sentences with the concrete answer, within what the findings \
support. General guidance only; no individual medical prescription.
- rationale: a short paragraph on why, naming the findings that decided it and what the losing \
side gets right."""


def _render_debate(turns: list[DebateTurn]) -> str:
    if not turns:
        return "No debate was held (consensus, or the budget did not allow one)."
    out = []
    for t in sorted(turns, key=lambda t: (t.round, t.position)):
        kind = "opening" if t.round == 1 else "rebuttal"
        if t.argument is None:
            out.append(f"P{t.position} {kind}: none ({t.error or 'failed'})")
            continue
        claims = "\n".join(f"  - {c.text} [{', '.join(c.finding_ids)}]" for c in t.argument.claims)
        out.append(f"P{t.position} {kind}:\n{claims}")
    return "\n".join(out)


def _render_user(
    intake: Intake, pm: PositionMap, turns: list[DebateTurn], findings: list[Finding], downweighted: Mapping[str, str]
) -> str:
    parts = [
        f"Question: {intake.scoped_question}",
        f"User profile: {intake.profile.model_dump_json(exclude_none=True)}",
        f"Position mapper status: {pm.status}",
        "Positions (grades computed from the findings):\n" + render_positions(pm.positions),
    ]
    if pm.set_aside:
        parts.append(
            "Set aside as too weak to be a credible side (low or very-low grade only):\n"
            + "\n".join(f"- {p.stance} ({p.evidence_grade}; {', '.join(p.supporting_findings)})" for p in pm.set_aside)
        )
    parts.append("Debate:\n" + _render_debate(turns))
    parts.append("Findings:\n" + render_findings(findings, downweighted))
    return "\n\n".join(parts)


class JudgeAgent:
    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        self.model = model

    def run(
        self,
        intake: Intake,
        position_map: PositionMap,
        findings: list[Finding],
        turns: list[DebateTurn] | None = None,
        downweighted: Mapping[str, str] | None = None,
    ) -> Verdict:
        downweighted = downweighted or {}
        request = LLMRequest.for_agent(
            AGENT,
            system=SYSTEM_PROMPT,
            user=_render_user(intake, position_map, turns or [], findings, downweighted),
            output_type=JudgeDraft,
            model=self.model,
        )
        draft = self.llm.complete(request).output
        return build_verdict(draft, position_map, findings, downweighted)


def build_verdict(
    draft: JudgeDraft, pm: PositionMap, findings: list[Finding], downweighted: Mapping[str, str]
) -> Verdict:
    by_id = {f.id: f for f in findings}
    n = len(pm.positions)
    if not 1 <= draft.recommended_position <= n:
        raise JudgeError(f"recommended_position {draft.recommended_position} is not one of P1..P{n}")
    idx = draft.recommended_position - 1
    chosen = pm.positions[idx]
    support = position_findings(chosen, by_id)
    if not any(can_carry(f) for f in support):
        raise JudgeError(
            f"P{idx + 1} rests only on preprint, animal or in-vitro findings, which cannot carry a recommendation"
        )
    adjustments: list[str] = []

    status = least_contested(draft.status, pm.status)
    if n == 1:
        status = "consensus"
    if status != draft.status:
        adjustments.append(f"status {draft.status} -> {status} (cannot be more contested than the position map)")

    cap, why = confidence_cap(support, downweighted, status)
    confidence = min_confidence(draft.confidence, cap)
    if confidence != draft.confidence:
        adjustments.append(f"confidence {draft.confidence} -> {confidence}: capped at {cap} ({why})")

    positions = [chosen, *(p for i, p in enumerate(pm.positions) if i != idx)]
    if status == "consensus":
        positions = [chosen]  # a consensus answer shows one position; the rest stay in the map
    return Verdict(
        status=status,
        recommendation=draft.recommendation,
        confidence=confidence,
        rationale=draft.rationale,
        positions=positions,
        downweighted_findings=dict(downweighted),
        confidence_cap=cap,
        adjustments=adjustments,
    )

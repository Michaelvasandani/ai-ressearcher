"""Coach Writer: findings (+ optional positions / verdict) + profile -> CoachAnswer.

In Phase 1 the writer can work from findings alone, grouping them into
positions itself. When Phase 4 supplies a PositionMap or a Verdict, those
decide status, confidence and positions, and the writer only explains them.

Code, not the model, guarantees: every cited id is a real finding id, the
disclaimer is present, and a contested answer shows at least two positions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from backend.llm import LLMClient, LLMRequest
from backend.schemas import (
    DISCLAIMER,
    CoachAnswer,
    Confidence,
    EvidenceGrade,
    EvidenceStatus,
    Finding,
    Intake,
    Position,
    PositionMap,
    Verdict,
)

AGENT = "writer"


class PositionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: str
    supporting_findings: list[str]
    evidence_grade: EvidenceGrade
    best_counterargument: str


class WriterDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: EvidenceStatus
    recommendation: str
    confidence: Confidence
    why: str
    positions: list[PositionDraft]
    what_would_change_this: str
    applies_to_you: str
    practical_protocol: str
    whats_new: str | None
    cited_finding_ids: list[str]


class WriterError(ValueError):
    pass


SYSTEM_PROMPT = """\
You write the final answer for an evidence-based training and nutrition coach. You get the user's \
question and profile, and a list of findings extracted from studies, each with an id, a grade and \
a verbatim snippet. Answer only from these findings.

Rules:
- Every factual claim and every number (sets, grams, percentages, effect sizes) must come from a \
finding. Cite the finding ids that support the answer in cited_finding_ids, and in each \
position's supporting_findings. Use the exact ids given; never invent one.
- Weigh evidence by grade: meta-analyses and well-powered trials over single small trials. \
Preprints and animal studies may support a point but never carry the recommendation alone.
- status: consensus when the credible evidence points one way (a weaker side with only low or \
very-low grade evidence does not make it contested); leaning when most good evidence points one \
way but credible evidence disagrees; genuinely_contested when good-quality evidence supports two \
or more incompatible answers. A contested answer needs at least two positions; a consensus answer \
usually has one.
- confidence: high only with consistent high or moderate grade evidence in a population like the \
user's; low when evidence is sparse, low grade, or from a different population.
- evidence_grade for a position: the certainty of the body of evidence behind it (high, moderate, \
low, very_low).
- recommendation: one or two sentences with the concrete answer.
- why: a short paragraph on what the evidence shows and how strong it is.
- applies_to_you: how the user's profile (training status, sex, age, goal, schedule) changes the \
answer, and where the studied populations differ from the user. If the profile is empty, say what \
would change the advice.
- practical_protocol: what to do next week, concretely, within what the findings support. No \
individual medical prescriptions.
- what_would_change_this: what new evidence or personal circumstance would change the answer.
- whats_new: only if findings from the last few years shift an older answer; otherwise null.
Write plainly and briefly. Do not write a disclaimer; one is added for you."""


def _render_finding(f: Finding) -> str:
    n = f.sample_size if f.sample_size is not None else "?"
    return (
        f"[{f.id}] ({f.study_type}, grade {f.grade}, n={n}, {f.published.year}) {f.claim}\n"
        f"  population: {f.population}\n  effect: {f.effect or 'not reported'}\n"
        f'  snippet: "{f.snippet}"'
    )


def _render_user(
    intake: Intake,
    findings: list[Finding],
    positions: PositionMap | None,
    verdict: Verdict | None,
    feedback: str | None = None,
) -> str:
    parts = [
        f"Question: {intake.scoped_question}",
        f"User profile: {intake.profile.model_dump_json(exclude_none=True)}",
        "Findings:\n" + "\n".join(_render_finding(f) for f in findings),
    ]
    if verdict is not None:
        parts.append(
            "The judge's verdict is final. Use its status, confidence and positions as given and "
            "explain them:\n" + verdict.model_dump_json(indent=2)
        )
    elif positions is not None:
        parts.append(
            "The position mapper's status and positions are final. Use them as given:\n"
            + positions.model_dump_json(indent=2)
        )
    if feedback:
        parts.append(f"A previous draft of this answer was rejected by review. Fix these problems:\n{feedback}")
    return "\n\n".join(parts)


def _clean_positions(drafts: list[PositionDraft], known: set[str]) -> list[Position]:
    cleaned: list[Position] = []
    for p in drafts:
        ids = [i for i in dict.fromkeys(p.supporting_findings) if i in known]
        if ids:
            cleaned.append(
                Position(
                    stance=p.stance,
                    supporting_findings=ids,
                    evidence_grade=p.evidence_grade,
                    best_counterargument=p.best_counterargument,
                )
            )
    return cleaned


class WriterAgent:
    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        self.model = model

    def run(
        self,
        intake: Intake,
        findings: list[Finding],
        positions: PositionMap | None = None,
        verdict: Verdict | None = None,
        *,
        feedback: str | None = None,
    ) -> CoachAnswer:
        """`feedback` carries the supervisor's reasons when a rejected draft is rewritten."""
        if not intake.in_scope:
            raise WriterError("out-of-scope questions get the intake referral, not a coach answer")
        if not findings:
            raise WriterError("the writer needs at least one finding")

        request = LLMRequest.for_agent(
            AGENT,
            system=SYSTEM_PROMPT,
            user=_render_user(intake, findings, positions, verdict, feedback),
            output_type=WriterDraft,
            model=self.model,
        )
        draft = self.llm.complete(request).output
        known = {f.id for f in findings}

        # Upstream decisions override the model's.
        status: EvidenceStatus = draft.status
        confidence: Confidence = draft.confidence
        if verdict is not None:
            status, confidence, final_positions = verdict.status, verdict.confidence, verdict.positions
        elif positions is not None:
            status, final_positions = positions.status, positions.positions
        else:
            final_positions = _clean_positions(draft.positions, known)
            if not final_positions:
                raise WriterError("writer produced no position backed by a known finding")
            if status == "genuinely_contested" and len(final_positions) < 2:
                status = "leaning"

        cited = [i for i in dict.fromkeys(draft.cited_finding_ids) if i in known]
        if not cited:
            cited = list(dict.fromkeys(i for p in final_positions for i in p.supporting_findings))

        return CoachAnswer(
            status=status,
            recommendation=draft.recommendation,
            confidence=confidence,
            why=draft.why,
            positions=final_positions,
            what_would_change_this=draft.what_would_change_this,
            applies_to_you=draft.applies_to_you,
            practical_protocol=draft.practical_protocol,
            whats_new=draft.whats_new or None,
            cited_finding_ids=cited,
            disclaimer=DISCLAIMER,
        )

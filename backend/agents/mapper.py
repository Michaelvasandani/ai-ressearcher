"""Position Mapper: all findings -> the positions that exist in the literature + a status.

The model only groups findings into stances and proposes a status. Code then:
- drops ids that are not real findings, and gives each finding to one position;
- grades each position from its findings (agents.evidence.grade_body), never
  from the model;
- applies the false-balance rule (agents.evidence.decide_status): weak dissent
  against a stronger lead is set aside and the answer is `consensus`.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from backend.agents.evidence import decide_status, grade_body, render_findings
from backend.llm import LLMClient, LLMRequest
from backend.schemas import EvidenceStatus, Finding, Intake, Position, PositionMap

AGENT = "position_mapper"
MAX_POSITIONS = 4  # bounds the advocate fan-out


class MappedPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: str
    supporting_findings: list[str]
    best_counterargument: str


class MapDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positions: list[MappedPosition]
    status: EvidenceStatus
    rationale: str


class MapperError(ValueError):
    pass


SYSTEM_PROMPT = f"""\
You map the positions that exist in the research on a training or nutrition question. You get the \
question, the user's profile and the findings extracted from studies, each with an id, design, \
grade and verbatim snippet.

Group the findings into positions:
- A position is one answer to the question, stated as a short stance (for example "Sets taken to \
failure produce more muscle growth than sets stopped short"). Group findings by the answer they \
support, not by topic or sub-question.
- Make a separate position only when findings point to a different, incompatible answer. Findings \
that agree but add nuance (a dose, a population) belong to the same position.
- Use 1 to {MAX_POSITIONS} positions. One position is right when the findings all point the same \
way, even if one weak study disagrees; list the weak study as its own position only if it \
really supports a different answer.
- supporting_findings: the exact ids of the findings that support the stance. Each finding goes \
in at most one position. Leave out findings that bear on none of the answers.
- best_counterargument: the strongest point against the position, drawn from the other findings.

status:
- consensus: the credible evidence points one way.
- leaning: most good evidence points one way, but credible evidence disagrees.
- genuinely_contested: good-quality evidence supports two or more incompatible answers.
A side backed only by low or very-low grade evidence (case reports, animal or in-vitro studies, \
preprints, small or conflicted trials) does not make a question contested.

Findings marked down-weighted studied people unlike this user; they still count, but less. \
Do not grade the positions; grades are computed from the findings. rationale: two or three \
sentences on how the findings split."""


def _render_user(intake: Intake, findings: list[Finding], downweighted: Mapping[str, str]) -> str:
    return "\n\n".join(
        [
            f"Question: {intake.scoped_question}",
            f"User profile: {intake.profile.model_dump_json(exclude_none=True)}",
            "Findings:\n" + render_findings(findings, downweighted),
        ]
    )


class MapperAgent:
    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        self.model = model

    def run(self, intake: Intake, findings: list[Finding], downweighted: Mapping[str, str] | None = None) -> PositionMap:
        if not findings:
            raise MapperError("the position mapper needs at least one finding")
        downweighted = downweighted or {}
        request = LLMRequest.for_agent(
            AGENT,
            system=SYSTEM_PROMPT,
            user=_render_user(intake, findings, downweighted),
            output_type=MapDraft,
            model=self.model,
        )
        draft = self.llm.complete(request).output
        return build_position_map(draft, findings, downweighted)


def build_position_map(draft: MapDraft, findings: list[Finding], downweighted: Mapping[str, str]) -> PositionMap:
    """Turn the model's grouping into a PositionMap: clean ids, grade in code, set the status in code."""
    by_id = {f.id: f for f in findings}
    notes: list[str] = []
    used: set[str] = set()
    positions: list[Position] = []
    for p in draft.positions:
        ids: list[str] = []
        for fid in dict.fromkeys(p.supporting_findings):
            if fid not in by_id:
                notes.append(f"dropped unknown finding id {fid!r} from {p.stance!r}")
            elif fid in used:
                notes.append(f"{fid} already supports another position; not counted for {p.stance!r}")
            else:
                ids.append(fid)
                used.add(fid)
        if not ids:
            notes.append(f"dropped position {p.stance!r}: no valid supporting findings")
            continue
        grade, reasons = grade_body([by_id[i] for i in ids], downweighted)
        notes.append(f"{p.stance!r} graded {grade}: {'; '.join(reasons)}")
        positions.append(
            Position(stance=p.stance, supporting_findings=ids, evidence_grade=grade, best_counterargument=p.best_counterargument)
        )
    if not positions:
        raise MapperError("the mapper produced no position backed by a known finding")

    decision = decide_status(positions, by_id, downweighted, draft.status)
    kept, set_aside = decision.positions, list(decision.set_aside)
    if len(kept) > MAX_POSITIONS:
        notes.append(f"kept the {MAX_POSITIONS} strongest of {len(kept)} positions")
        kept, set_aside = kept[:MAX_POSITIONS], kept[MAX_POSITIONS:] + set_aside
    return PositionMap(
        status=decision.status,
        positions=kept,
        proposed_status=draft.status,
        status_reasons=notes + decision.reasons,
        set_aside=set_aside,
        unassigned=[f.id for f in findings if f.id not in used],
    )

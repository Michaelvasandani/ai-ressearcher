"""Advocate: builds the strongest evidence-based case for one position.

One advocate per position. Round 1 is an opening argument; round 2 is the one
rebuttal of the other sides' openings (the graph enforces that there is
exactly one). Advocates argue from finding records only:

- the model's claims carry finding ids; a claim with no id, or with an id that
  is not a finding of this run, is rejected (recorded with its reason, never
  passed on);
- the surviving claims are built into the `Argument` schema (which rejects a
  claim without ids) and checked again with `Argument.check_citations`;
- if no claim survives, the turn fails: the judge still weighs the position
  on its findings.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from backend.agents.evidence import render_findings
from backend.llm import LLMClient, LLMRequest
from backend.schemas import Argument, ArgumentClaim, DebateTurn, Finding, Intake, Position, RejectedClaim

AGENT = "advocate"
MAX_CLAIMS = 6


class ClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    finding_ids: list[str]


class ArgumentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimDraft]


class AdvocateError(ValueError):
    pass


SYSTEM_PROMPT = f"""\
You argue for one position in a structured debate about a training or nutrition question. A \
judge will weigh the evidence quality behind each side, not your rhetoric, so build the \
strongest honest case from the findings given.

Rules:
- Make 2 to {MAX_CLAIMS} claims. Each claim is one or two sentences and cites, in finding_ids, the \
exact ids of the findings that state it. A claim without a valid finding id is thrown out.
- Use only what the findings say: their numbers, designs, populations and grades. No outside \
knowledge, no invented numbers.
- Lead with your strongest evidence (higher grade, larger samples, populations like the user's).
- Be accurate about weaknesses: do not present a preprint, animal, in-vitro or down-weighted \
finding as stronger than it is.
In a rebuttal, answer the other sides' arguments: point to weaknesses in the findings they cite \
(design, grade, sample, population, consistency) and to findings that contradict them. You may \
cite any finding, including theirs."""


def _render_user(
    intake: Intake,
    label: str,
    position: Position,
    findings: list[Finding],
    downweighted: Mapping[str, str],
    opponents: list[DebateTurn],
) -> str:
    parts = [
        f"Question: {intake.scoped_question}",
        f"User profile: {intake.profile.model_dump_json(exclude_none=True)}",
        f"Your position ({label}): {position.stance}\n"
        f"Its supporting findings: {', '.join(position.supporting_findings)} (evidence grade {position.evidence_grade})",
        "Findings:\n" + render_findings(findings, downweighted),
    ]
    if opponents:
        rendered = []
        for t in opponents:
            assert t.argument is not None
            claims = "\n".join(f"  - {c.text} [{', '.join(c.finding_ids)}]" for c in t.argument.claims)
            rendered.append(f"P{t.position}: {t.stance}\n{claims}")
        parts.append("Opening arguments of the other sides, to rebut:\n" + "\n".join(rendered))
        parts.append("Write your rebuttal.")
    else:
        parts.append("Write your opening argument.")
    return "\n\n".join(parts)


def vet_claims(draft: ArgumentDraft, known: set[str]) -> tuple[list[ArgumentClaim], list[RejectedClaim]]:
    """Keep claims that cite only real findings; reject the rest with a reason."""
    kept: list[ArgumentClaim] = []
    rejected: list[RejectedClaim] = []
    for c in draft.claims:
        ids = list(dict.fromkeys(i.strip() for i in c.finding_ids if i.strip()))
        unknown = [i for i in ids if i not in known]
        if not ids:
            rejected.append(RejectedClaim(text=c.text, finding_ids=[], reason="cites no finding id"))
        elif unknown:
            rejected.append(RejectedClaim(text=c.text, finding_ids=ids, reason=f"cites unknown finding ids: {unknown}"))
        elif len(kept) >= MAX_CLAIMS:
            rejected.append(RejectedClaim(text=c.text, finding_ids=ids, reason=f"over the {MAX_CLAIMS}-claim limit"))
        else:
            kept.append(ArgumentClaim(text=c.text, finding_ids=ids))
    return kept, rejected


class AdvocateAgent:
    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        self.model = model

    def run(
        self,
        intake: Intake,
        position_no: int,
        position: Position,
        findings: list[Finding],
        downweighted: Mapping[str, str] | None = None,
        opponents: list[DebateTurn] | None = None,
    ) -> DebateTurn:
        """Opening argument when `opponents` is empty, else the rebuttal of their openings."""
        downweighted = downweighted or {}
        opponents = [t for t in (opponents or []) if t.argument is not None]
        round_ = 2 if opponents else 1
        request = LLMRequest.for_agent(
            AGENT,
            system=SYSTEM_PROMPT,
            user=_render_user(intake, f"P{position_no}", position, findings, downweighted, opponents),
            output_type=ArgumentDraft,
            model=self.model,
        )
        draft = self.llm.complete(request).output
        known = {f.id for f in findings}
        claims, rejected = vet_claims(draft, known)
        turn = DebateTurn(position=position_no, stance=position.stance, round=round_, rejected_claims=rejected)
        if not claims:
            return turn.model_copy(update={"error": "every claim was rejected: none cited a valid finding id"})
        argument = Argument(
            stance=position.stance,
            claims=claims,
            rebuttal_of=" | ".join(f"P{t.position}: {t.stance}" for t in opponents) or None,
        )
        argument.check_citations(known)  # defense in depth: never pass on an unknown id
        return turn.model_copy(update={"argument": argument})

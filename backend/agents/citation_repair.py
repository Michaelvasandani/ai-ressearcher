"""Citation repair: a narrow, cheap fix for a draft that failed only the citation gate.

The writer's draft is good except that some sentences cite nothing, or state a number
their cited findings do not contain. Instead of a full escalated rewrite (and instead of
spending the run's single send-back), one Sonnet call gets just the failing fields, the
gate's problems and the findings, and returns those fields with the citations fixed:
cite the finding that supports a sentence, or soften / drop what no finding supports.
Position stances and counterarguments with an unbacked number are repaired the same way
(they have no inline citations: a stance is backed by its supporting findings, a
counterargument by any finding the answer cites). Status, confidence, which findings
support which position, and every other field stay as they were.

Code applies the new field texts (invented ids are dropped, markers renumbered), and the
graph sends the result through the citation gate again: a repair is never trusted.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from backend.agents.writer import _render_finding
from backend.citations import RAW_ID_MARKER, TEXT_FIELDS, CitationReport, answer_id_texts, with_id_texts
from backend.llm import LLMClient, LLMRequest
from backend.schemas import CoachAnswer, Finding, Intake

AGENT = "citation_repair"

POSITION_FIELD = re.compile(r"^positions\[(\d+)\]\.(stance|best_counterargument)$")


class FieldText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str  # a field name from "Fields to fix", e.g. "why" or "positions[2].best_counterargument"
    text: str


class RepairDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[FieldText]


SYSTEM_PROMPT = """\
You fix citations in a coaching answer. A code check found sentences that cite no finding, \
or that state a number none of their cited findings contains. You get the affected fields \
(with inline citations as finding ids in square brackets), the problems, and the findings.

Return each affected field in full, changed only as needed:
- A sentence that makes a claim without a citation: add the id of the finding that supports \
it, in brackets at the end of the sentence, e.g. "... [sq1:PMID:123]." Sentences that \
appraise the evidence (its size, quality, population, limits) are claims about specific \
findings: cite those findings. If no finding supports the sentence, rewrite it so one does, \
or delete it.
- A number that its cited findings do not contain: cite the finding that does state it, or \
remove the number (do not replace it with a spelled-out number). Never compute numbers \
(no unit conversions, no multiplying a per-kg dose by the user's weight, no sums). Numbers \
from the user's own profile or question are fine without a citation.
- In recommendation, why and whats_new every sentence needs a citation; in the other fields, \
every sentence that states a number does.
- Fields named positions[k].stance or positions[k].best_counterargument take no inline \
citations: keep a number there only if the findings (the position's supporting findings for a \
stance, any finding the answer cites for a counterargument) state it; otherwise rephrase \
without the number.
- Keep every other sentence word for word, keep the meaning and tone, and use only finding \
ids from the list. Return only the fields listed under "Fields to fix"."""


class CitationRepairError(ValueError):
    pass


def _repairable(field: str) -> bool:
    return field in TEXT_FIELDS or POSITION_FIELD.match(field) is not None


def repairable_fields(report: CitationReport) -> list[str]:
    """Fields with gate issues that the repair can rewrite: text fields, position stances and counterarguments."""
    return [f for f in dict.fromkeys(i.field for i in report.issues) if _repairable(f)]


def can_repair(report: CitationReport) -> bool:
    """Every issue is in a field the repair can rewrite (not an unknown finding id, for example)."""
    return not report.ok and bool(report.issues) and all(_repairable(i.field) for i in report.issues)


def relevant_findings(answer: CoachAnswer, findings: list[Finding], report: CitationReport) -> list[Finding]:
    """The findings the answer cites plus those the gate says hold a missing number: enough to fix
    the citations, and a much shorter (cheaper) prompt than every finding of the run."""
    wanted = answer.all_cited_ids() | {fid for i in report.issues for fid in (*i.cited, *i.found_in)}
    return [f for f in findings if f.id in wanted] or findings


def _field_text(answer: CoachAnswer, texts: dict[str, str | None], field: str) -> str:
    m = POSITION_FIELD.match(field)
    if m:
        k, attr = int(m.group(1)), m.group(2)
        return getattr(answer.positions[k - 1], attr) if 1 <= k <= len(answer.positions) else ""
    return texts.get(field) or ""


def _render_user(
    intake: Intake, findings: list[Finding], answer: CoachAnswer, fields: list[str], report: CitationReport
) -> str:
    texts = answer_id_texts(answer)
    problems = "\n".join(f"- {i.field}: {i.detail}" for i in report.issues)
    blocks = "\n\n".join(f"<{f}>\n{_field_text(answer, texts, f)}\n</{f}>" for f in fields)
    return (
        f"Question: {intake.scoped_question}\n"
        f"User profile: {intake.profile.model_dump_json(exclude_none=True)}\n\n"
        f"Fields to fix: {', '.join(fields)}\n\n{blocks}\n\nProblems found by the check:\n{problems}\n\n"
        "Findings:\n" + "\n".join(_render_finding(f) for f in findings)
    )


class CitationRepairAgent:
    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        self.model = model
        self.invented_ids: list[str] = []

    def run(self, intake: Intake, answer: CoachAnswer, findings: list[Finding], report: CitationReport) -> CoachAnswer:
        fields = repairable_fields(report)
        if not fields:
            raise CitationRepairError("no text field to repair")
        request = LLMRequest.for_agent(
            AGENT,
            system=SYSTEM_PROMPT,
            user=_render_user(intake, relevant_findings(answer, findings, report), answer, fields, report),
            output_type=RepairDraft,
            model=self.model,
        )
        draft = self.llm.complete(request).output
        updates: dict[str, str | None] = {}
        positions = list(answer.positions)
        changed_positions = False
        for ft in draft.fields:
            name = ft.field.strip()
            if name not in fields:  # only the fields that were asked for
                continue
            text = ft.text.strip()
            m = POSITION_FIELD.match(name)
            if m:
                k, attr = int(m.group(1)), m.group(2)
                # positions carry no inline citations
                text = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(RAW_ID_MARKER.sub("", text).split()))
                if text and 1 <= k <= len(positions):
                    positions[k - 1] = positions[k - 1].model_copy(update={attr: text})
                    changed_positions = True
                continue
            if not text and name in ("recommendation", "why"):
                continue  # required fields cannot be emptied; the gate will report them again
            updates[name] = text or None
        if not updates and not changed_positions:
            raise CitationRepairError("the repair returned none of the requested fields")
        base = answer.model_copy(update={"positions": positions}) if changed_positions else answer
        repaired, self.invented_ids = with_id_texts(base, updates, {f.id for f in findings})
        return repaired

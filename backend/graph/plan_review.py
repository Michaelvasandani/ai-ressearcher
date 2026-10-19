"""Plan checkpoint (human in the loop): the payloads around the LangGraph `interrupt`.

When a run is started with `require_plan_approval=True`, the graph pauses after
the planner with a `PlanApprovalRequest` (the interrupt value, plain JSON) and
resumes with `Command(resume=PlanDecision)`:

    {"action": "approve"}
    {"action": "edit", "sub_questions": [{"id": "sq1", "question": "..."}, {"question": "a new one"}]}

An edit replaces the sub-questions to research, in order. A sub-question that
keeps its id and its question keeps the planner's search terms; a changed or
new one is searched with the terms given, or with the question itself. An
invalid decision does not resume the run: the graph asks again with `error`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from backend.schemas import Record, SubQuestion


class SubQuestionEdit(Record):
    id: str | None = None  # keep an existing id to keep its search terms
    question: str = Field(min_length=10, max_length=500)
    rationale: str | None = None
    search_terms: list[str] | None = None


class PlanDecision(Record):
    action: Literal["approve", "edit"]
    sub_questions: list[SubQuestionEdit] | None = None
    note: str | None = None  # free text from the reviewer, kept in the trace

    @model_validator(mode="after")
    def _edit_has_questions(self) -> "PlanDecision":
        if self.action == "edit" and not self.sub_questions:
            raise ValueError("an edit must list the sub-questions to research")
        return self

    def apply(self, current: list[SubQuestion], max_sub_questions: int) -> list[SubQuestion]:
        """The sub-questions to research after this decision. Raises ValueError if the edit is invalid."""
        if self.action == "approve":
            return current
        edits = self.sub_questions or []
        if len(edits) > max_sub_questions:
            raise ValueError(f"at most {max_sub_questions} sub-questions fit this run's budget, got {len(edits)}")
        by_id = {sq.id: sq for sq in current}
        given = [e.id for e in edits if e.id]
        if len(given) != len(set(given)):
            raise ValueError("sub-question ids must be unique")
        unknown = [i for i in given if i not in by_id]
        if unknown:
            raise ValueError(f"unknown sub-question ids: {', '.join(unknown)}")
        used = set(given)
        next_no = 1
        out: list[SubQuestion] = []
        for e in edits:
            sid = e.id
            if sid is None:
                while f"sq{next_no}" in used or f"sq{next_no}" in by_id:
                    next_no += 1
                sid = f"sq{next_no}"
                used.add(sid)
            old = by_id.get(sid)
            question = " ".join(e.question.split())
            unchanged = old is not None and old.question == question
            terms = [t.strip() for t in e.search_terms or [] if t.strip()]
            if not terms:
                terms = old.search_terms if unchanged and old else [question.rstrip("?")]
            out.append(
                SubQuestion(
                    id=sid,
                    question=question,
                    rationale=e.rationale or (old.rationale if old else "added by the user at the plan checkpoint"),
                    search_terms=terms,
                )
            )
        return out


class PlanApprovalRequest(Record):
    """What the user reviews. Sent as the interrupt value (JSON) and served by the API."""

    run_id: str
    question: str
    sub_questions: list[SubQuestion]
    dropped: list[SubQuestion] = Field(default_factory=list)  # cut by max_sub_questions
    max_sub_questions: int
    error: str | None = None  # why the previous decision was rejected

"""Planner: turns a scoped question into 3-6 concrete, searchable sub-questions."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from backend.llm import LLMClient, LLMRequest
from backend.schemas import Intake, ResearchPlan, SubQuestion

AGENT = "planner"
MIN_SUB_QUESTIONS = 3
MAX_SUB_QUESTIONS = 6


class SubQuestionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    rationale: str
    search_terms: list[str]


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sub_questions: list[SubQuestionDraft]


class PlannerError(ValueError):
    pass


SYSTEM_PROMPT = f"""\
You plan the literature search for an evidence-based training and nutrition coach. Given the \
user's question and profile, break it into {MIN_SUB_QUESTIONS}-{MAX_SUB_QUESTIONS} sub-questions \
that, answered together from published studies, answer the main question.

Each sub-question must be concrete and answerable by a specific body of research:
- Name the outcome and the comparison or variable, e.g. "Does training a muscle with 10 or more \
weekly sets produce more hypertrophy than fewer than 10 sets in trained adults?", not "What is \
the optimal volume?".
- Name the population when it matters (trained vs untrained, age, sex), using the user's profile \
when it is relevant.
- Prefer sub-questions that meta-analyses and randomized trials address. Include at least one \
sub-question about where the dose-response levels off or what limits it, and one about the \
population or context that best matches this user, when they apply.
- Do not ask about injuries, medical conditions, or medications.
- No duplicates or near-duplicates.

For each sub-question give:
- rationale: one sentence on how it helps answer the main question.
- search_terms: 2-4 PubMed-style queries a researcher would type, using standard terminology \
(e.g. "resistance training volume hypertrophy meta-analysis", "weekly sets muscle thickness \
trained men"). Plain keywords, no boolean operators or field tags."""


_VAGUE_PATTERNS = re.compile(
    r"\b(in general|generally|best way|what about|anything|everything|etc\.?|optimal approach)\b", re.I
)


def lint_sub_question(sq: SubQuestion) -> list[str]:
    """Cheap heuristics that flag vague sub-questions. Empty list = looks searchable."""
    problems: list[str] = []
    words = sq.question.split()
    if len(words) < 6:
        problems.append("too short to be specific")
    if not sq.question.rstrip().endswith("?"):
        problems.append("not phrased as a question")
    if _VAGUE_PATTERNS.search(sq.question):
        problems.append("contains vague wording")
    if not any(len(t.split()) >= 2 for t in sq.search_terms):
        problems.append("search terms are single words")
    return problems


def _render_user(intake: Intake) -> str:
    profile = intake.profile.model_dump_json(exclude_none=True)
    return f"Question: {intake.scoped_question}\nUser profile: {profile}"


def _to_plan(question: str, draft: PlanDraft) -> ResearchPlan:
    seen: set[str] = set()
    subs: list[SubQuestion] = []
    for d in draft.sub_questions:
        key = re.sub(r"\W+", " ", d.question.lower()).strip()
        terms = [t.strip() for t in d.search_terms if t.strip()]
        if not key or key in seen or not terms:
            continue
        seen.add(key)
        subs.append(
            SubQuestion(
                id=f"sq{len(subs) + 1}",
                question=d.question.strip(),
                rationale=d.rationale.strip(),
                search_terms=terms[:4],
            )
        )
    if len(subs) < MIN_SUB_QUESTIONS:
        raise PlannerError(f"planner produced {len(subs)} usable sub-questions; need {MIN_SUB_QUESTIONS}")
    return ResearchPlan(question=question, sub_questions=subs[:MAX_SUB_QUESTIONS])


class PlannerAgent:
    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        self.model = model

    def run(self, intake: Intake) -> ResearchPlan:
        if not intake.in_scope:
            raise PlannerError("out-of-scope questions never reach the planner")
        request = LLMRequest.for_agent(
            AGENT,
            system=SYSTEM_PROMPT,
            user=_render_user(intake),
            output_type=PlanDraft,
            model=self.model,
        )
        draft = self.llm.complete(request).output
        return _to_plan(intake.scoped_question, draft)

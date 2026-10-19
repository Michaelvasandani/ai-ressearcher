"""Offline eval mode: the real graph driven by a deterministic FakeLLM.

`run_eval.py --llm fake --judge heuristic` runs the whole harness (runner, cache, metrics,
report, gate) with no network and no spend. The fake intake is an oracle (it returns the
golden item's expected scope), the planner picks fixture-library sub-questions by topic,
and the other agents reuse the canned responses from the backend's graph tests. The numbers
this produces test the harness and the gate logic; they say nothing about answer quality.
"""

from __future__ import annotations

from typing import Any

from backend.agents.intake import IntakeDraft
from backend.llm import FakeLLM, LLMRequest
from backend.schemas import UserProfile
from backend.tests import graph_fakes as gf
from backend.tools.search import SearchTool
from evals.golden import GoldenItem

PROTEIN_SUB_QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "Does protein supplementation increase resistance training-induced gains in muscle mass?",
        "rationale": "Core question.",
        "search_terms": ["protein supplementation resistance training muscle mass meta-analysis"],
    },
    {
        "question": "Does higher protein intake preserve lean mass during an energy deficit?",
        "rationale": "Dieting context.",
        "search_terms": ["higher protein energy deficit lean mass"],
    },
    {
        "question": "Is habitual protein intake associated with lean mass in older adults?",
        "rationale": "Population context.",
        "search_terms": ["habitual protein intake lean mass older adults"],
    },
]

TOPIC_PLANS: dict[str, tuple[str, ...]] = {
    "training_volume": ("volume", "plateau", "frequency"),
    "training_frequency": ("frequency", "volume", "plateau"),
    "proximity_to_failure": ("failure_similar", "failure_proximity", "failure_rir"),
    "creatine": ("creatine_safety", "creatine_longterm", "creatine_reports"),
}


def _plan(item: GoldenItem) -> dict[str, Any]:
    if item.topic in ("protein", "body_composition", "nutrient_timing", "diet_composition"):
        return {"sub_questions": PROTEIN_SUB_QUESTIONS}
    return gf.plan(*TOPIC_PLANS.get(item.topic, ("volume", "plateau", "frequency")))


def _intake(item: GoldenItem):
    def respond(request: LLMRequest[Any]) -> IntakeDraft:
        return IntakeDraft(
            scope_category=item.expected_scope_category,
            scope_reason="offline oracle: the golden item's expected scope",
            scoped_question=item.question,
            profile=item.profile or UserProfile(),
        )

    return respond


def _writer():
    base = gf.writer_response()

    def respond(request: LLMRequest[Any]) -> Any:
        draft = base(request)
        # The canned protocol names the graph tests' "4 training days"; golden profiles differ.
        return draft.model_copy(update={"practical_protocol": "Spread your sets for each muscle across the week."})

    return respond


def offline_pipeline_llm(item: GoldenItem, search: SearchTool) -> FakeLLM:
    records = getattr(search, "records", [])
    return FakeLLM(
        {
            "intake": _intake(item),
            "planner": _plan(item),
            "researcher": gf.extractor(records),
            "position_mapper": gf.mapper_response(),
            "advocate": gf.advocate_response(),
            "judge": gf.judge_response(),
            "writer": _writer(),
        }
    )

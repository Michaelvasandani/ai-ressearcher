"""Step overrides for time-travel forks.

A fork re-runs a finished run from one step with one thing changed: a prompt
addendum, a different model, or a different tool result. The override lives
on the fork's run record (not in graph state), and `GraphDeps.override_for`
looks it up by run id, so it applies however the fork's graph is resumed
(another process, another serverless invocation).

It is scoped to the forked step's agent, and for researchers to its
sub-question and for advocates to its position, so parallel siblings re-run
unchanged (and replay their recorded calls from the copied ledger).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic import Field, model_validator

from backend import config
from backend.llm import LLMRequest
from backend.schemas import Record, SearchResponse, StudyRecord, utcnow

# Agents whose steps can be forked, and what can be overridden for each.
LLM_AGENTS = frozenset(
    {"intake", "planner", "researcher", "position_mapper", "advocate", "judge", "writer", "citation_repair"}
)
TOOL_AGENTS = frozenset({"researcher"})

ADDENDUM_HEADER = "\n\nAdditional instructions for this run (time-travel override):\n"


class StepOverride(Record):
    """What a fork changes. At least one of prompt_addendum, model, tool_result."""

    prompt_addendum: str | None = Field(None, max_length=4_000)  # appended to the agent's system prompt
    model: str | None = None  # one of config.PRICES (Haiku / Sonnet / Opus)
    # Researcher steps only: these records are what every search of that sub-question returns.
    tool_result: list[StudyRecord] | None = None

    @model_validator(mode="after")
    def _something(self) -> "StepOverride":
        if not (self.prompt_addendum and self.prompt_addendum.strip()) and self.model is None and self.tool_result is None:
            raise ValueError("an override needs a prompt_addendum, a model or a tool_result")
        if self.model is not None and self.model not in config.PRICES:
            raise ValueError(f"unknown model {self.model!r}; use one of {sorted(config.PRICES)}")
        return self


class ScopedOverride(Record):
    """An override bound to the step it was forked from."""

    agent: str
    sub_question_id: str | None = None
    position_no: int | None = None
    override: StepOverride

    def applies(self, agent: str, *, sub_question_id: str | None = None, position_no: int | None = None) -> bool:
        if agent != self.agent:
            return False
        if self.sub_question_id is not None and sub_question_id != self.sub_question_id:
            return False
        if self.position_no is not None and position_no != self.position_no:
            return False
        return True

    def transform(self, request: LLMRequest[Any]) -> LLMRequest[Any]:
        o = self.override
        if o.prompt_addendum and o.prompt_addendum.strip():
            request = replace(request, system=request.system + ADDENDUM_HEADER + o.prompt_addendum.strip())
        if o.model is not None:
            request = replace(request, model=o.model)
        return request

    def search_tool(self) -> "StaticSearchTool | None":
        if self.override.tool_result is None:
            return None
        return StaticSearchTool(self.override.tool_result)


class StaticSearchTool:
    """A SearchTool that returns the same records for every query (the overridden tool result)."""

    name = "override"

    def __init__(self, records: list[StudyRecord]):
        # Provenance: an overridden record is labelled as such and stamped when it is served.
        self.records = [
            r.model_copy(update={"retrieved_at": r.retrieved_at or utcnow(), "endpoint": r.endpoint or "override"})
            for r in records
        ]

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        records = self.records[:max_results]
        return SearchResponse(
            query=query, state="ok" if records else "no_results", records=records,
            message="tool result overridden by a time-travel fork",
        )

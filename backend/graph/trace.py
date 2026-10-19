"""Step-level trace.

Every node appends one `TraceStep` to `RunState.trace`. A step records which
node and agent ran, what it was given (a short summary) and what it produced
(a JSON-able snapshot), the LLM calls and tool calls it made, their tokens,
cost and latency, and the supervisor decision it led to.

Designed for later phases:
- Phase 5 persists the list with the rest of the state (it is plain Pydantic).
- Phase 6 streams steps as SSE events and serves them at /trace.
- Phase 6 forks from a step: `step_id` is stable across replays, `index` is the
  position in the run, and `node` + `sub_question_id` + `attempt` identify the
  graph task to replay.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.llm import CallRecord
from backend.schemas import Record, utcnow

StepStatus = Literal["ok", "degraded", "failed", "skipped"]


class ToolCall(Record):
    tool: str  # e.g. "search:fixture"
    input: str  # the query
    state: str  # SearchResponse.state, or "refused" when a budget stopped it
    results: int = 0
    latency_ms: int = 0
    message: str | None = None


class TraceStep(Record):
    index: int = -1  # assigned by the state reducer, in graph order
    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str
    node: str  # graph node name
    agent: str | None = None  # specialist that ran, or "supervisor"
    sub_question_id: str | None = None
    position_id: str | None = None  # advocate steps: which position ("P1", "P2") argued
    attempt: int = 1  # research pass or writer attempt
    model: str | None = None
    escalated: bool = False
    route_reason: str | None = None
    input_summary: str
    output_summary: str
    output: dict[str, Any] | None = None  # JSON-able snapshot of what the node produced
    decision: str | None = None  # supervisor routing decision taken after this step
    status: StepStatus = "ok"
    error: str | None = None
    llm_calls: list[CallRecord] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    started_at: datetime
    finished_at: datetime


def append_steps(left: list[TraceStep] | None, right: list[TraceStep] | None) -> list[TraceStep]:
    """State reducer: append steps and number them in the order the graph applies writes."""
    out = list(left or [])
    for step in right or []:
        out.append(step.model_copy(update={"index": len(out)}))
    return out


def dump(obj: BaseModel | list[BaseModel] | None) -> Any:
    if obj is None:
        return None
    if isinstance(obj, list):
        return [o.model_dump(mode="json") for o in obj]
    return obj.model_dump(mode="json")


class StepTimer:
    """Times a node and builds its TraceStep."""

    def __init__(self, run_id: str, node: str):
        self.run_id = run_id
        self.node = node
        self.started_at = utcnow()
        self._t0 = time.perf_counter()

    def step(
        self,
        *,
        input_summary: str,
        output_summary: str,
        llm_calls: list[CallRecord] | None = None,
        tool_calls: list[ToolCall] | None = None,
        output: Any = None,
        **kw: Any,
    ) -> TraceStep:
        calls = list(llm_calls or [])
        actual = calls[-1].model if calls else None
        if actual and kw.get("model") and actual != kw["model"] and not calls[-1].replayed:
            # A time-travel override swapped the model: the step shows what actually ran.
            kw["model"] = actual
            kw["route_reason"] = f"{kw.get('route_reason') or 'routed'}; model overridden to {actual}"
        if isinstance(output, (BaseModel, list)):
            output = {"value": dump(output)} if isinstance(output, list) else dump(output)
        return TraceStep(
            run_id=self.run_id,
            node=self.node,
            input_summary=input_summary,
            output_summary=output_summary,
            output=output,
            llm_calls=calls,
            tool_calls=list(tool_calls or []),
            input_tokens=sum(
                c.input_tokens + c.cache_creation_input_tokens + c.cache_read_input_tokens for c in calls
            ),
            output_tokens=sum(c.output_tokens for c in calls),
            cost_usd=sum(c.cost_usd for c in calls),
            latency_ms=int((time.perf_counter() - self._t0) * 1000),
            started_at=self.started_at,
            finished_at=utcnow(),
            **kw,
        )

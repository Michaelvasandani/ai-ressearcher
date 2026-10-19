"""Run budgets, enforced in code rather than in prompts.

A `Budget` is passed per run. It is enforced at three levels:

1. Graph: the plan is cut to `max_sub_questions`; the dispatcher splits what is
   left of the money and tokens between the parallel researchers, holding back
   a reserve for the writer; the supervisor refuses another pass that the
   budget cannot cover.
2. Calls: every LLM call goes through a `MeteredLLM` bound to the node's
   `Allowance`. It refuses a call once the allowance or the deadline is gone,
   and lowers `max_tokens` so the worst-case cost of the call fits what is left.
3. Searches: `MeteredSearch` caps searches per sub-question and stops at the
   deadline.

A refused call raises `BudgetExceeded` (an `LLMError`, so agents treat it like
any failed call); nodes turn it into a degradation or a structured failure.
Nothing here lets a budget problem escape the graph as an exception.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from backend import config
from backend.graph.trace import ToolCall
from backend.llm import CallRecord, LLMClient, LLMError, LLMRequest, LLMResult
from backend.schemas import Record, SearchResponse
from backend.tools.search import SearchTool

T = TypeVar("T", bound=BaseModel)

BudgetKind = Literal["max_sub_questions", "max_searches", "max_tokens", "max_usd", "wall_clock"]

MIN_OUTPUT_TOKENS = 1_024  # below this a structured answer would be truncated; refuse instead
SCHEMA_OVERHEAD_TOKENS = 400  # output-format schema + message framing, not in our text
CHARS_PER_TOKEN = 3.5  # deliberately low so input estimates err high


class Budget(BaseModel):
    """Per-run caps. Phase 7 sweeps `max_usd` over 0.05 / 0.20 / 1.00."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_sub_questions: int = Field(6, ge=1, le=6)
    max_searches_per_sub_question: int = Field(4, ge=1, le=20)
    max_total_tokens: int = Field(400_000, ge=1)
    max_usd: float = Field(1.00, gt=0)
    wall_clock_s: float = Field(300.0, gt=0)
    # Held back from research so the writer can still answer with what was found.
    writer_reserve_fraction: float = Field(0.3, ge=0, lt=1)
    writer_reserve_s: float = Field(60.0, ge=0)

    @classmethod
    def capped(cls, max_usd: float, **kw: Any) -> "Budget":
        return cls(max_usd=max_usd, **kw)


class Spend(Record):
    input_tokens: int = 0  # includes cache reads/writes
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    searches: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Spend") -> "Spend":
        return Spend(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            llm_calls=self.llm_calls + other.llm_calls,
            searches=self.searches + other.searches,
        )

    @classmethod
    def of(cls, records: Iterable[CallRecord], searches: int = 0) -> "Spend":
        recs = list(records)
        return cls(
            input_tokens=sum(
                r.input_tokens + r.cache_creation_input_tokens + r.cache_read_input_tokens for r in recs
            ),
            output_tokens=sum(r.output_tokens for r in recs),
            cost_usd=sum(r.cost_usd for r in recs),
            llm_calls=len(recs),
            searches=searches,
        )


def add_spend(left: Spend | None, right: Spend | None) -> Spend:
    """State reducer: spend from parallel branches adds up."""
    return (left or Spend()) + (right or Spend())


class Allowance(Record):
    """What one node may spend: dollars, tokens, and an absolute deadline (epoch seconds)."""

    usd: float
    tokens: int
    deadline: float

    def split(self, n: int) -> "Allowance":
        return Allowance(usd=self.usd / n, tokens=self.tokens // n, deadline=self.deadline)


def run_deadline(budget: Budget, started_at: float) -> float:
    return started_at + budget.wall_clock_s


def remaining(budget: Budget, spend: Spend, started_at: float) -> Allowance:
    """Everything that is left of the run budget."""
    return Allowance(
        usd=max(0.0, budget.max_usd - spend.cost_usd),
        tokens=max(0, budget.max_total_tokens - spend.total_tokens),
        deadline=run_deadline(budget, started_at),
    )


def research_pool(
    budget: Budget, spend: Spend, started_at: float, now: float
) -> tuple[Allowance | None, BudgetKind | None]:
    """What research may spend now, keeping the writer's reserve. None + the binding budget if nothing."""
    deadline = run_deadline(budget, started_at) - budget.writer_reserve_s
    if now >= deadline:
        return None, "wall_clock"
    usd = budget.max_usd * (1 - budget.writer_reserve_fraction) - spend.cost_usd
    tokens = int(budget.max_total_tokens * (1 - budget.writer_reserve_fraction)) - spend.total_tokens
    if usd <= 0:
        return None, "max_usd"
    if tokens <= 0:
        return None, "max_tokens"
    return Allowance(usd=usd, tokens=tokens, deadline=deadline), None


TYPICAL_INPUT_TOKENS = 2_000  # a researcher extraction prompt; used to judge whether a pass is affordable


def min_call(model: str, input_tokens: int = TYPICAL_INPUT_TOKENS) -> tuple[float, int]:
    """(dollars, tokens) for the smallest useful call to `model`."""
    price = config.price_for(model)
    usd = (input_tokens * price.input + MIN_OUTPUT_TOKENS * price.output) / 1_000_000
    return usd, input_tokens + MIN_OUTPUT_TOKENS


def affordable_count(pool: Allowance, model: str, n: int) -> tuple[int, BudgetKind | None]:
    """How many of `n` parallel passes on `model` the pool can fund (and the binding budget if < n)."""
    usd, tokens = min_call(model)
    by_usd, by_tokens = int(pool.usd // usd), int(pool.tokens // tokens)
    k = min(n, by_usd, by_tokens)
    if k >= n:
        return n, None
    return k, ("max_usd" if by_usd <= by_tokens else "max_tokens")


class BudgetExceeded(LLMError):
    def __init__(self, kind: BudgetKind, message: str):
        super().__init__(message, None)
        self.kind = kind


def estimate_input_tokens(request: LLMRequest[Any]) -> int:
    return math.ceil((len(request.system) + len(request.user)) / CHARS_PER_TOKEN) + SCHEMA_OVERHEAD_TOKENS


class MeteredLLM:
    """Wraps the run's LLM client for one node: enforces its Allowance and keeps its call records.

    Thread-safe; each parallel researcher gets its own instance.
    """

    def __init__(self, inner: LLMClient, allowance: Allowance, clock: Callable[[], float] = time.time):
        self.inner = inner
        self.allowance = allowance
        self.clock = clock
        self.records: list[CallRecord] = []
        self.exhausted: BudgetKind | None = None
        self._lock = threading.Lock()

    @property
    def spend(self) -> Spend:
        return Spend.of(self.records)

    def _refuse(self, kind: BudgetKind, message: str) -> BudgetExceeded:
        self.exhausted = kind
        return BudgetExceeded(kind, message)

    def _admit(self, request: LLMRequest[T]) -> LLMRequest[T]:
        if self.clock() >= self.allowance.deadline:
            raise self._refuse("wall_clock", f"{request.agent}: deadline reached")
        spent = self.spend
        usd_left = self.allowance.usd - spent.cost_usd
        tokens_left = self.allowance.tokens - spent.total_tokens
        est_in = estimate_input_tokens(request)
        price = config.price_for(request.model)
        out_by_tokens = tokens_left - est_in
        out_by_usd = (usd_left - est_in * price.input / 1_000_000) / price.output * 1_000_000
        cap = min(request.max_tokens, out_by_tokens, out_by_usd)
        if cap < MIN_OUTPUT_TOKENS:
            kind: BudgetKind = "max_tokens" if out_by_tokens <= out_by_usd else "max_usd"
            raise self._refuse(
                kind,
                f"{request.agent}: {kind} allowance cannot cover a call to {request.model} "
                f"(~{est_in} input tokens, ${usd_left:.4f} / {tokens_left} tokens left)",
            )
        if cap < request.max_tokens:
            request = replace(request, max_tokens=int(cap))
        return request

    def complete(self, request: LLMRequest[T]) -> LLMResult[T]:
        with self._lock:
            request = self._admit(request)
        try:
            result = self.inner.complete(request)
        except LLMError as e:
            if e.record is not None:  # failed calls still cost money
                with self._lock:
                    self.records.append(e.record)
            raise
        with self._lock:
            self.records.append(result.record)
        return result


class MeteredSearch:
    """Wraps a SearchTool for one researcher pass: caps searches, honours the deadline, logs calls."""

    def __init__(
        self,
        inner: SearchTool,
        max_searches: int,
        deadline: float,
        clock: Callable[[], float] = time.time,
    ):
        self.inner = inner
        self.name = getattr(inner, "name", "search")
        self.max_searches = max_searches
        self.deadline = deadline
        self.clock = clock
        self.calls: list[ToolCall] = []
        self.exhausted: BudgetKind | None = None

    @property
    def executed(self) -> int:
        return sum(1 for c in self.calls if c.state != "refused")

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        tool = f"search:{self.name}"
        if self.executed >= self.max_searches:
            self.exhausted = "max_searches"
            self.calls.append(ToolCall(tool=tool, input=query, state="refused", message="max_searches"))
            return SearchResponse(query=query, state="error", message="search budget exhausted")
        if self.clock() >= self.deadline:
            self.exhausted = "wall_clock"
            self.calls.append(ToolCall(tool=tool, input=query, state="refused", message="wall_clock"))
            return SearchResponse(query=query, state="timed_out", message="run deadline reached")
        t0 = time.perf_counter()
        try:
            resp = self.inner.search(query, max_results=max_results)
        except Exception as e:  # a tool bug must not kill the run
            resp = SearchResponse(query=query, state="error", message=f"{type(e).__name__}: {e}")
        self.calls.append(
            ToolCall(
                tool=tool,
                input=query,
                state=resp.state,
                results=len(resp.records),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                message=resp.message,
            )
        )
        return resp

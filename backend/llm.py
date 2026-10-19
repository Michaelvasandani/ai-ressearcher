"""LLM client abstraction.

Agents call `LLMClient.complete(LLMRequest)` and get back a validated Pydantic
object plus a `CallRecord` (tokens, dollars, latency). Later phases build the
trace and the budget enforcement on top of these records.

Structured output uses the Messages API's JSON-schema output format
(`output_config.format`) rather than a forced tool call: Claude Sonnet 5.5 and
Claude Opus 5.5 reject forced `tool_choice`, and the output format gives the
same schema guarantee. The response is validated again with Pydantic here so
client-side-only constraints (min/max lengths) are enforced too.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from backend import config
from backend.schemas import utcnow

T = TypeVar("T", bound=BaseModel)


# --- Request / result types -------------------------------------------------


@dataclass(frozen=True)
class LLMRequest(Generic[T]):
    agent: str  # which specialist made the call (for traces and fakes)
    system: str
    user: str
    output_type: type[T]
    model: str
    max_tokens: int = 8_000
    effort: config.Effort | None = None

    @classmethod
    def for_agent(cls, agent: str, *, system: str, user: str, output_type: type[T], **overrides: Any) -> "LLMRequest[T]":
        """Build a request using the agent's default model settings from config."""
        defaults = config.AGENT_MODELS[agent]
        params: dict[str, Any] = {
            "model": defaults.model,
            "effort": defaults.effort,
            "max_tokens": defaults.max_tokens,
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(agent=agent, system=system, user=user, output_type=output_type, **params)


class CallRecord(BaseModel):
    """One LLM call, as it will appear in the trace."""

    call_id: str
    agent: str
    model: str
    output_type: str
    started_at: datetime
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: str | None = None
    ok: bool = True
    error: str | None = None


@dataclass
class LLMResult(Generic[T]):
    output: T
    record: CallRecord


class LLMError(Exception):
    def __init__(self, message: str, record: CallRecord | None = None):
        super().__init__(message)
        self.record = record


class StructuredOutputError(LLMError):
    """The model's output did not validate against the requested schema."""


class RefusalError(LLMError):
    """The model declined the request (stop_reason == 'refusal')."""


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    p = config.price_for(model)
    return (
        input_tokens * p.input
        + output_tokens * p.output
        + cache_write * p.cache_write
        + cache_read * p.cache_read
    ) / 1_000_000


class LLMClient(Protocol):
    records: list[CallRecord]

    def complete(self, request: LLMRequest[T]) -> LLMResult[T]: ...


# --- Shared bookkeeping -----------------------------------------------------


@dataclass
class _Raw:
    """What a backend returns before validation."""

    text: str
    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class BaseLLM:
    """Times calls, prices them, validates output and keeps a ledger of CallRecords."""

    def __init__(self, on_call: Callable[[CallRecord], None] | None = None):
        self.records: list[CallRecord] = []
        self._listeners: list[Callable[[CallRecord], None]] = [on_call] if on_call else []

    def subscribe(self, listener: Callable[[CallRecord], None]) -> None:
        self._listeners.append(listener)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def _call(self, request: LLMRequest[Any]) -> _Raw:  # pragma: no cover - abstract
        raise NotImplementedError

    def complete(self, request: LLMRequest[T]) -> LLMResult[T]:
        started_at = utcnow()
        t0 = time.perf_counter()
        record = CallRecord(
            call_id=uuid.uuid4().hex[:12],
            agent=request.agent,
            model=request.model,
            output_type=request.output_type.__name__,
            started_at=started_at,
            latency_ms=0,
        )
        try:
            raw = self._call(request)
        except LLMError as e:
            record = self._finish(record, t0, None, error=str(e))
            e.record = record
            raise
        except Exception as e:  # SDK / network errors
            record = self._finish(record, t0, None, error=f"{type(e).__name__}: {e}")
            raise LLMError(str(e), record) from e

        if raw.stop_reason == "refusal":
            record = self._finish(record, t0, raw, error="refusal")
            raise RefusalError("model refused the request", record)
        try:
            output = request.output_type.model_validate_json(raw.text)
        except ValidationError as e:
            reason = "max_tokens reached" if raw.stop_reason == "max_tokens" else "schema validation failed"
            record = self._finish(record, t0, raw, error=f"{reason}: {e.error_count()} errors")
            raise StructuredOutputError(f"{request.agent}: {reason}\n{e}", record) from e

        record = self._finish(record, t0, raw)
        return LLMResult(output=output, record=record)

    def _finish(self, record: CallRecord, t0: float, raw: _Raw | None, error: str | None = None) -> CallRecord:
        update: dict[str, Any] = {
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "ok": error is None,
            "error": error,
        }
        if raw is not None:
            update.update(
                stop_reason=raw.stop_reason,
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                cache_creation_input_tokens=raw.cache_creation_input_tokens,
                cache_read_input_tokens=raw.cache_read_input_tokens,
                cost_usd=cost_usd(
                    record.model,
                    raw.input_tokens,
                    raw.output_tokens,
                    raw.cache_creation_input_tokens,
                    raw.cache_read_input_tokens,
                ),
            )
        record = record.model_copy(update=update)
        self.records.append(record)
        for listener in self._listeners:
            listener(record)
        return record


# --- Real client ------------------------------------------------------------


class AnthropicLLM(BaseLLM):
    def __init__(self, client: Any | None = None, on_call: Callable[[CallRecord], None] | None = None):
        super().__init__(on_call)
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=config.anthropic_api_key())
        self._client = client
        self._schemas: dict[type[BaseModel], dict[str, Any]] = {}

    def _schema(self, output_type: type[BaseModel]) -> dict[str, Any]:
        if output_type not in self._schemas:
            from anthropic import transform_schema

            self._schemas[output_type] = transform_schema(output_type.model_json_schema())
        return self._schemas[output_type]

    def _call(self, request: LLMRequest[Any]) -> _Raw:
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": self._schema(request.output_type)}
        }
        if request.effort:
            output_config["effort"] = request.effort
        response = self._client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            output_config=output_config,
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        usage = response.usage
        return _Raw(
            text=text,
            stop_reason=response.stop_reason,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        )


# --- Fake client for offline tests ------------------------------------------

FakeResponse = BaseModel | dict[str, Any] | str | Callable[[LLMRequest[Any]], Any] | Exception


class FakeLLM(BaseLLM):
    """Deterministic stand-in for AnthropicLLM.

    `responses` maps an agent name to one response or a list of responses
    (consumed in order; the last one repeats). A response can be a Pydantic
    object, a dict / JSON string (validated against the request's output type,
    so fixtures are schema-checked), a callable taking the request, or an
    exception to raise. Requests are kept in `.requests` for assertions.
    Token counts are estimated from text length so cost accounting is exercised.
    """

    def __init__(self, responses: dict[str, FakeResponse | list[FakeResponse]] | None = None, **kw: Any):
        super().__init__(**kw)
        self._responses: dict[str, list[FakeResponse]] = {}
        for agent, resp in (responses or {}).items():
            self.set(agent, resp)
        self.requests: list[LLMRequest[Any]] = []

    def set(self, agent: str, resp: FakeResponse | list[FakeResponse]) -> None:
        self._responses[agent] = list(resp) if isinstance(resp, list) else [resp]

    def calls_for(self, agent: str) -> list[LLMRequest[Any]]:
        return [r for r in self.requests if r.agent == agent]

    def _call(self, request: LLMRequest[Any]) -> _Raw:
        self.requests.append(request)
        queue = self._responses.get(request.agent)
        if not queue:
            raise LLMError(f"FakeLLM has no response for agent {request.agent!r}")
        resp = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(resp) and not isinstance(resp, (BaseModel, type)):
            resp = resp(request)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, BaseModel):
            text = resp.model_dump_json()
        elif isinstance(resp, dict):
            text = json.dumps(resp, default=str)
        else:
            text = str(resp)
        return _Raw(
            text=text,
            stop_reason="end_turn",
            input_tokens=(len(request.system) + len(request.user)) // 4,
            output_tokens=len(text) // 4,
        )

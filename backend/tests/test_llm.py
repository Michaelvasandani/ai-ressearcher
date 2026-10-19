from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend import config
from backend.llm import (
    AnthropicLLM,
    FakeLLM,
    LLMError,
    LLMRequest,
    RefusalError,
    StructuredOutputError,
    cost_usd,
)


class Out(BaseModel):
    answer: str
    n: int


def _req(agent: str = "planner") -> LLMRequest[Out]:
    return LLMRequest.for_agent(agent, system="sys", user="hello", output_type=Out)


def test_for_agent_uses_config_defaults_and_overrides():
    r = _req("intake")
    assert r.model == config.HAIKU and r.effort == "low"
    r2 = LLMRequest.for_agent("intake", system="s", user="u", output_type=Out, model=config.OPUS, effort=None)
    assert r2.model == config.OPUS and r2.effort == "low"  # None override keeps the default


def test_cost_usd_uses_price_table():
    assert cost_usd(config.SONNET, 1_000_000, 0) == pytest.approx(2.0)
    assert cost_usd(config.SONNET, 0, 1_000_000) == pytest.approx(10.0)
    assert cost_usd(config.HAIKU, 1_000_000, 1_000_000) == pytest.approx(0.6)
    assert cost_usd("unknown-model", 1_000_000, 0) == pytest.approx(config.PRICES[config.OPUS].input)


def test_fake_llm_validates_and_records():
    seen = []
    llm = FakeLLM({"planner": [{"answer": "a", "n": 1}, Out(answer="b", n=2)]}, on_call=seen.append)
    r1 = llm.complete(_req())
    r2 = llm.complete(_req())
    r3 = llm.complete(_req())  # last response repeats
    assert (r1.output.answer, r2.output.answer, r3.output.answer) == ("a", "b", "b")
    assert len(llm.records) == 3 and seen == llm.records
    rec = r1.record
    assert rec.agent == "planner" and rec.model == config.SONNET and rec.ok
    assert rec.input_tokens > 0 and rec.output_tokens > 0 and rec.cost_usd > 0
    assert llm.total_cost_usd == pytest.approx(sum(r.cost_usd for r in llm.records))
    assert len(llm.calls_for("planner")) == 3


def test_fake_llm_schema_violation_raises_and_is_recorded():
    llm = FakeLLM({"planner": {"answer": "a"}})
    with pytest.raises(StructuredOutputError) as exc:
        llm.complete(_req())
    assert exc.value.record is not None and not exc.value.record.ok
    assert llm.records[-1].error.startswith("schema validation failed")


def test_fake_llm_callable_and_exception_responses():
    llm = FakeLLM({"planner": lambda req: {"answer": req.user, "n": 0}})
    assert llm.complete(_req()).output.answer == "hello"
    llm.set("planner", LLMError("boom"))
    with pytest.raises(LLMError):
        llm.complete(_req())
    assert not llm.records[-1].ok
    with pytest.raises(LLMError, match="no response"):
        llm.complete(_req("writer"))


class _StubMessages:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.text, self.stop_reason, self.kwargs = text, stop_reason, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=1000, output_tokens=200, cache_creation_input_tokens=None, cache_read_input_tokens=0
            ),
        )


def test_anthropic_llm_builds_structured_output_request():
    messages = _StubMessages('{"answer": "x", "n": 3}')
    llm = AnthropicLLM(client=SimpleNamespace(messages=messages))
    result = llm.complete(_req())
    assert result.output == Out(answer="x", n=3)
    kw = messages.kwargs
    assert kw["model"] == config.SONNET and kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "hello"}]
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert kw["output_config"]["effort"] == "medium"
    assert "tool_choice" not in kw
    assert result.record.cost_usd == pytest.approx(cost_usd(config.SONNET, 1000, 200))


def test_anthropic_llm_refusal_and_truncation():
    llm = AnthropicLLM(client=SimpleNamespace(messages=_StubMessages("", "refusal")))
    with pytest.raises(RefusalError):
        llm.complete(_req())
    llm = AnthropicLLM(client=SimpleNamespace(messages=_StubMessages('{"answer": "x"', "max_tokens")))
    with pytest.raises(StructuredOutputError, match="max_tokens"):
        llm.complete(_req())
    assert llm.records[-1].input_tokens == 1000  # spend is recorded even on failure


def test_api_key_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTROPIC_API_KEY", "k2")
    assert config.anthropic_api_key() == "k2"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k1")
    assert config.anthropic_api_key() == "k1"

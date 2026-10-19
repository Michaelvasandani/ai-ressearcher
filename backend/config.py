"""Central configuration: model IDs, prices, credentials.

Every model ID used anywhere in the backend comes from this module so that
routing (Phase 2) and the cost-quality frontier (Phase 7) have one place to
change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

# --- Models ---------------------------------------------------------------

HAIKU = "claude-haiku-5-5"  # extraction, grading, classification
SONNET = "claude-sonnet-5-5"  # planning, debate, judge, writing
OPUS = "claude-opus-5-5"  # escalation only

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class AgentModel:
    model: str
    effort: Effort
    max_tokens: int


# Default model per agent. Phase 2's router may override these per call.
AGENT_MODELS: dict[str, AgentModel] = {
    "intake": AgentModel(HAIKU, "low", 4_000),
    "planner": AgentModel(SONNET, "medium", 8_000),
    "researcher": AgentModel(HAIKU, "low", 16_000),
    "writer": AgentModel(SONNET, "medium", 16_000),
}


# --- Prices (USD per 1M tokens, first-party API) ----------------------------


@dataclass(frozen=True)
class Price:
    input: float
    output: float

    @property
    def cache_write(self) -> float:  # 5-minute cache write = 1.25x input
        return self.input * 1.25

    @property
    def cache_read(self) -> float:  # cache read = 0.1x input
        return self.input * 0.1


PRICES: dict[str, Price] = {
    HAIKU: Price(0.10, 0.50),  # prompts <= 100K tokens
    SONNET: Price(2.00, 10.00),
    OPUS: Price(4.00, 20.00),
}


def price_for(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError:
        # Unknown model: price it as Opus so budgets err on the safe side.
        return PRICES[OPUS]


# --- Credentials ------------------------------------------------------------


def anthropic_api_key() -> str | None:
    """Read the API key, accepting the misspelled ANTROPIC_API_KEY as a fallback."""
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTROPIC_API_KEY")

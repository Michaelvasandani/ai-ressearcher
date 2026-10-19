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
    "citation_repair": AgentModel(SONNET, "low", 8_000),
    "position_mapper": AgentModel(SONNET, "medium", 8_000),
    "advocate": AgentModel(SONNET, "low", 4_000),
    "judge": AgentModel(SONNET, "medium", 6_000),
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


# --- Literature search --------------------------------------------------------

SearchBackend = Literal["fixture", "live"]


@dataclass(frozen=True)
class SearchSettings:
    """Search configuration from the environment. `fixture` is the default (and the test default)."""

    backend: SearchBackend = "fixture"
    ncbi_api_key: str | None = None  # raises the PubMed limit from 3 to 10 requests/second
    contact_email: str | None = None  # sent to NCBI (email=), Europe PMC and Crossref (mailto=)
    include_preprints: bool = False  # add SportRxiv as a labeled preprint source
    timeout_s: float = 15.0  # per HTTP request
    max_retries: int = 3  # per HTTP request, for 429 / 5xx / timeouts
    time_budget_s: float = 60.0  # per search() call, across sources and full-text fetches

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SearchSettings":
        e = os.environ if env is None else env
        backend = (e.get("SEARCH_BACKEND") or "fixture").strip().lower()
        if backend not in ("fixture", "live"):
            raise ValueError(f"SEARCH_BACKEND must be 'fixture' or 'live', got {backend!r}")
        return cls(
            backend=backend,  # type: ignore[arg-type]
            ncbi_api_key=e.get("NCBI_API_KEY") or None,
            contact_email=e.get("SEARCH_CONTACT_EMAIL") or None,
            include_preprints=e.get("SEARCH_INCLUDE_PREPRINTS", "").strip().lower() in ("1", "true", "yes"),
            timeout_s=float(e.get("SEARCH_TIMEOUT_S") or 15.0),
            max_retries=int(e.get("SEARCH_MAX_RETRIES") or 3),
            time_budget_s=float(e.get("SEARCH_TIME_BUDGET_S") or 60.0),
        )

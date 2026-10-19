"""Model routing by cost.

One place decides which model each agent call uses:
- Haiku for extraction and classification (intake, researcher extraction).
  Evidence grading is deterministic code (agents/grading.py) and uses no model.
- Sonnet for planning, position mapping, debate, judging, writing and the
  citation-only repair of a draft that failed the citation gate.
- Opus only on escalation: a re-research pass or a rewrite that follows a
  failed supervisor review.

Agents still read their default effort / max_tokens from config.AGENT_MODELS;
the router only picks the model and records why, so the trace can show it.
"""

from __future__ import annotations

from backend import config
from backend.schemas import Record

DEFAULT_MODELS: dict[str, str] = {
    "intake": config.HAIKU,
    "researcher": config.HAIKU,
    "planner": config.SONNET,
    "writer": config.SONNET,
    "citation_repair": config.SONNET,  # citation-only fix of a draft; does not use the send-back
    "position_mapper": config.SONNET,
    "advocate": config.SONNET,
    "judge": config.SONNET,
}

# Agents whose work the supervisor can send back, and the model a second attempt uses.
ESCALATION_MODELS: dict[str, str] = {
    "researcher": config.OPUS,
    "writer": config.OPUS,
}


class Route(Record):
    agent: str
    model: str
    escalated: bool = False
    reason: str


class ModelRouter:
    def __init__(
        self,
        defaults: dict[str, str] | None = None,
        escalations: dict[str, str] | None = None,
    ):
        self.defaults = {**DEFAULT_MODELS, **(defaults or {})}
        self.escalations = {**ESCALATION_MODELS, **(escalations or {})}

    def route(self, agent: str, *, escalate: bool = False, reason: str | None = None) -> Route:
        if escalate and agent in self.escalations:
            return Route(
                agent=agent,
                model=self.escalations[agent],
                escalated=True,
                reason=reason or "escalated after a failed supervisor review",
            )
        try:
            model = self.defaults[agent]
        except KeyError:
            raise KeyError(f"no route for agent {agent!r}") from None
        return Route(agent=agent, model=model, reason=reason or "default route by cost")

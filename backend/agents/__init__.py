"""Specialist agents. Each is a prompt, a tool set and an output schema."""

from backend.agents.advocate import AdvocateAgent
from backend.agents.intake import IntakeAgent
from backend.agents.judge import JudgeAgent
from backend.agents.mapper import MapperAgent
from backend.agents.planner import PlannerAgent
from backend.agents.researcher import ResearcherAgent
from backend.agents.writer import WriterAgent

__all__ = ["AdvocateAgent", "IntakeAgent", "JudgeAgent", "MapperAgent", "PlannerAgent", "ResearcherAgent", "WriterAgent"]

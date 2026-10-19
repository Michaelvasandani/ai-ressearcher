"""Checkpointer wiring.

The graph takes any LangGraph checkpointer. Tests and local runs use the
in-memory saver; Phase 5 plugs in a Postgres (Supabase) saver built with the
same `serializer()`, so state written by one process can be read by another.

The serializer allowlists our own Pydantic models for msgpack deserialization
(LangGraph warns on, and will later block, unregistered types).
"""

from __future__ import annotations

import inspect
from types import ModuleType

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from backend import llm, schemas
from backend.graph import budget, routing, state, trace

_STATE_MODULES: tuple[ModuleType, ...] = (schemas, llm, budget, routing, state, trace)


def state_types() -> list[tuple[str, str]]:
    """(module, class) for every Pydantic model that can appear in run state."""
    out: list[tuple[str, str]] = []
    for module in _STATE_MODULES:
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__:
                out.append((module.__name__, name))
    return sorted(set(out))


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=state_types())


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=serializer())

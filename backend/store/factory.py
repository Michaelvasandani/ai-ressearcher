"""Build the stores from the environment: Postgres when DATABASE_URL is set, memory otherwise."""

from __future__ import annotations

import os

from backend.store.base import Stores


def database_url(env: dict[str, str] | None = None) -> str | None:
    e = os.environ if env is None else env
    return (e.get("DATABASE_URL") or "").strip() or None


def build_stores(env: dict[str, str] | None = None) -> Stores:
    url = database_url(env)
    if url is None:
        from backend.store.memory import memory_stores

        return memory_stores()
    from backend.store.postgres import postgres_stores

    return postgres_stores(url)

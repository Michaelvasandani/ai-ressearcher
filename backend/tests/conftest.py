from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from backend.llm import FakeLLM
from backend.schemas import (
    Finding,
    Intake,
    Provenance,
    SubQuestion,
    UserProfile,
    utcnow,
)
from backend.tools.search import FixtureSearchTool

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gates = {
        "live": ("RUN_LIVE", "live test: set RUN_LIVE=1 to run"),
        "live_network": ("RUN_LIVE_NETWORK", "live network test: set RUN_LIVE_NETWORK=1 to run"),
    }
    for marker, (env, reason) in gates.items():
        if os.environ.get(env) == "1":
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip)
    if not os.environ.get("TEST_DATABASE_URL"):
        skip = pytest.mark.skip(reason="postgres test: set TEST_DATABASE_URL to a throwaway database to run")
        for item in items:
            if item.get_closest_marker("postgres") is not None:
                item.add_marker(skip)


# --- Postgres (integration tests) ---------------------------------------------

APP_TABLES = (
    "run_events", "runs", "conversations", "profiles", "topic_findings", "llm_calls",
    "checkpoint_writes", "checkpoint_blobs", "checkpoints",
)


@pytest.fixture(scope="session")
def pg_url() -> str:
    """TEST_DATABASE_URL with the migrations applied. Tables are truncated per test: use a throwaway DB."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from backend.store.migrate import apply_migrations

    apply_migrations(url)
    return url


@pytest.fixture
def clean_pg(pg_url: str) -> str:
    import psycopg

    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(f"truncate {', '.join(APP_TABLES)}")
    return pg_url


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def search_tool() -> FixtureSearchTool:
    return FixtureSearchTool.from_file()


@pytest.fixture
def profile() -> UserProfile:
    return UserProfile(sex="female", training_status="intermediate", training_age_years=3, days_per_week=4)


@pytest.fixture
def sets_intake(profile: UserProfile) -> Intake:
    return Intake(
        original_question="How many sets per week for hypertrophy?",
        scoped_question="How many sets per muscle per week maximize muscle hypertrophy?",
        in_scope=True,
        scope_category="in_scope",
        scope_reason="training volume question",
        classified_by="llm",
        profile=profile,
    )


@pytest.fixture
def volume_sub_question() -> SubQuestion:
    return SubQuestion(
        id="sq1",
        question="Is there a dose-response relationship between weekly sets per muscle and hypertrophy in resistance-trained adults?",
        rationale="Establishes whether more weekly volume produces more muscle growth.",
        search_terms=["resistance training volume hypertrophy meta-analysis", "weekly sets muscle thickness trained"],
    )


def make_finding(fid: str, **kw: Any) -> Finding:
    data: dict[str, Any] = dict(
        id=fid,
        sub_question_id="sq1",
        claim="More weekly sets produced more hypertrophy.",
        source_id=fid.split(":")[-1],
        source_url=f"https://example.org/fixtures/{fid.split(':')[-1]}",
        study_type="meta_analysis",
        population="trained adults",
        sample_size=100,
        effect="+0.37% per set",
        published=date(2020, 1, 1),
        funding_conflict=False,
        retracted=False,
        snippet="Each additional weekly set per muscle was associated with a 0.37% increase in muscle size.",
        retrieved_at=utcnow(),
        grade="high",
    )
    data.update(kw)
    if "provenance" not in kw:
        data["provenance"] = Provenance(
            source="fixture", endpoint="fixture:studies.json", source_id=data["source_id"], url=data["source_url"],
            retrieved_at=data["retrieved_at"], text_source="full_text",
        )
    return Finding(**data)


@pytest.fixture
def sets_findings() -> list[Finding]:
    return [
        make_finding("sq1:FIXTURE-0001"),
        make_finding("sq1:FIXTURE-0002", study_type="rct", grade="moderate", sample_size=34),
        make_finding("sq1:FIXTURE-0004", study_type="preprint", grade="low", sample_size=2058),
    ]


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.postgres)])
def open_stores(request: pytest.FixtureRequest):
    """Returns a callable that opens the stores. Each call is like a new process: for Postgres a
    new connection pool (and checkpointer) on the same database; for memory the same objects."""
    if request.param == "memory":
        from backend.store.memory import memory_stores

        shared = memory_stores()
        yield lambda: shared
        return
    from backend.store.postgres import connect, postgres_stores

    url = request.getfixturevalue("clean_pg")
    opened = []

    def make():
        stores = postgres_stores(url, pool=connect(url, min_size=1, max_size=4))
        opened.append(stores)
        return stores

    yield make
    for stores in opened:
        stores.close()

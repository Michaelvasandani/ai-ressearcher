"""Run golden items through the real pipeline (`graph.Runner`) and cache the results.

A run is cached as one JSON file per item under `<cache>/<tag>/<item id>.json`
(the full `RunResult`: answer, findings, citation reports, spend, trace, plus
wall-clock time), so metrics and reports can be recomputed without paying for
the runs again. The tag names the configuration (search backend, LLM, budget);
change it, or pass `rerun=True`, to run again.

The harness is backend-agnostic: `search_backend="fixture"` uses the synthetic
library in `backend/tools/fixtures/studies.json` (offline, but it only covers a
few topics, so agreement and coverage on real questions mean little);
`"live"` uses PubMed / Europe PMC / Crossref and needs network access.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.config import SearchSettings
from backend.graph import Budget, GraphDeps, Runner
from backend.graph.builder import RunResult
from backend.llm import LLMClient
from backend.store.memory import memory_stores
from backend.tools.factory import build_search_tool
from backend.tools.search import SearchTool
from evals.golden import GoldenItem

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DEFAULT_CACHE = ROOT / ".cache"

SearchBackendName = Literal["fixture", "live"]
LLMMode = Literal["anthropic", "fake"]


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_backend: SearchBackendName = "fixture"
    llm: LLMMode = "anthropic"
    max_usd: float = 0.20
    max_sub_questions: int | None = None  # None: the Budget default
    wall_clock_s: float | None = None
    tag: str | None = None

    def budget(self) -> Budget:
        kw: dict[str, float | int] = {"max_usd": self.max_usd}
        if self.max_sub_questions is not None:
            kw["max_sub_questions"] = self.max_sub_questions
        if self.wall_clock_s is not None:
            kw["wall_clock_s"] = self.wall_clock_s
        return Budget(**kw)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        if self.tag:
            return self.tag
        sq = f"-sq{self.max_sub_questions}" if self.max_sub_questions else ""
        return f"{self.search_backend}-{self.llm}-usd{self.max_usd:g}{sq}"


class RunCase(BaseModel):
    """One golden item's run, as cached."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    config: EvalConfig
    started_at: datetime
    wall_s: float
    result: RunResult | None = None
    error: str | None = None  # the harness itself failed (the graph never raises)


# --- Building the pipeline ------------------------------------------------------------


def build_llm(mode: LLMMode, item: GoldenItem, search: SearchTool) -> LLMClient:
    if mode == "fake":
        from evals.offline import offline_pipeline_llm

        return offline_pipeline_llm(item, search)
    from backend.llm import AnthropicLLM

    return AnthropicLLM()


def build_search(backend: SearchBackendName) -> SearchTool:
    return build_search_tool(SearchSettings.from_env({**os.environ, "SEARCH_BACKEND": backend}))


def run_one(item: GoldenItem, cfg: EvalConfig, search: SearchTool | None = None) -> RunCase:
    """Run one item through a fresh Runner (in-memory stores, no topic-findings reuse)."""
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        search = search or build_search(cfg.search_backend)
        deps = GraphDeps(llm=build_llm(cfg.llm, item, search), search_tool=search)
        runner = Runner(deps, memory_stores(), reuse_findings=False)  # evals must not reuse earlier runs' findings
        result = runner.start(item.question, item.profile, budget=cfg.budget(), run_id=f"eval-{item.id}")
        return RunCase(item_id=item.id, config=cfg, started_at=started, wall_s=time.perf_counter() - t0, result=result)
    except Exception as e:  # keep the sweep going; the report shows the error
        log.exception("eval run %s crashed", item.id)
        return RunCase(
            item_id=item.id, config=cfg, started_at=started, wall_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}"[:1000],
        )


# --- Cache --------------------------------------------------------------------------


class RunCache:
    def __init__(self, root: Path = DEFAULT_CACHE):
        self.root = Path(root)

    def path(self, cfg: EvalConfig, item_id: str) -> Path:
        return self.root / "runs" / cfg.name / f"{item_id}.json"

    def get(self, cfg: EvalConfig, item_id: str) -> RunCase | None:
        p = self.path(cfg, item_id)
        if not p.exists():
            return None
        try:
            return RunCase.model_validate_json(p.read_text())
        except Exception:
            log.warning("ignoring unreadable cache entry %s", p)
            return None

    def put(self, case: RunCase) -> None:
        p = self.path(case.config, case.item_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(case.model_dump_json())

    def judge_dir(self) -> Path:
        return self.root / "judge"


def run_items(
    items: list[GoldenItem],
    cfg: EvalConfig,
    *,
    cache: RunCache | None = None,
    concurrency: int = 2,
    rerun: bool = False,
    max_total_usd: float | None = None,
    cached_only: bool = False,
    on_done: Callable[[RunCase, bool], None] | None = None,
) -> list[RunCase]:
    """Run (or load from cache) every item. Order of the result follows `items`.

    `max_total_usd` stops starting new runs once the new spend reaches it (runs in flight
    finish). `cached_only` never runs anything: missing items are left out.
    `on_done(case, from_cache)` is called as each item completes.
    """
    cases: dict[str, RunCase] = {}
    todo: list[GoldenItem] = []
    for item in items:
        hit = None if (rerun or cache is None) else cache.get(cfg, item.id)
        if hit is not None:
            cases[item.id] = hit
            if on_done:
                on_done(hit, True)
        elif not cached_only:
            todo.append(item)

    spent = 0.0
    lock = threading.Lock()
    stop = threading.Event()
    # Fixture search is cheap to build; live search shares one rate-limited HTTP client across runs.
    shared_search = build_search(cfg.search_backend) if (todo and cfg.search_backend == "live") else None

    def work(item: GoldenItem) -> RunCase | None:
        if stop.is_set():
            return None
        case = run_one(item, cfg, search=shared_search)
        nonlocal spent
        with lock:
            spent += case.result.spend.cost_usd if case.result else 0.0
            if max_total_usd is not None and spent >= max_total_usd:
                stop.set()
        return case

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, item): item for item in todo}
        for fut in as_completed(futures):
            case = fut.result()
            if case is None:
                continue
            cases[case.item_id] = case
            if cache is not None:
                cache.put(case)
            if on_done:
                on_done(case, False)
    return [cases[i.id] for i in items if i.id in cases]


def load_cases(paths: list[Path]) -> list[RunCase]:
    return [RunCase.model_validate(json.loads(Path(p).read_text())) for p in paths]

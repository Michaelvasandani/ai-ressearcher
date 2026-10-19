"""Worker: advance one run by one slice under a lease, and the reaper.

    POST /internal/runs/{id}/advance     -> Worker.advance(id)   (then re-trigger if paused)
    GET  /internal/cron/reap             -> Worker.reap()        (Vercel cron, every minute)

A slice: take the run's lease (atomic; a second advancer gets `busy`), run the
graph from its checkpoint until it finishes, pauses at the plan checkpoint, or
the slice time is up at a superstep boundary, renewing the lease at every
boundary, then release it. If the slice paused, the caller schedules the next
one. A worker killed mid-slice (crash, or the platform's time limit) leaves an
expired lease behind; the reaper hands the run to a new worker, which resumes
from the last checkpoint (the call ledger makes re-run steps free).
"""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from backend.api.settings import ApiSettings
from backend.graph.runner import Runner

log = logging.getLogger(__name__)

SliceState = Literal[
    "finished", "awaiting_approval", "paused", "error", "missing", "not_running", "busy"
]


@dataclass
class SliceResult:
    run_id: str
    state: SliceState
    steps: int = 0
    seconds: float = 0.0

    @property
    def needs_more(self) -> bool:
        return self.state == "paused"


@dataclass
class ReapResult:
    dispatched: list[str] = field(default_factory=list)
    retried: list[str] = field(default_factory=list)  # `error` runs put back to `running`


class Worker:
    def __init__(self, runner: Runner, settings: ApiSettings, *, clock=time.monotonic):
        self.runner = runner
        self.runs = runner.stores.runs
        self.settings = settings
        self.clock = clock
        self.identity = f"{socket.gethostname()}:{os.getpid()}"

    def advance(self, run_id: str, *, slice_s: float | None = None) -> SliceResult:
        owner = f"{self.identity}:{uuid.uuid4().hex[:8]}"
        ttl = self.settings.lease_ttl_s
        record = self.runs.acquire_lease(run_id, owner, ttl)
        if record is None:
            current = self.runs.get(run_id)
            if current is None:
                return SliceResult(run_id, "missing")
            return SliceResult(run_id, "busy" if current.status == "running" else "not_running")
        limit = self.settings.slice_s if slice_s is None else slice_s
        t0 = self.clock()
        try:
            out = self.runner.advance(
                run_id,
                should_stop=lambda: self.clock() - t0 >= limit,
                heartbeat=lambda: self.runs.renew_lease(run_id, owner, ttl),
            )
        finally:
            try:
                self.runs.release_lease(run_id, owner)
            except Exception:  # the lease expires on its own
                log.warning("could not release the lease on run %s", run_id, exc_info=True)
        result = SliceResult(run_id, out.state, out.steps, round(self.clock() - t0, 3))
        log.info("run %s: slice ended %s after %d steps in %.1fs", run_id, result.state, result.steps, result.seconds)
        return result

    def run_to_completion(self, run_id: str, *, max_slices: int = 1_000) -> SliceResult:
        """Advance slice after slice in this thread (local / thread executor)."""
        result = SliceResult(run_id, "missing")
        for _ in range(max_slices):
            result = self.advance(run_id)
            if not result.needs_more:
                break
        return result

    def reap(self, *, limit: int = 20) -> ReapResult:
        """Runs whose worker is gone: expired leases, lost first dispatches, retryable errors."""
        out = ReapResult()
        for rec in self.runs.reapable(
            stale_after_s=self.settings.reap_stale_s, max_attempts=self.settings.max_auto_retries, limit=limit
        ):
            if rec.status == "error":
                if self.runs.transition(rec.run_id, ("error",), status="running", error=None) is None:
                    continue
                out.retried.append(rec.run_id)
            out.dispatched.append(rec.run_id)
        return out

"""Executors: how a run gets advanced after the API hands it off.

`Executor.submit(run_id)` means "make sure this run gets advanced soon". It never
runs the graph in the caller's request on Vercel, and it is safe to call twice:
the worker's lease lets only one advancer in at a time.

- `ThreadExecutor` (local dev, tests): a thread pool in this process runs slice
  after slice until the run finishes or waits for plan approval.
- `HttpExecutor` (Vercel): POSTs `/internal/runs/{id}/advance` to the deployment
  with the worker secret. That request is its own function invocation (its own
  time limit); it runs one slice and, if the run is not done, submits the next
  one the same way. The POST waits only until the request is accepted (a short
  read timeout), not for the slice. If the hand-off is lost anyway, the reaper
  cron finds the run (running, no lease, no progress) and dispatches it again.
- `InlineExecutor` (tests): runs to completion in the caller.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol

import httpx

from backend.api.worker import Worker

log = logging.getLogger(__name__)


class Executor(Protocol):
    def submit(self, run_id: str) -> None: ...

    def shutdown(self) -> None: ...


class InlineExecutor:
    def __init__(self, worker: Worker):
        self.worker = worker

    def submit(self, run_id: str) -> None:
        self.worker.run_to_completion(run_id)

    def shutdown(self) -> None:
        pass


class ThreadExecutor:
    def __init__(self, worker: Worker, threads: int = 4):
        self.worker = worker
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="run-worker")
        self._futures: dict[str, Future[object]] = {}
        self._lock = threading.Lock()

    def submit(self, run_id: str) -> None:
        with self._lock:
            running = self._futures.get(run_id)
            if running is not None and not running.done():
                # Already being advanced here. It may be just finishing a slice, so queue a
                # follow-up; the lease turns it into a no-op if nothing is left to do.
                self._futures[run_id] = self.pool.submit(self._after, running, run_id)
                return
            self._futures[run_id] = self.pool.submit(self._run, run_id)

    def _after(self, previous: Future[object], run_id: str) -> object:
        try:
            previous.result()
        except Exception:
            pass
        return self._run(run_id)

    def _run(self, run_id: str) -> object:
        try:
            return self.worker.run_to_completion(run_id)
        except BaseException:
            log.exception("worker crashed on run %s", run_id)
            raise

    def wait(self, run_id: str, timeout: float | None = 30.0) -> object:
        """Block until the run's current work is done (tests)."""
        while True:
            with self._lock:
                fut = self._futures.get(run_id)
            if fut is None:
                return None
            result = fut.result(timeout=timeout)
            with self._lock:
                if self._futures.get(run_id) is fut:
                    return result

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


class HttpExecutor:
    """Hands each slice to a new function invocation through the deployment's own URL."""

    ACCEPT_TIMEOUT_S = 3.0  # wait for the request to be accepted, not for the slice

    def __init__(
        self,
        base_url: str,
        secret: str,
        *,
        transport: httpx.BaseTransport | None = None,
        bypass_secret: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.secret = secret
        # Vercel Deployment Protection blocks un-authenticated requests to preview deployments;
        # "Protection Bypass for Automation" exposes VERCEL_AUTOMATION_BYPASS_SECRET for this.
        self.headers = {"Authorization": f"Bearer {secret}"}
        bypass = bypass_secret if bypass_secret is not None else os.environ.get("VERCEL_AUTOMATION_BYPASS_SECRET")
        if bypass:
            self.headers["x-vercel-protection-bypass"] = bypass
        self.client = httpx.Client(
            timeout=httpx.Timeout(self.ACCEPT_TIMEOUT_S, connect=5.0), transport=transport
        )

    def submit(self, run_id: str) -> None:
        url = f"{self.base_url}/internal/runs/{run_id}/advance"
        try:
            resp = self.client.post(url, headers=self.headers)
            if resp.status_code >= 400:
                log.warning("dispatch of run %s got HTTP %s; the reaper will retry", run_id, resp.status_code)
        except httpx.ReadTimeout:
            pass  # accepted and running: the slice outlives this short wait
        except httpx.HTTPError:
            log.warning("could not dispatch run %s to %s; the reaper will pick it up", run_id, url, exc_info=True)

    def shutdown(self) -> None:
        self.client.close()

"""Shared HTTP client for the literature APIs: rate limits, retries, timeouts.

Every live source goes through `HttpClient.get`, which
- waits for its host's rate limiter (thread-safe, shared by parallel researchers),
- retries 429 / 5xx with exponential backoff (honouring Retry-After), up to a cap,
- turns transport failures into typed `FetchError`s.

Source tools catch `FetchError` and turn it into a `SearchResponse.state`, so a
failure never escapes as an exception into the graph.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

import httpx

USER_AGENT = "EvidenceCoach/0.1 (research assistant; +https://github.com/)"
FailureKind = Literal["rate_limited", "timed_out", "not_found", "error"]


class FetchError(Exception):
    """A request that failed after retries. `kind` maps onto SearchState."""

    def __init__(self, kind: FailureKind, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


class RateLimiter:
    """Minimum spacing between requests to one host, shared across threads.

    NCBI allows 3 requests/second without an API key and 10 with one; Europe PMC
    and Crossref ask clients to be polite. Reserving a slot under the lock and
    sleeping outside it keeps parallel researchers from bursting together.
    """

    def __init__(
        self,
        per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.interval = 1.0 / per_second if per_second > 0 else 0.0
        self.clock = clock
        self.sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = self.clock()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            self.sleep(slot - now)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None  # HTTP-date form: fall back to our own backoff


class HttpClient:
    """Thin, thread-safe wrapper over `httpx.Client` with per-host limits and retries."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        max_backoff: float = 8.0,
        rate_limits: Mapping[str, float] | None = None,  # host -> requests/second
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ):
        self.client = client or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.sleep = sleep
        self.jitter = jitter
        self.limiters = {host: RateLimiter(rps, sleep=sleep) for host, rps in (rate_limits or {}).items()}
        self.requests: list[str] = []  # every attempted URL, for tests and debugging

    def _backoff(self, attempt: int, response: httpx.Response | None) -> float:
        hinted = _retry_after(response) if response is not None else None
        delay = hinted if hinted is not None else self.backoff_base * (2**attempt) * (1 + 0.25 * self.jitter())
        return min(delay, self.max_backoff)

    def get(self, url: str, params: Mapping[str, Any] | None = None) -> httpx.Response:
        """GET with retries. Returns a 2xx response or raises FetchError."""
        host = httpx.URL(url).host
        limiter = self.limiters.get(host)
        last: FetchError | None = None
        for attempt in range(self.max_retries + 1):
            if limiter:
                limiter.wait()
            self.requests.append(url)
            response: httpx.Response | None = None
            try:
                response = self.client.get(url, params=params, timeout=self.timeout)
            except httpx.TimeoutException as e:
                last = FetchError("timed_out", f"{host}: timed out ({type(e).__name__})")
            except httpx.HTTPError as e:
                last = FetchError("error", f"{host}: {type(e).__name__}: {e}")
            else:
                code = response.status_code
                if code < 300:
                    return response
                if code == 404:
                    raise FetchError("not_found", f"{host}: 404", code)
                if code == 429:
                    last = FetchError("rate_limited", f"{host}: 429 rate limited", code)
                elif code >= 500:
                    last = FetchError("error", f"{host}: HTTP {code}", code)
                else:  # other 4xx: retrying will not help
                    raise FetchError("error", f"{host}: HTTP {code}", code)
            if attempt < self.max_retries:
                self.sleep(self._backoff(attempt, response))
        assert last is not None
        last.args = (f"{last.args[0]} after {self.max_retries + 1} attempts",)
        raise last

    def close(self) -> None:
        self.client.close()

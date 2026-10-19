"""API settings from the environment (`backend/.env` is loaded by the entrypoints)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

ExecutorKind = Literal["thread", "http", "inline"]
AuthMode = Literal["header", "jwt"]


def _float(env: dict[str, str] | os._Environ[str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class ApiSettings:
    # Execution. "thread": advance runs in a background thread pool in this process (local dev,
    # tests). "http": POST /internal/runs/{id}/advance on WORKER_BASE_URL (Vercel: each slice is
    # its own function invocation). "inline": advance synchronously in the request (tests).
    executor: ExecutorKind = "thread"
    worker_secret: str | None = None  # protects /internal/* (Authorization: Bearer <secret>)
    cron_secret: str | None = None  # Vercel cron sends Authorization: Bearer $CRON_SECRET
    worker_base_url: str | None = None  # where the http executor reaches /internal/...
    # One advance slice stops at the first superstep boundary after this many seconds. Keep it well
    # under the function's maxDuration (300s): a superstep (parallel researchers) can take ~90s.
    slice_s: float = 180.0
    # A lease outlives the function that holds it (maxDuration + margin), so an expired lease
    # means its worker is gone. Renewed at every superstep boundary.
    lease_ttl_s: float = 330.0
    reap_stale_s: float = 60.0  # running, no lease, no progress for this long -> re-dispatch
    max_auto_retries: int = 3  # advances that may end in an error before a run stays `error`
    threads: int = 4  # thread executor pool size
    # SSE: poll run_events every sse_poll_s; end the response after sse_max_s (the browser's
    # EventSource reconnects with Last-Event-ID), well inside the function time limit.
    sse_poll_s: float = 1.0
    sse_max_s: float = 240.0
    sse_keepalive_s: float = 15.0
    # Auth (minimal): X-User-Id header, or a Supabase JWT (HS256, SUPABASE_JWT_SECRET) when set.
    auth: AuthMode = "header"
    supabase_jwt_secret: str | None = None
    cors_origins: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ApiSettings":
        e = os.environ if env is None else env
        on_vercel = bool(e.get("VERCEL"))
        executor = (e.get("EXECUTOR") or ("http" if on_vercel else "thread")).strip().lower()
        if executor not in ("thread", "http", "inline"):
            raise ValueError(f"EXECUTOR must be thread, http or inline, got {executor!r}")
        base = (e.get("WORKER_BASE_URL") or "").strip() or None
        if base is None and e.get("VERCEL_PROJECT_PRODUCTION_URL") and e.get("VERCEL_ENV") == "production":
            base = f"https://{e['VERCEL_PROJECT_PRODUCTION_URL']}"
        if base is None and e.get("VERCEL_URL"):
            base = f"https://{e['VERCEL_URL']}"
        jwt_secret = (e.get("SUPABASE_JWT_SECRET") or "").strip() or None
        auth = (e.get("AUTH_MODE") or ("jwt" if jwt_secret else "header")).strip().lower()
        if auth not in ("header", "jwt"):
            raise ValueError(f"AUTH_MODE must be header or jwt, got {auth!r}")
        return cls(
            executor=executor,  # type: ignore[arg-type]
            worker_secret=(e.get("WORKER_SECRET") or "").strip() or None,
            cron_secret=(e.get("CRON_SECRET") or "").strip() or None,
            worker_base_url=base.rstrip("/") if base else None,
            slice_s=_float(e, "ADVANCE_SLICE_S", 180.0),
            lease_ttl_s=_float(e, "LEASE_TTL_S", 330.0),
            reap_stale_s=_float(e, "REAP_STALE_S", 60.0),
            max_auto_retries=int(_float(e, "MAX_AUTO_RETRIES", 3)),
            threads=int(_float(e, "EXECUTOR_THREADS", 4)),
            sse_poll_s=_float(e, "SSE_POLL_S", 1.0),
            sse_max_s=_float(e, "SSE_MAX_S", 240.0),
            auth=auth,  # type: ignore[arg-type]
            supabase_jwt_secret=jwt_secret,
            cors_origins=tuple(o.strip() for o in (e.get("CORS_ORIGINS") or "").split(",") if o.strip()),
        )

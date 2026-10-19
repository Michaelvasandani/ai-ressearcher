"""Server-sent events for run progress.

The stream polls `run_events` for `seq > last seen` (no in-process pub/sub, so it works
across serverless instances: the worker writing events and the function serving the
stream are different invocations). Each event goes out as

    id: <seq>
    event: <type>          run_started | step | plan_ready | plan_decided | run_resumed | run_finished | run_error
    data: <json>           step events carry a UI `message` and running `totals` (tokens, dollars)

A response lasts at most SSE_MAX_S (Vercel functions are short-lived); the browser's
EventSource reconnects by itself and sends `Last-Event-ID`, so nothing is missed or repeated.
When the run is finished (and its final events are out) the stream sends `event: end` and
closes; the client should then close its EventSource instead of reconnecting.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from starlette.concurrency import run_in_threadpool

from backend.api.settings import ApiSettings
from backend.store.base import Stores
from backend.store.models import RunEvent

END_EVENTS = frozenset({"run_finished"})
GRACE_POLLS = 3  # polls to wait for a terminal run's last event to land


def format_event(event: RunEvent) -> str:
    data = {"seq": event.seq, "type": event.type, "at": event.created_at.isoformat(), **event.data}
    return f"id: {event.seq}\nevent: {event.type}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


def format_end(payload: dict[str, Any]) -> str:
    return f"event: end\ndata: {json.dumps(payload, default=str)}\n\n"


async def event_stream(stores: Stores, settings: ApiSettings, run_id: str, after: int = 0) -> AsyncIterator[str]:
    last = after
    started = time.monotonic()
    last_sent = started
    terminal_polls = 0
    seen_end = False
    yield f"retry: {int(max(settings.sse_poll_s, 0.5) * 1000)}\n\n"
    while True:
        # Status first, then events: an event written before the status changed is never missed.
        record = await run_in_threadpool(stores.runs.get, run_id)
        events = await run_in_threadpool(stores.events.list, run_id, after_seq=last)
        for event in events:
            yield format_event(event)
            last = event.seq
            last_sent = time.monotonic()
            seen_end = seen_end or event.type in END_EVENTS
        if record is None:
            yield format_end({"status": "missing"})
            return
        done = record.status == "completed" or (
            record.status == "error" and record.attempts >= settings.max_auto_retries
        )
        if done:
            if seen_end or terminal_polls >= GRACE_POLLS or (record.status == "error" and not events):
                yield format_end({"status": record.status, "outcome_status": record.outcome_status, "last_event_id": last})
                return
            terminal_polls += 1
        now = time.monotonic()
        if now - started >= settings.sse_max_s:
            return  # the client reconnects with Last-Event-ID
        if now - last_sent >= settings.sse_keepalive_s:
            yield ": keep-alive\n\n"
            last_sent = now
        await asyncio.sleep(settings.sse_poll_s)

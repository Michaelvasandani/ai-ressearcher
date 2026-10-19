"""FastAPI app: research runs, progress (SSE), traces, time-travel forks, conversations, profile.

    POST /research                      start a run; returns its id at once (202)
    GET  /research/{id}                 status, plan while awaiting approval, answer + outcome when done
    POST /research/{id}/plan            approve or edit the plan; the run resumes
    GET  /research/{id}/events          SSE progress (polls run_events; Last-Event-ID resumes)
    GET  /research/{id}/trace           the full step log
    POST /research/{id}/fork?from_step  re-run from a step with an override, as a new run
    GET  /conversations, GET/DELETE /conversations/{id}, GET/PUT /profile
    POST /internal/runs/{id}/advance    worker: one slice under a lease (WORKER_SECRET)
    GET  /internal/cron/reap            reaper: re-dispatch runs whose worker is gone (CRON_SECRET)

`create_app(services)` takes everything injectable (stores, runner, executor), so
tests run it on in-memory stores with a FakeLLM; `create_app()` builds it all
from the environment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from backend.api.auth import check_secret, user_from_request
from backend.api.executor import Executor, HttpExecutor, InlineExecutor, ThreadExecutor
from backend.api.models import (
    ConversationView,
    ProfileView,
    ResearchCreated,
    ResearchRequest,
    RunSummary,
    RunView,
    TraceView,
)
from backend.api.settings import ApiSettings
from backend.api.sse import event_stream
from backend.api.worker import Worker
from backend.graph.fork import ForkError, ForkInfo, answer_diff
from backend.graph.overrides import StepOverride
from backend.graph.plan_review import PlanDecision
from backend.graph.runner import Runner
from backend.schemas import UserProfile, utcnow
from backend.store.models import Conversation, RunRecord

log = logging.getLogger(__name__)

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env() -> None:
    """Load backend/.env for local runs. Variables already set (e.g. on Vercel) win."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional in production, where the platform sets the environment
        return
    load_dotenv(ENV_FILE, override=False)


@dataclass
class Services:
    settings: ApiSettings
    runner: Runner
    worker: Worker
    executor: Executor

    @property
    def stores(self):  # noqa: ANN201
        return self.runner.stores


def current_user(request: Request) -> str | None:
    return user_from_request(request, request.app.state.services.settings)


def require_user(request: Request) -> str:
    uid = current_user(request)
    if uid is None:
        raise HTTPException(401, "sign in (or send X-User-Id)")
    return uid


User = Annotated[str | None, Depends(current_user)]
RequiredUser = Annotated[str, Depends(require_user)]


def build_executor(settings: ApiSettings, worker: Worker) -> Executor:
    if settings.executor == "inline":
        return InlineExecutor(worker)
    if settings.executor == "http":
        if not settings.worker_base_url or not settings.worker_secret:
            raise RuntimeError("EXECUTOR=http needs WORKER_BASE_URL (or VERCEL_URL) and WORKER_SECRET")
        return HttpExecutor(settings.worker_base_url, settings.worker_secret)
    return ThreadExecutor(worker, threads=settings.threads)


def build_services(settings: ApiSettings | None = None) -> Services:
    """Everything from the environment: Postgres if DATABASE_URL is set, the real model, SEARCH_BACKEND."""
    from backend.graph.nodes import GraphDeps
    from backend.llm import AnthropicLLM
    from backend.store import build_stores
    from backend.tools.factory import build_search_tool

    settings = settings or ApiSettings.from_env()
    runner = Runner(GraphDeps(llm=AnthropicLLM(), search_tool=build_search_tool()), build_stores())
    worker = Worker(runner, settings)
    return Services(settings, runner, worker, build_executor(settings, worker))


def create_app(services: Services | None = None) -> FastAPI:
    if services is None:
        load_env()
        if not logging.getLogger().handlers:
            logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
        services = build_services()
    s = services
    app = FastAPI(title="Evidence Coach API", version="0.6.0")
    app.state.services = s
    if s.settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(s.settings.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

    def run_for(run_id: str, uid: str | None) -> RunRecord:
        record = s.stores.runs.get(run_id)
        # Someone else's run is reported as missing, not forbidden: run ids are not enumerable.
        if record is None or (record.user_id is not None and record.user_id != uid):
            raise HTTPException(404, f"run {run_id} not found")
        return record

    def links(run_id: str) -> dict[str, str]:
        base = f"/research/{run_id}"
        return {"self": base, "events": f"{base}/events", "trace": f"{base}/trace", "plan": f"{base}/plan"}

    # --- health ---------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "executor": s.settings.executor, "time": utcnow().isoformat()}

    # --- research -------------------------------------------------------------

    @app.post("/research", status_code=202, response_model=ResearchCreated)
    def start_research(body: ResearchRequest, uid: User) -> ResearchCreated:
        profile = body.profile
        if profile is None and uid is not None:
            profile = s.stores.profiles.get(uid)
        conversation_id = body.conversation_id
        if conversation_id is not None:
            conv = s.stores.conversations.get(conversation_id)
            if conv is None or conv.user_id != uid:
                raise HTTPException(404, f"conversation {conversation_id} not found")
            s.stores.conversations.touch(conversation_id)
        else:
            conv = s.stores.conversations.create(
                Conversation(id=_new_id(), user_id=uid, title=_title(body.question))
            )
            conversation_id = conv.id
        record = s.runner.create(
            body.question, profile, budget=body.budget, user_id=uid, conversation_id=conversation_id,
            require_plan_approval=body.require_plan_approval,
        )
        s.executor.submit(record.run_id)
        return ResearchCreated(
            run_id=record.run_id, conversation_id=conversation_id, status="running", links=links(record.run_id)
        )

    @app.get("/research/{run_id}", response_model=RunView)
    def get_research(run_id: str, uid: User) -> RunView:
        record = run_for(run_id, uid)
        result = s.runner.get(run_id)
        view = RunView(
            run_id=run_id, status=record.status, question=record.question,
            conversation_id=record.conversation_id, created_at=record.created_at, updated_at=record.updated_at,
            plan_decision_pending=record.pending_decision is not None, error=record.error,
            parent_run_id=record.parent_run_id, links=links(run_id),
            fork=ForkInfo.model_validate(record.fork) if record.fork else None,
        )
        if result is None:
            return view
        view.sub_questions = result.sub_questions
        view.spend = result.spend
        view.steps = len(result.trace)
        view.citations = result.citation_reports[-1] if result.citation_reports else None
        if record.status == "awaiting_approval" and record.pending_decision is None:
            view.plan = result.awaiting_approval
        if record.status == "completed":
            view.answer = result.answer
            view.outcome = result.outcome
            cited = result.answer.all_cited_ids() if result.answer else None
            view.findings = [f for f in result.findings if cited is None or f.id in cited]
            if view.fork is not None:
                parent = s.runner.get(view.fork.parent_run_id)
                if parent is not None:
                    view.diff = answer_diff(parent, result, view.fork.from_step)
        return view

    @app.post("/research/{run_id}/plan", status_code=202, response_model=ResearchCreated)
    def decide_plan(run_id: str, decision: PlanDecision, uid: User) -> ResearchCreated:
        record = run_for(run_id, uid)
        if record.status != "awaiting_approval" or record.pending_decision is not None:
            raise HTTPException(409, f"run {run_id} is {record.status}, not waiting for plan approval")
        result = s.runner.get(run_id)
        request = result.awaiting_approval if result else None
        if request is None:
            raise HTTPException(409, f"run {run_id} has no pending plan")
        try:
            decision.apply(request.sub_questions, request.max_sub_questions)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        payload = decision.model_dump(mode="json")
        if s.stores.runs.transition(run_id, ("awaiting_approval",), status="running", pending_decision=payload) is None:
            raise HTTPException(409, "the plan was decided concurrently")
        s.stores.events.append(run_id, "plan_decided", {"action": decision.action, "note": decision.note})
        s.executor.submit(run_id)
        return ResearchCreated(
            run_id=run_id, conversation_id=record.conversation_id, status="running", links=links(run_id)
        )

    @app.get("/research/{run_id}/events")
    def research_events(
        run_id: str,
        uid: User,
        after: Annotated[int | None, Query(ge=0, description="resume after this event id")] = None,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        run_for(run_id, uid)
        start = after or 0
        if last_event_id and last_event_id.strip().isdigit():
            start = max(start, int(last_event_id.strip()))
        return StreamingResponse(
            event_stream(s.stores, s.settings, run_id, start),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.get("/research/{run_id}/trace", response_model=TraceView)
    def research_trace(run_id: str, uid: User) -> TraceView:
        record = run_for(run_id, uid)
        result = s.runner.get(run_id)
        if result is None:
            return TraceView(run_id=run_id, status=record.status, steps=[])
        inherited = sum(1 for st in result.trace if st.run_id != run_id)
        spend = result.spend
        return TraceView(
            run_id=run_id, status=record.status, steps=result.trace, inherited_steps=inherited, spend=spend,
            totals={
                "steps": len(result.trace),
                "tokens": spend.total_tokens,
                "cost_usd": spend.cost_usd,
                "latency_ms": sum(st.latency_ms for st in result.trace),
                "llm_calls": spend.llm_calls,
                "tool_calls": sum(len(st.tool_calls) for st in result.trace),
            },
        )

    @app.post("/research/{run_id}/fork", status_code=202, response_model=ResearchCreated)
    def fork_research(
        run_id: str,
        override: StepOverride,
        uid: User,
        from_step: Annotated[int, Query(ge=0)],
    ) -> ResearchCreated:
        record = run_for(run_id, uid)
        try:
            fork = s.runner.fork(run_id, from_step, override, user_id=uid if uid is not None else record.user_id)
        except KeyError as e:
            raise HTTPException(404, f"run {run_id} not found") from e
        except ForkError as e:
            raise HTTPException(409 if "fork a finished run" in str(e) else 422, str(e)) from e
        if fork.conversation_id:
            s.stores.conversations.touch(fork.conversation_id)
        s.executor.submit(fork.run_id)
        return ResearchCreated(
            run_id=fork.run_id, conversation_id=fork.conversation_id, status="running",
            parent_run_id=run_id, links=links(fork.run_id),
        )

    # --- conversations and profile ----------------------------------------------

    @app.get("/conversations", response_model=list[Conversation])
    def list_conversations(uid: User, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[Conversation]:
        return s.stores.conversations.list(uid, limit=limit)

    def conversation_for(conversation_id: str, uid: str | None) -> Conversation:
        conv = s.stores.conversations.get(conversation_id)
        if conv is None or conv.user_id != uid:
            raise HTTPException(404, f"conversation {conversation_id} not found")
        return conv

    @app.get("/conversations/{conversation_id}", response_model=ConversationView)
    def get_conversation(conversation_id: str, uid: User) -> ConversationView:
        conv = conversation_for(conversation_id, uid)
        runs = s.stores.runs.list(conversation_id=conversation_id, limit=200)
        summaries = [
            RunSummary(
                run_id=r.run_id, question=r.question, status=r.status, outcome_status=r.outcome_status,
                cost_usd=r.cost_usd, parent_run_id=r.parent_run_id, created_at=r.created_at,
            )
            for r in sorted(runs, key=lambda r: r.created_at)
        ]
        return ConversationView(conversation=conv, runs=summaries)

    @app.delete("/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str, uid: User) -> None:
        conversation_for(conversation_id, uid)
        s.stores.conversations.delete(conversation_id)

    @app.get("/profile", response_model=ProfileView)
    def get_profile(uid: RequiredUser) -> ProfileView:
        return ProfileView(user_id=uid, profile=s.stores.profiles.get(uid) or UserProfile())

    @app.put("/profile", response_model=ProfileView)
    def put_profile(profile: UserProfile, uid: RequiredUser) -> ProfileView:
        s.stores.profiles.put(uid, profile)
        return ProfileView(user_id=uid, profile=profile)

    # --- internal: worker and reaper ---------------------------------------------

    @app.post("/internal/runs/{run_id}/advance")
    def advance_run(run_id: str, request: Request) -> dict[str, Any]:
        check_secret(request, s.settings.worker_secret)
        result = s.worker.advance(run_id)
        if result.needs_more:
            s.executor.submit(run_id)  # the next slice, in a fresh invocation
        return {"run_id": run_id, "state": result.state, "steps": result.steps, "seconds": result.seconds}

    @app.api_route("/internal/cron/reap", methods=["GET", "POST"])
    def reap(request: Request) -> dict[str, Any]:
        check_secret(request, s.settings.cron_secret, s.settings.worker_secret)
        out = s.worker.reap()
        for run_id in out.dispatched:
            s.executor.submit(run_id)
        return {"dispatched": out.dispatched, "retried": out.retried}

    @app.exception_handler(ValidationError)
    def validation_error(_: Request, exc: ValidationError) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"detail": exc.errors(include_url=False)})

    return app


def _new_id() -> str:
    import uuid

    return uuid.uuid4().hex


def _title(question: str) -> str:
    q = " ".join(question.split())
    return q if len(q) <= 80 else q[:79] + "…"

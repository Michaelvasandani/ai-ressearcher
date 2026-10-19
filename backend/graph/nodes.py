"""Graph nodes. Each node runs one agent (or one supervisor decision), meters
its spend against the run budget, and appends exactly one trace step.

Nodes never raise: an agent failure becomes a `RunFailure` (fatal) or a
`Degradation` (the run carries on with less), and the routing functions in
builder.py send the run to `finalize`.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_args

from backend.agents.intake import IntakeAgent
from backend.agents.planner import PlannerAgent
from backend.agents.researcher import MAX_SEARCHES as RESEARCHER_MAX_SEARCHES
from backend.agents.researcher import ResearcherAgent
from backend.agents.writer import WriterAgent
from backend.graph.budget import (
    BudgetExceeded,
    BudgetKind,
    MeteredLLM,
    MeteredSearch,
    Spend,
    affordable_count,
    remaining,
    research_pool,
)
from backend.graph.routing import ModelRouter
from backend.graph.state import (
    MAX_SEND_BACKS,
    AnswerReview,
    Degradation,
    FailureReason,
    PassRecord,
    ResearchRecord,
    ResearchTask,
    RunFailure,
    RunOutcome,
    RunState,
    all_findings,
)
from backend.graph.supervisor import provenance_gate, review_answer, review_research
from backend.graph.trace import StepTimer
from backend.llm import LLMClient
from backend.schemas import DroppedRecord, ResearchResult, SearchStatus
from backend.tools.search import SearchTool

log = logging.getLogger(__name__)

BUDGET_KINDS: frozenset[str] = frozenset(get_args(BudgetKind))
RETRYABLE_SEARCH_STATES = frozenset({"rate_limited", "timed_out", "error"})


@dataclass
class GraphDeps:
    """Everything nodes need that is not run state. Not checkpointed."""

    llm: LLMClient
    search_tool: SearchTool
    router: ModelRouter = field(default_factory=ModelRouter)
    clock: Callable[[], float] = time.time  # epoch seconds; injectable for wall-clock tests


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:500]


def _clip(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def guarded(node: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Turn an unexpected exception in a sequential node into a structured failure."""

    def deco(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(self: "Nodes", state: RunState) -> dict[str, Any]:
            try:
                return fn(self, state)
            except Exception as e:  # a bug must end the run cleanly, not crash it
                log.exception("node %s crashed", node)
                timer = StepTimer(state.get("run_id", "?"), node)
                return {
                    "failure": RunFailure(reason="internal_error", node=node, message=_err(e)),
                    "trace": [
                        timer.step(input_summary="", output_summary="node crashed", status="failed", error=_err(e))
                    ],
                }

        return wrapper

    return deco


class Nodes:
    def __init__(self, deps: GraphDeps):
        self.deps = deps
        self.router = deps.router

    # --- helpers ------------------------------------------------------------

    def _now(self) -> float:
        return self.deps.clock()

    def _metered(self, state: RunState) -> MeteredLLM:
        allowance = remaining(state["budget"], state.get("spend") or Spend(), state["started_at"])
        return MeteredLLM(self.deps.llm, allowance, self.deps.clock)

    @staticmethod
    def _failure(reason: FailureReason, node: str, e: BaseException, llm: MeteredLLM) -> RunFailure:
        kind = e.kind if isinstance(e, BudgetExceeded) else llm.exhausted
        if kind is not None:
            return RunFailure(reason="budget_exhausted", node=node, message=_err(e), budget_kind=kind)
        return RunFailure(reason=reason, node=node, message=_err(e))

    # --- intake / referral --------------------------------------------------

    @guarded("intake")
    def intake(self, state: RunState) -> dict[str, Any]:
        timer = StepTimer(state["run_id"], "intake")
        route = self.router.route("intake")
        llm = self._metered(state)
        summary = f"question: {_clip(state['question'], 300)}"
        try:
            intake = IntakeAgent(llm, model=route.model).run(state["question"], state.get("profile"))
        except Exception as e:
            failure = self._failure("intake_failed", "intake", e, llm)
            step = timer.step(
                agent="intake", model=route.model, input_summary=summary, output_summary="intake failed",
                llm_calls=llm.records, status="failed", error=failure.message, decision="finalize",
            )
            return {"failure": failure, "spend": llm.spend, "trace": [step]}
        step = timer.step(
            agent="intake",
            model=route.model if llm.records else None,
            route_reason=route.reason if llm.records else "rule screen, no model call",
            input_summary=summary,
            output_summary=f"{intake.scope_category} ({intake.classified_by}): {_clip(intake.scoped_question)}",
            output=intake,
            llm_calls=llm.records,
            decision="planner" if intake.in_scope else "referral",
        )
        return {"intake": intake, "spend": llm.spend, "trace": [step]}

    @guarded("referral")
    def referral(self, state: RunState) -> dict[str, Any]:
        intake = state["intake"]
        timer = StepTimer(state["run_id"], "referral")
        outcome = RunOutcome(
            status="referred",
            message=f"Out of scope ({intake.scope_category}): {intake.scope_reason}",
            referral_message=intake.referral_message,
        )
        step = timer.step(
            agent="supervisor",
            input_summary=f"scope: {intake.scope_category} via {intake.classified_by}",
            output_summary="fixed referral response; planner not run",
            output=outcome,
            decision="end",
        )
        return {"outcome": outcome, "trace": [step]}

    # --- planning -----------------------------------------------------------

    @guarded("planner")
    def planner(self, state: RunState) -> dict[str, Any]:
        timer = StepTimer(state["run_id"], "planner")
        intake = state["intake"]
        route = self.router.route("planner")
        llm = self._metered(state)
        summary = f"scoped question: {_clip(intake.scoped_question, 300)}"
        try:
            plan = PlannerAgent(llm, model=route.model).run(intake)
        except Exception as e:
            failure = self._failure("planner_failed", "planner", e, llm)
            step = timer.step(
                agent="planner", model=route.model, route_reason=route.reason, input_summary=summary,
                output_summary="planning failed", llm_calls=llm.records, status="failed",
                error=failure.message, decision="finalize",
            )
            return {"failure": failure, "spend": llm.spend, "trace": [step]}

        cap = state["budget"].max_sub_questions
        selected, dropped = plan.sub_questions[:cap], plan.sub_questions[cap:]
        degradations = []
        if dropped:
            degradations.append(
                Degradation(
                    kind="max_sub_questions",
                    node="planner",
                    detail=f"researching {len(selected)} of {len(plan.sub_questions)} sub-questions; "
                    f"dropped {', '.join(sq.id for sq in dropped)}",
                )
            )
        step = timer.step(
            agent="planner",
            model=route.model,
            route_reason=route.reason,
            input_summary=summary,
            output_summary=f"{len(plan.sub_questions)} sub-questions"
            + (f", {len(dropped)} dropped by max_sub_questions={cap}" if dropped else ""),
            output=plan,
            llm_calls=llm.records,
            status="degraded" if dropped else "ok",
            decision="dispatch",
        )
        return {
            "plan": plan,
            "sub_questions": selected,
            "degradations": degradations,
            "spend": llm.spend,
            "trace": [step],
        }

    # --- research -----------------------------------------------------------

    @guarded("dispatch")
    def dispatch(self, state: RunState) -> dict[str, Any]:
        """Supervisor: split the research budget between one researcher per sub-question."""
        timer = StepTimer(state["run_id"], "dispatch")
        budget = state["budget"]
        sqs = state["sub_questions"]
        pool, block = research_pool(budget, state.get("spend") or Spend(), state["started_at"], self._now())
        if pool is None:
            deg = Degradation(kind=block, node="dispatch", detail="no research budget left after planning")  # type: ignore[arg-type]
            step = timer.step(
                agent="supervisor", input_summary=f"{len(sqs)} sub-questions", status="degraded",
                output_summary=f"research skipped: {block} exhausted", decision="review",
            )
            return {"pending_tasks": [], "degradations": [deg], "trace": [step]}

        route = self.router.route("researcher")
        funded, kind = affordable_count(pool, route.model, len(sqs))
        degradations = []
        if kind is not None:
            degradations = [
                Degradation(
                    kind=kind, node="dispatch", sub_question_id=sq.id,
                    detail=f"not researched: the research budget funds {funded} of {len(sqs)} researchers",
                )
                for sq in sqs[funded:]
            ]
        if funded == 0:
            step = timer.step(
                agent="supervisor", input_summary=f"{len(sqs)} sub-questions", status="degraded",
                output_summary=f"research skipped: {kind} cannot fund one researcher", decision="review",
            )
            return {"pending_tasks": [], "degradations": degradations, "trace": [step]}
        share = pool.split(funded)
        max_searches = min(RESEARCHER_MAX_SEARCHES, budget.max_searches_per_sub_question)
        profile = state["intake"].profile
        tasks = [
            ResearchTask(
                run_id=state["run_id"], sub_question=sq, pass_no=1, route=route,
                max_searches=max_searches, allowance=share, profile=profile,
            )
            for sq in sqs[:funded]
        ]
        step = timer.step(
            agent="supervisor",
            input_summary=f"{len(sqs)} sub-questions; research pool ${pool.usd:.4f} / {pool.tokens} tokens",
            output_summary=f"fan out {len(tasks)} researchers on {route.model}, "
            f"each ${share.usd:.4f} / {share.tokens} tokens / {max_searches} searches",
            output={"tasks": [t.sub_question.id for t in tasks], "allowance": share.model_dump()},
            status="degraded" if degradations else "ok",
            decision=f"researcher x{len(tasks)}",
        )
        return {"pending_tasks": tasks, "degradations": degradations, "trace": [step]}

    def researcher(self, task: Any) -> dict[str, Any]:
        """One researcher pass on one sub-question. Runs in parallel with its siblings, so it
        only writes reducer channels and never sets `failure`."""
        task = task if isinstance(task, ResearchTask) else ResearchTask.model_validate(task)
        sq = task.sub_question
        timer = StepTimer(task.run_id, "researcher")
        llm = MeteredLLM(self.deps.llm, task.allowance, self.deps.clock)
        search = MeteredSearch(self.deps.search_tool, task.max_searches, task.allowance.deadline, self.deps.clock)
        error: str | None = None
        try:
            agent = ResearcherAgent(llm, search, model=task.route.model, max_searches=task.max_searches)
            result = agent.run(sq, task.profile)
        except Exception as e:  # agent bug: report as a failed search status
            log.exception("researcher crashed on %s", sq.id)
            error = _err(e)
            result = ResearchResult(
                sub_question_id=sq.id,
                findings=[],
                status=SearchStatus(sub_question_id=sq.id, state="error", queries=sq.search_terms, errors=[error]),
            )

        # Provenance gate: a finding that cannot be traced to its source never enters state.
        kept, rejected = provenance_gate(result.findings)
        if rejected:
            status = result.status.model_copy(deep=True)
            source_ids = {f.id: f.source_id for f in result.findings}
            status.dropped += [
                DroppedRecord(source_id=source_ids[fid], reason="no_provenance", detail=f"finding {fid}: {why}")
                for fid, why in rejected.items()
            ]
            status.findings_kept = len(kept)
            result = ResearchResult(sub_question_id=sq.id, findings=kept, status=status)

        kind: BudgetKind | None = llm.exhausted or search.exhausted
        executed = [c for c in search.calls if c.state != "refused"]
        record = ResearchRecord(
            sub_question_id=sq.id,
            findings=result.findings,
            passes=[
                PassRecord(
                    pass_no=task.pass_no,
                    model=task.route.model,
                    escalated=task.route.escalated,
                    queries=[c.input for c in executed],
                    failed_queries=[c.input for c in executed if c.state in RETRYABLE_SEARCH_STATES],
                    status=result.status,
                    finding_ids=[f.id for f in result.findings],
                    budget_exhausted=kind,
                )
            ],
            searches_used=len(executed),
            queries_run=[c.input for c in executed],
        )
        degradations = []
        if rejected:
            degradations.append(
                Degradation(
                    kind="provenance_rejected", node="researcher", sub_question_id=sq.id,
                    detail=f"{len(rejected)} finding(s) without provenance rejected: {', '.join(sorted(rejected))}"[:500],
                )
            )
        if kind is not None:
            degradations.append(
                Degradation(
                    kind=kind, node="researcher", sub_question_id=sq.id,
                    detail=f"pass {task.pass_no} stopped early: {kind} reached",
                )
            )
        state_ = result.status.state
        step = timer.step(
            agent="researcher",
            sub_question_id=sq.id,
            attempt=task.pass_no,
            model=task.route.model,
            escalated=task.route.escalated,
            route_reason=task.route.reason,
            input_summary=f"{sq.id}: {_clip(sq.question)} | queries: {'; '.join(sq.search_terms[: task.max_searches])}",
            output_summary=f"{state_}: {len(result.findings)} findings from {result.status.records_seen} records "
            f"({len(executed)} searches)",
            output=result,
            llm_calls=llm.records,
            tool_calls=search.calls,
            status="ok" if state_ == "ok" else ("degraded" if result.findings else "failed"),
            error=error or ("; ".join(result.status.errors)[:500] or None),
            decision="review",
        )
        return {
            "research": {sq.id: record},
            "spend": Spend.of(llm.records, searches=len(executed)),
            "degradations": degradations,
            "trace": [step],
        }

    # --- supervisor review --------------------------------------------------

    @guarded("review")
    def review(self, state: RunState) -> dict[str, Any]:
        timer = StepTimer(state["run_id"], "review")
        budget = state["budget"]
        research = state.get("research") or {}
        sqs = state["sub_questions"]
        send_backs_used = state.get("send_backs_used", 0)
        pool, block = research_pool(budget, state.get("spend") or Spend(), state["started_at"], self._now())
        escalation_model = self.router.route("researcher", escalate=True).model
        if pool is not None and block is None:
            _, block = affordable_count(pool, escalation_model, 1)  # can it fund even one escalated pass?
        pass_no = len(state.get("reviews") or []) + 1
        report = review_research(
            sqs,
            research,
            state["intake"].profile,
            pass_no=pass_no,
            send_backs_used=send_backs_used,
            max_searches_per_sub_question=budget.max_searches_per_sub_question,
            budget_block=block,
        )
        update: dict[str, Any] = {"reviews": [report], "pending_tasks": []}

        if report.decision == "send_back":
            assert pool is not None
            funded, kind = affordable_count(pool, escalation_model, len(report.send_back_ids))
            share = pool.split(funded)
            by_id = {sq.id: sq for sq in sqs}
            tasks = []
            degradations = []
            for r in report.sub_questions:
                if not r.another_pass:
                    continue
                if len(tasks) == funded:
                    degradations.append(
                        Degradation(
                            kind=kind or "max_usd", node="review", sub_question_id=r.sub_question_id,
                            detail=f"send-back not funded: budget covers {funded} of {len(report.send_back_ids)} passes",
                        )
                    )
                    continue
                rec = research.get(r.sub_question_id)
                route = self.router.route(
                    "researcher",
                    escalate=r.escalate,
                    reason=f"re-research after failed review ({', '.join(r.issues)})"
                    if r.escalate
                    else f"retry after search failure ({', '.join(r.issues)})",
                )
                tasks.append(
                    ResearchTask(
                        run_id=state["run_id"],
                        sub_question=by_id[r.sub_question_id].model_copy(update={"search_terms": r.refined_queries}),
                        pass_no=(len(rec.passes) if rec else 0) + 1,
                        route=route,
                        max_searches=len(r.refined_queries),
                        allowance=share,
                        profile=state["intake"].profile,
                        reason=r.another_pass_reason,
                    )
                )
            update |= {"pending_tasks": tasks, "send_backs_used": send_backs_used + 1, "degradations": degradations}
        else:
            update["degradations"] = self._review_degradations(report)
            if report.decision == "fail":
                update["failure"] = self._research_failure(state, report.reason, block)

        step = timer.step(
            agent="supervisor",
            attempt=pass_no,
            input_summary=f"pass {pass_no}: {sum(len(r.findings) for r in research.values())} findings over "
            f"{len(research)} of {len(sqs)} sub-questions; send-backs used {send_backs_used}/{MAX_SEND_BACKS}",
            output_summary=report.reason,
            output=report,
            status={"proceed": "ok", "send_back": "degraded", "fail": "failed"}[report.decision],
            decision={"proceed": "writer", "send_back": f"researcher x{len(report.send_back_ids)}", "fail": "finalize"}[
                report.decision
            ],
        )
        update["trace"] = [step]
        return update

    @staticmethod
    def _review_degradations(report: Any) -> list[Degradation]:
        out: list[Degradation] = []
        for r in report.sub_questions:
            if r.blocked_by in BUDGET_KINDS:
                out.append(
                    Degradation(kind=r.blocked_by, node="review", sub_question_id=r.sub_question_id, detail=r.another_pass_reason)
                )
            elif r.blocked_by == "send_back_used" and report.pass_no == 1:
                out.append(
                    Degradation(
                        kind="send_back_unavailable", node="review", sub_question_id=r.sub_question_id,
                        detail=r.another_pass_reason,
                    )
                )
            if not r.answered:
                out.append(
                    Degradation(
                        kind="sub_question_unanswered", node="review", sub_question_id=r.sub_question_id,
                        detail=f"{', '.join(r.issues) or 'unanswered'}",
                    )
                )
            elif "abstract_only" in r.issues:
                out.append(
                    Degradation(
                        kind="abstract_only_evidence", node="review", sub_question_id=r.sub_question_id,
                        detail="no open-access full text: findings rest on abstracts",
                    )
                )
            if r.answered and "population_mismatch" in r.issues:
                out.append(
                    Degradation(
                        kind="population_mismatch", node="review", sub_question_id=r.sub_question_id,
                        detail="; ".join(sorted(set(r.population_mismatches.values()))),
                    )
                )
        return out

    @staticmethod
    def _research_failure(state: RunState, reason: str, block: BudgetKind | None) -> RunFailure:
        kinds = [d.kind for d in state.get("degradations") or [] if d.kind in BUDGET_KINDS and d.kind != "max_sub_questions"]
        kind = block or (kinds[0] if kinds else None)
        if kind is not None:
            return RunFailure(reason="budget_exhausted", node="review", message=reason, budget_kind=kind)  # type: ignore[arg-type]
        return RunFailure(reason="insufficient_evidence", node="review", message=reason)

    # --- writing ------------------------------------------------------------

    @guarded("writer")
    def writer(self, state: RunState) -> dict[str, Any]:
        timer = StepTimer(state["run_id"], "writer")
        attempt = state.get("writer_attempts", 0) + 1
        route = state.get("writer_route") or self.router.route("writer")
        llm = self._metered(state)
        findings = all_findings(state)
        answer, error = None, None
        try:
            answer = WriterAgent(llm, model=route.model).run(
                state["intake"],
                findings,
                positions=state.get("position_map"),
                verdict=state.get("verdict"),
                feedback=state.get("writer_feedback"),
            )
        except Exception as e:  # WriterError, LLMError, BudgetExceeded: the supervisor decides
            error = _err(e)
        step = timer.step(
            agent="writer",
            attempt=attempt,
            model=route.model,
            escalated=route.escalated,
            route_reason=route.reason,
            input_summary=f"{len(findings)} findings" + (" + review feedback" if state.get("writer_feedback") else ""),
            output_summary=f"{answer.status}, confidence {answer.confidence}: {_clip(answer.recommendation)}"
            if answer
            else f"no answer: {error}",
            output=answer,
            llm_calls=llm.records,
            status="ok" if answer else "failed",
            error=error,
            decision="check_answer",
        )
        return {
            "draft_answer": answer,
            "writer_error": error,
            "writer_budget_kind": llm.exhausted,
            "writer_attempts": attempt,
            "spend": llm.spend,
            "trace": [step],
        }

    @guarded("check_answer")
    def check_answer(self, state: RunState) -> dict[str, Any]:
        """Supervisor: accept the draft, send it back once for an escalated rewrite, or fail cleanly."""
        timer = StepTimer(state["run_id"], "check_answer")
        attempt = state.get("writer_attempts", 1)
        draft = state.get("draft_answer")
        problems = review_answer(draft, all_findings(state), state.get("writer_error"))
        send_backs_used = state.get("send_backs_used", 0)
        update: dict[str, Any] = {}

        if not problems:
            review = AnswerReview(attempt=attempt, decision="accept", reason="answer passed review")
            update["answer"] = draft
        else:
            kind = state.get("writer_budget_kind")
            left = remaining(state["budget"], state.get("spend") or Spend(), state["started_at"])
            if kind is None and self._now() >= left.deadline:
                kind = "wall_clock"
            if kind is not None:
                review = AnswerReview(attempt=attempt, decision="fail", problems=problems, reason=f"{kind} exhausted")
                update["failure"] = RunFailure(
                    reason="budget_exhausted", node="writer", message="; ".join(problems), budget_kind=kind
                )
            elif send_backs_used >= MAX_SEND_BACKS:
                review = AnswerReview(
                    attempt=attempt, decision="fail", problems=problems, reason="send-back already used"
                )
                update["failure"] = RunFailure(reason="writer_failed", node="writer", message="; ".join(problems))
            else:
                review = AnswerReview(
                    attempt=attempt, decision="rewrite", problems=problems, reason="rewrite with the escalation model"
                )
                update |= {
                    "writer_route": self.router.route(
                        "writer", escalate=True, reason=f"rewrite after failed answer review: {_clip(problems[0])}"
                    ),
                    "writer_feedback": "\n".join(f"- {p}" for p in problems),
                    "send_backs_used": send_backs_used + 1,
                    "draft_answer": None,
                }
        update["answer_reviews"] = [review]
        update["trace"] = [
            timer.step(
                agent="supervisor",
                attempt=attempt,
                input_summary="draft answer" if draft else f"writer error: {state.get('writer_error')}",
                output_summary=review.reason + (f": {'; '.join(problems)}" if problems else ""),
                output=review,
                status={"accept": "ok", "rewrite": "degraded", "fail": "failed"}[review.decision],
                decision={"accept": "finalize", "rewrite": "writer", "fail": "finalize"}[review.decision],
            )
        ]
        return update

    # --- end ----------------------------------------------------------------

    def finalize(self, state: RunState) -> dict[str, Any]:
        timer = StepTimer(state.get("run_id", "?"), "finalize")
        failure = state.get("failure")
        answer = state.get("answer")
        degradations = list(state.get("degradations") or [])
        if failure is not None:
            outcome = RunOutcome(
                status="failed",
                message=f"{failure.reason} at {failure.node}: {failure.message}",
                failure=failure,
                degradations=degradations,
            )
        elif answer is not None:
            outcome = RunOutcome(
                status="degraded" if degradations else "answered",
                message="answered with caveats: " + "; ".join(sorted({d.kind for d in degradations}))
                if degradations
                else "answered",
                degradations=degradations,
            )
        else:
            failure = RunFailure(reason="internal_error", node="finalize", message="run ended without an answer")
            outcome = RunOutcome(status="failed", message=failure.message, failure=failure, degradations=degradations)
        spend = state.get("spend") or Spend()
        step = timer.step(
            agent="supervisor",
            input_summary=f"spent ${spend.cost_usd:.4f}, {spend.total_tokens} tokens, {spend.llm_calls} calls, "
            f"{spend.searches} searches",
            output_summary=outcome.message[:300],
            output=outcome,
            status={"answered": "ok", "degraded": "degraded"}.get(outcome.status, "failed"),
            decision="end",
        )
        return {"outcome": outcome, "trace": [step]}

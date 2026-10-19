"""Synthesis nodes: personalization, position mapping, the debate and the judge.

    review --proceed--> personalize -> position_mapper --consensus / no debate--> judge -> writer
                                             |                                     ^
                                             +==Send x N==> advocate (round 1) -> debate_review
                                                            advocate (round 2) <==Send x N==+
                                                                  |
                                                                  +-> debate_review -> judge

Same conventions as nodes.py: every node appends exactly one trace step,
meters its spend, and never raises. A synthesis problem degrades the answer
instead of failing the run: no position map -> the writer works from findings;
no verdict -> the writer works from the position map.

Budgets are enforced here, before each stage, with conservative cost and time
estimates per call (`Cost`). Every stage keeps a reserve for what must still
run after it (the judge and the writer). The debate runs only if all of it
(openings and the rebuttal round) fits; otherwise it is skipped and recorded
as a `debate_skipped` degradation. Each advocate gets its own allowance, so a
verbose advocate cannot spend the judge's or the writer's share.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.agents import advocate as advocate_mod
from backend.agents import judge as judge_mod
from backend.agents import mapper as mapper_mod
from backend.agents import writer as writer_mod
from backend.agents.advocate import AdvocateAgent
from backend.agents.evidence import DOWNWEIGHT, render_findings
from backend.agents.judge import JudgeAgent
from backend.agents.mapper import MapperAgent
from backend.config import AGENT_MODELS, price_for
from backend.graph.budget import (
    CHARS_PER_TOKEN,
    MIN_OUTPUT_TOKENS,
    SCHEMA_OVERHEAD_TOKENS,
    Allowance,
    BudgetKind,
    MeteredLLM,
    Spend,
    remaining,
)
from backend.graph.nodes import GraphDeps, _clip, _err, guarded
from backend.graph.state import DEBATE_ROUNDS, AdvocateTask, Degradation, RunState, all_findings
from backend.graph.supervisor import downweight_reasons
from backend.graph.trace import StepTimer, dump
from backend.schemas import DebateTurn, Finding, PositionMap

log = logging.getLogger(__name__)

# Expected output tokens and seconds per call, used to decide what the budget can fund.
EXPECTED_OUTPUT_TOKENS = {"position_mapper": 2_000, "advocate": 1_200, "judge": 1_500, "writer": 2_500}
STAGE_SECONDS = {"position_mapper": 40.0, "advocate": 30.0, "judge": 40.0}
ARGUMENT_CHARS = 1_500  # one rendered argument, for the judge's and the rebuttals' prompts
SYSTEM_PROMPTS = {
    "position_mapper": mapper_mod.SYSTEM_PROMPT,
    "advocate": advocate_mod.SYSTEM_PROMPT,
    "judge": judge_mod.SYSTEM_PROMPT,
    "writer": writer_mod.SYSTEM_PROMPT,
}


@dataclass(frozen=True)
class Cost:
    """Estimated dollars, tokens and seconds for some calls."""

    usd: float = 0.0
    tokens: int = 0
    seconds: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.usd + other.usd, self.tokens + other.tokens, self.seconds + other.seconds)

    def times(self, n: int, *, parallel: bool = False) -> "Cost":
        """n calls; parallel calls cost n times the money but take the time of one."""
        return Cost(self.usd * n, self.tokens * n, self.seconds if parallel else self.seconds * n)


def estimate(agent: str, model: str, prompt_chars: int, seconds: float = 0.0) -> Cost:
    """Cost of one call: input from the prompt length, output from the agent's typical answer."""
    input_tokens = int((len(SYSTEM_PROMPTS[agent]) + prompt_chars) / CHARS_PER_TOKEN) + SCHEMA_OVERHEAD_TOKENS
    output_tokens = max(EXPECTED_OUTPUT_TOKENS[agent], MIN_OUTPUT_TOKENS)
    output_tokens = min(output_tokens, AGENT_MODELS[agent].max_tokens)
    price = price_for(model)
    usd = (input_tokens * price.input + output_tokens * price.output) / 1_000_000
    return Cost(usd, input_tokens + output_tokens, seconds)


def fund(left: Allowance, need: Cost, reserve: Cost, now: float) -> tuple[Allowance | None, BudgetKind | None]:
    """What a stage may spend, keeping `reserve` for later stages; None + the binding budget if `need` does not fit."""
    usd, tokens, deadline = left.usd - reserve.usd, left.tokens - reserve.tokens, left.deadline - reserve.seconds
    if now + need.seconds > deadline:
        return None, "wall_clock"
    if usd < need.usd:
        return None, "max_usd"
    if tokens < need.tokens:
        return None, "max_tokens"
    return Allowance(usd=usd, tokens=tokens, deadline=deadline), None


class SynthesisNodes:
    def __init__(self, deps: GraphDeps):
        self.deps = deps
        self.router = deps.router

    # --- helpers ------------------------------------------------------------

    def _now(self) -> float:
        return self.deps.clock()

    @staticmethod
    def _left(state: RunState, extra: Spend | None = None) -> Allowance:
        spend = (state.get("spend") or Spend()) + (extra or Spend())
        return remaining(state["budget"], spend, state["started_at"])

    def _costs(self, state: RunState, findings: list[Finding], positions: int = 2) -> dict[str, Cost]:
        block = len(render_findings(findings, state.get("downweighted") or {})) + 1_000  # + question, profile
        r = self.router
        return {
            "position_mapper": estimate(
                "position_mapper", r.route("position_mapper").model, block, STAGE_SECONDS["position_mapper"]
            ),
            "advocate": estimate(
                "advocate", r.route("advocate").model, block + positions * ARGUMENT_CHARS, STAGE_SECONDS["advocate"]
            ),
            "judge": estimate(
                "judge", r.route("judge").model, block + 2 * positions * ARGUMENT_CHARS, STAGE_SECONDS["judge"]
            ),
            "writer": estimate("writer", r.route("writer").model, block + 2_000, state["budget"].writer_reserve_s),
        }

    # --- personalization ------------------------------------------------------

    @guarded("personalize")
    def personalize(self, state: RunState) -> dict[str, Any]:
        """Supervisor: down-weight findings whose study population does not match the user."""
        timer = StepTimer(state["run_id"], "personalize")
        findings = all_findings(state)
        profile = state["intake"].profile
        down = {f.id: "; ".join(reasons) for f in findings if (reasons := downweight_reasons(profile, f))}
        profile_desc = profile.model_dump_json(exclude_none=True) if not profile.is_empty() else "empty profile"
        step = timer.step(
            agent="supervisor",
            input_summary=f"{len(findings)} findings; profile: {_clip(profile_desc, 200)}",
            output_summary=f"{len(down)} of {len(findings)} findings down-weighted"
            + (f": {_clip('; '.join(f'{k} ({v})' for k, v in down.items()), 300)}" if down else ""),
            output={"downweighted": down, "weight": DOWNWEIGHT},
            decision="position_mapper",
        )
        return {"downweighted": down, "trace": [step]}

    # --- position mapping -------------------------------------------------------

    @guarded("position_mapper")
    def position_mapper(self, state: RunState) -> dict[str, Any]:
        timer = StepTimer(state["run_id"], "position_mapper")
        findings = all_findings(state)
        down = state.get("downweighted") or {}
        route = self.router.route("position_mapper")
        costs = self._costs(state, findings)
        left = self._left(state)
        summary = f"{len(findings)} findings, {len(down)} down-weighted"

        allowance, kind = fund(left, costs["position_mapper"], costs["judge"] + costs["writer"], self._now())
        if allowance is None:  # map anyway if the judge is what does not fit; the judge will skip itself
            allowance, kind = fund(left, costs["position_mapper"], costs["writer"], self._now())
        if allowance is None:
            deg = Degradation(
                kind="mapping_skipped", node="position_mapper",
                detail=f"{kind}: cannot fund position mapping and keep the writer's reserve",
            )
            step = timer.step(
                agent="position_mapper", input_summary=summary, status="skipped",
                output_summary=f"skipped: {kind} cannot fund position mapping", decision="writer",
            )
            return {"position_map": None, "pending_advocates": [], "degradations": [deg], "trace": [step]}

        llm = MeteredLLM(self.deps.llm, allowance, self.deps.clock)
        try:
            pm = MapperAgent(llm, model=route.model).run(state["intake"], findings, down)
        except Exception as e:  # MapperError, LLMError, BudgetExceeded: degrade, the writer maps itself
            error = _err(e)
            deg = Degradation(kind="mapping_skipped", node="position_mapper", detail=f"mapping failed: {error}"[:500])
            step = timer.step(
                agent="position_mapper", model=route.model, route_reason=route.reason, input_summary=summary,
                output_summary="mapping failed; the writer works from findings", llm_calls=llm.records,
                status="failed", error=error, decision="writer",
            )
            return {"position_map": None, "pending_advocates": [], "degradations": [deg], "spend": llm.spend, "trace": [step]}

        tasks, degradations, debate_note = self._plan_debate(state, pm, findings, extra=llm.spend)
        decision = f"advocate x{len(tasks)}" if tasks else "judge"
        step = timer.step(
            agent="position_mapper",
            model=route.model,
            route_reason=route.reason,
            input_summary=summary,
            output_summary=f"{pm.status} (proposed {pm.proposed_status}): {len(pm.positions)} position(s), "
            f"{len(pm.set_aside)} set aside; {debate_note}",
            output={"position_map": dump(pm), "debate": debate_note},
            llm_calls=llm.records,
            status="degraded" if degradations else "ok",
            decision=decision,
        )
        return {
            "position_map": pm,
            "pending_advocates": tasks,
            "degradations": degradations,
            "spend": llm.spend,
            "trace": [step],
        }

    def _plan_debate(
        self, state: RunState, pm: PositionMap, findings: list[Finding], extra: Spend
    ) -> tuple[list[AdvocateTask], list[Degradation], str]:
        """Opening-round tasks if the debate is warranted and the whole debate is affordable."""
        if pm.status == "consensus" or len(pm.positions) < 2:
            return [], [], "consensus: debate skipped"
        n = len(pm.positions)
        costs = self._costs(state, findings, positions=n)
        # Openings and the one rebuttal round: 2n advocate calls, one round after the other.
        debate = costs["advocate"].times(n, parallel=True).times(DEBATE_ROUNDS)
        pool, kind = fund(self._left(state, extra), debate, costs["judge"] + costs["writer"], self._now())
        if pool is None:
            deg = Degradation(
                kind="debate_skipped", node="position_mapper",
                detail=f"{kind}: the budget cannot fund {n} advocates x {DEBATE_ROUNDS} rounds and keep the judge "
                "and writer reserve; the judge weighs the positions on their findings",
            )
            return [], [deg], f"debate skipped ({kind})"
        share = Allowance(
            usd=pool.usd / (n * DEBATE_ROUNDS),
            tokens=pool.tokens // (n * DEBATE_ROUNDS),
            deadline=pool.deadline - costs["advocate"].seconds,  # leave time for the rebuttal round
        )
        route = self.router.route("advocate")
        tasks = [
            AdvocateTask(
                run_id=state["run_id"], round=1, position_no=i, position=p, intake=state["intake"],
                findings=findings, downweighted=state.get("downweighted") or {}, route=route, allowance=share,
            )
            for i, p in enumerate(pm.positions, 1)
        ]
        return tasks, [], f"debate: {n} advocates, {DEBATE_ROUNDS} rounds"

    # --- debate ---------------------------------------------------------------

    def advocate(self, task: Any) -> dict[str, Any]:
        """One advocate, one round. Runs in parallel with the other advocates, so it only writes
        reducer channels and never sets `failure`."""
        task = task if isinstance(task, AdvocateTask) else AdvocateTask.model_validate(task)
        timer = StepTimer(task.run_id, "advocate")
        llm = MeteredLLM(self.deps.llm, task.allowance, self.deps.clock)
        kind = "opening" if task.round == 1 else "rebuttal"
        try:
            turn = AdvocateAgent(llm, model=task.route.model).run(
                task.intake, task.position_no, task.position, task.findings, task.downweighted, task.opponents
            )
        except Exception as e:  # LLMError, BudgetExceeded, a schema or citation failure
            turn = DebateTurn(position=task.position_no, stance=task.position.stance, round=task.round, error=_err(e))
        degradations = []
        if turn.argument is None:
            degradations.append(
                Degradation(
                    kind="advocate_failed", node="advocate",
                    detail=f"P{task.position_no} {kind}: {turn.error or 'no argument'}"[:500],
                )
            )
        rejected = len(turn.rejected_claims)
        step = timer.step(
            agent="advocate",
            attempt=task.round,
            model=task.route.model,
            route_reason=task.route.reason,
            input_summary=f"P{task.position_no} {kind}: {_clip(task.position.stance)}"
            + (f" | rebutting {', '.join(f'P{t.position}' for t in task.opponents)}" if task.opponents else ""),
            output_summary=(
                f"{len(turn.argument.claims)} claims citing {len(turn.argument.cited_ids())} findings"
                if turn.argument
                else f"no argument: {turn.error}"
            )
            + (f"; {rejected} claim(s) rejected for invalid citations" if rejected else ""),
            output=turn,
            llm_calls=llm.records,
            status="failed" if turn.argument is None else ("degraded" if rejected else "ok"),
            error=turn.error,
            decision="debate_review",
        )
        return {"debate": [turn], "spend": llm.spend, "degradations": degradations, "trace": [step]}

    @guarded("debate_review")
    def debate_review(self, state: RunState) -> dict[str, Any]:
        """Supervisor: after the openings, send exactly one rebuttal round; after it, go to the judge."""
        timer = StepTimer(state["run_id"], "debate_review")
        turns = state.get("debate") or []
        rounds_done = max((t.round for t in turns), default=0)
        update: dict[str, Any] = {"debate_rounds": rounds_done, "pending_advocates": []}
        openings = sorted((t for t in turns if t.round == 1 and t.argument is not None), key=lambda t: t.position)
        tasks: list[AdvocateTask] = []
        degradations: list[Degradation] = []

        if rounds_done >= DEBATE_ROUNDS:
            note = f"rebuttal round done ({sum(1 for t in turns if t.round == 2)} rebuttals); to the judge"
        elif len(openings) < 2:
            note = f"rebuttal skipped: {len(openings)} valid opening argument(s), nothing to rebut"
            degradations.append(Degradation(kind="rebuttal_skipped", node="debate_review", detail=note))
        else:
            pm = state["position_map"]
            assert pm is not None
            findings = all_findings(state)
            costs = self._costs(state, findings, positions=len(pm.positions))
            need = costs["advocate"].times(len(openings), parallel=True)
            pool, kind = fund(self._left(state), need, costs["judge"] + costs["writer"], self._now())
            if pool is None:
                note = f"rebuttal skipped: {kind} cannot fund the rebuttal round"
                degradations.append(Degradation(kind="rebuttal_skipped", node="debate_review", detail=note))
            else:
                share = pool.split(len(openings))
                route = self.router.route("advocate")
                for t in openings:
                    tasks.append(
                        AdvocateTask(
                            run_id=state["run_id"], round=2, position_no=t.position, position=pm.positions[t.position - 1],
                            intake=state["intake"], findings=findings, downweighted=state.get("downweighted") or {},
                            opponents=[o for o in openings if o.position != t.position], route=route, allowance=share,
                        )
                    )
                note = f"openings from {', '.join(f'P{t.position}' for t in openings)}; one rebuttal round"
        update |= {"pending_advocates": tasks, "degradations": degradations}
        update["trace"] = [
            timer.step(
                agent="supervisor",
                attempt=rounds_done,
                input_summary=f"round {rounds_done} of {DEBATE_ROUNDS}: {len(turns)} turns, "
                f"{sum(1 for t in turns if t.argument is None)} failed",
                output_summary=note,
                output={"round": rounds_done, "rebuttals": [t.position_no for t in tasks]},
                status="degraded" if degradations else "ok",
                decision=f"advocate x{len(tasks)}" if tasks else "judge",
            )
        ]
        return update

    # --- judgment ---------------------------------------------------------------

    @guarded("judge")
    def judge(self, state: RunState) -> dict[str, Any]:
        timer = StepTimer(state["run_id"], "judge")
        pm = state.get("position_map")
        findings = all_findings(state)
        turns = state.get("debate") or []
        down = state.get("downweighted") or {}
        route = self.router.route("judge")
        summary = (
            f"{pm.status if pm else 'no map'}: {len(pm.positions) if pm else 0} position(s), "
            f"{len(turns)} debate turns, {len(findings)} findings ({len(down)} down-weighted)"
        )
        if pm is None:  # unreachable by routing; kept so the node is safe on its own
            step = timer.step(agent="judge", input_summary=summary, output_summary="no position map", status="skipped", decision="writer")
            return {"verdict": None, "trace": [step]}

        costs = self._costs(state, findings, positions=len(pm.positions))
        allowance, kind = fund(self._left(state), costs["judge"], costs["writer"], self._now())
        if allowance is None:
            deg = Degradation(kind="judge_skipped", node="judge", detail=f"{kind}: cannot fund the judge and keep the writer's reserve")
            step = timer.step(
                agent="judge", input_summary=summary, status="skipped",
                output_summary=f"skipped: {kind}; the writer works from the position map", decision="writer",
            )
            return {"verdict": None, "degradations": [deg], "trace": [step]}

        llm = MeteredLLM(self.deps.llm, allowance, self.deps.clock)
        try:
            verdict = JudgeAgent(llm, model=route.model).run(state["intake"], pm, findings, turns, down)
        except Exception as e:  # JudgeError, LLMError, BudgetExceeded
            error = _err(e)
            deg = Degradation(kind="judge_skipped", node="judge", detail=f"judge failed: {error}"[:500])
            step = timer.step(
                agent="judge", model=route.model, route_reason=route.reason, input_summary=summary,
                output_summary="no verdict; the writer works from the position map", llm_calls=llm.records,
                status="failed", error=error, decision="writer",
            )
            return {"verdict": None, "degradations": [deg], "spend": llm.spend, "trace": [step]}

        step = timer.step(
            agent="judge",
            model=route.model,
            route_reason=route.reason,
            input_summary=summary,
            output_summary=f"{verdict.status}, confidence {verdict.confidence} (cap {verdict.confidence_cap}): "
            f"{_clip(verdict.recommendation)}",
            output=verdict,
            llm_calls=llm.records,
            decision="writer",
        )
        return {"verdict": verdict, "spend": llm.spend, "trace": [step]}

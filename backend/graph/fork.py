"""Time-travel: fork a run from one of its steps with an override, as a new run.

    POST /research/{id}/fork?from_step=N   {"prompt_addendum": ..., "model": ..., "tool_result": [...]}

1. Find the checkpoint the step was produced from: the latest checkpoint in the
   run's LangGraph history whose trace is at most N steps long. Step N (and any
   parallel siblings in the same superstep) runs right after it.
2. Copy its state to a new thread (the fork's run id) with `update_state`,
   `as_node` = the node(s) that wrote that checkpoint, so LangGraph re-evaluates
   the same outgoing edges (including `Send` fan-outs) on the new thread. Run ids
   inside the state (and inside pending researcher / advocate tasks) are
   rewritten; the wall clock starts afresh; the plan checkpoint is not asked again.
3. Copy the original run's LLM call ledger to the fork. Requests the override
   does not change replay their recorded output for free; the overridden step,
   and everything downstream whose input changed, makes new calls.
4. The override is scoped to step N's agent (its sub-question for a researcher,
   its position for an advocate) and stored on the fork's run record, where
   `GraphDeps.override_for` finds it on every resume.

When the fork finishes, `answer_diff` compares it with the original: outcome,
status, recommendation, confidence, positions and citations.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from backend.graph.builder import RunResult
from backend.graph.overrides import LLM_AGENTS, TOOL_AGENTS, ScopedOverride, StepOverride
from backend.graph.trace import TraceStep
from backend.schemas import CoachAnswer, Position, Record


class ForkError(ValueError):
    """The fork request is invalid (unknown step, step not overridable, ...)."""


class ForkInfo(Record):
    """Stored on the fork's run record (`runs.fork`)."""

    parent_run_id: str
    from_step: int
    step_id: str
    node: str
    checkpoint_id: str | None = None  # the parent's checkpoint the fork was copied from
    scope: ScopedOverride
    inherited_steps: int  # trace steps copied from the parent (indices 0 .. from_step - 1 or fewer)
    copied_ledger_entries: int = 0


def scope_for(step: TraceStep, override: StepOverride) -> ScopedOverride:
    """Bind an override to the step's agent (and sub-question / position). Raises ForkError."""
    agent = step.agent or ""
    if agent not in LLM_AGENTS:
        raise ForkError(
            f"step {step.index} ({step.node}) is a {agent or 'supervisor'} step with no model call to override; "
            f"fork from one of: {', '.join(sorted(LLM_AGENTS))}"
        )
    if override.tool_result is not None and agent not in TOOL_AGENTS:
        raise ForkError(f"a tool_result override needs a researcher step; step {step.index} is {agent}")
    position_no = None
    if agent == "advocate":
        pid = step.position_id or ""
        if not pid.startswith("P") or not pid[1:].isdigit():
            raise ForkError(f"advocate step {step.index} does not record its position")
        position_no = int(pid[1:])
    return ScopedOverride(
        agent=agent,
        sub_question_id=step.sub_question_id if agent == "researcher" else None,
        position_no=position_no,
        override=override,
    )


def fork_point(graph: Any, config: dict[str, Any], from_step: int) -> tuple[Any, str]:
    """(checkpoint snapshot to copy, as_node) for re-running trace step `from_step`."""
    history = list(graph.get_state_history(config))  # newest first
    candidates = [
        h for h in history
        if h.next and h.values and len(h.values.get("trace") or []) <= from_step and "__start__" not in h.next
    ]
    if not candidates:
        raise ForkError(f"no checkpoint precedes step {from_step}")
    snap = max(candidates, key=lambda h: (h.metadata or {}).get("step", -1))
    parent = graph.get_state(snap.parent_config) if snap.parent_config else None
    writers = sorted({t.name for t in (parent.tasks if parent else ())})
    if len(writers) != 1:
        raise ForkError(f"cannot tell which node wrote the checkpoint before step {from_step}: {writers}")
    return snap, writers[0]


def fork_values(values: dict[str, Any], new_run_id: str, now: float) -> dict[str, Any]:
    """The parent's state at the fork point, re-addressed to the fork's run."""
    out = dict(values)
    out["run_id"] = new_run_id
    out["started_at"] = now  # a fresh wall-clock budget for the replay; spend carries over
    out["require_plan_approval"] = False  # the parent's plan decision (if any) is in the copied state
    for key in ("pending_tasks", "pending_advocates"):
        if out.get(key):
            out[key] = [t.model_copy(update={"run_id": new_run_id}) for t in out[key]]
    return out


# --- Diff ---------------------------------------------------------------------


class FieldChange(Record):
    field: str
    before: Any = None
    after: Any = None


class PositionChange(Record):
    stance_before: str | None = None
    stance_after: str | None = None
    grade_before: str | None = None
    grade_after: str | None = None
    findings_added: list[str] = Field(default_factory=list)
    findings_removed: list[str] = Field(default_factory=list)


class AnswerDiff(Record):
    original_run_id: str
    fork_run_id: str
    from_step: int
    changed: bool
    changes: list[FieldChange] = Field(default_factory=list)  # outcome, status, recommendation, confidence
    positions_added: list[Position] = Field(default_factory=list)
    positions_removed: list[Position] = Field(default_factory=list)
    positions_changed: list[PositionChange] = Field(default_factory=list)
    citations_added: list[str] = Field(default_factory=list)
    citations_removed: list[str] = Field(default_factory=list)
    cost_usd_original: float = 0.0
    cost_usd_fork: float = 0.0
    new_cost_usd: float = 0.0  # what the fork actually paid (calls not replayed from the parent's ledger)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def _match_positions(before: list[Position], after: list[Position]) -> list[tuple[Position | None, Position | None]]:
    """Pair positions by stance text, then by overlapping supporting findings (>= 0.5)."""
    pairs: list[tuple[Position | None, Position | None]] = []
    left = list(after)
    unmatched: list[Position] = []
    for p in before:
        same = next((q for q in left if _norm(q.stance) == _norm(p.stance)), None)
        if same is not None:
            left.remove(same)
            pairs.append((p, same))
        else:
            unmatched.append(p)
    for p in unmatched:
        best = max(left, key=lambda q: _jaccard(p.supporting_findings, q.supporting_findings), default=None)
        if best is not None and _jaccard(p.supporting_findings, best.supporting_findings) >= 0.5:
            left.remove(best)
            pairs.append((p, best))
        else:
            pairs.append((p, None))
    pairs.extend((None, q) for q in left)
    return pairs


def _cited(answer: CoachAnswer | None) -> list[str]:
    if answer is None:
        return []
    return list(dict.fromkeys([*answer.cited_finding_ids, *(i for p in answer.positions for i in p.supporting_findings)]))


def answer_diff(original: RunResult, fork: RunResult, from_step: int) -> AnswerDiff:
    a, b = original.answer, fork.answer
    changes: list[FieldChange] = []
    oa = original.outcome.status if original.outcome else None
    ob = fork.outcome.status if fork.outcome else None
    if oa != ob:
        changes.append(FieldChange(field="outcome", before=oa, after=ob))
    for name in ("status", "recommendation", "confidence"):
        va, vb = getattr(a, name, None), getattr(b, name, None)
        if va != vb:
            changes.append(FieldChange(field=name, before=va, after=vb))

    added: list[Position] = []
    removed: list[Position] = []
    changed: list[PositionChange] = []
    for p, q in _match_positions(a.positions if a else [], b.positions if b else []):
        if p is None and q is not None:
            added.append(q)
        elif q is None and p is not None:
            removed.append(p)
        elif p is not None and q is not None and p != q:
            changed.append(
                PositionChange(
                    stance_before=p.stance, stance_after=q.stance,
                    grade_before=p.evidence_grade, grade_after=q.evidence_grade,
                    findings_added=[i for i in q.supporting_findings if i not in p.supporting_findings],
                    findings_removed=[i for i in p.supporting_findings if i not in q.supporting_findings],
                )
            )
    ca, cb = _cited(a), _cited(b)
    new_steps = fork.trace[len([s for s in fork.trace if s.run_id != fork.run_id]):]
    new_cost = sum(c.cost_usd for s in new_steps for c in s.llm_calls if not c.replayed)
    diff = AnswerDiff(
        original_run_id=original.run_id,
        fork_run_id=fork.run_id,
        from_step=from_step,
        changed=False,
        changes=changes,
        positions_added=added,
        positions_removed=removed,
        positions_changed=changed,
        citations_added=[i for i in cb if i not in ca],
        citations_removed=[i for i in ca if i not in cb],
        cost_usd_original=original.spend.cost_usd,
        cost_usd_fork=fork.spend.cost_usd,
        new_cost_usd=round(new_cost, 6),
    )
    diff.changed = bool(changes or added or removed or changed or diff.citations_added or diff.citations_removed)
    return diff

"""Progress events for the UI: one short line per trace step, plus running totals.

The SSE stream shows these as they arrive ("Planned 5 sub-questions",
"PubMed: 23 studies → 9 kept", "Debate: round 1 · P2 opening"). The full step
is in the trace; an event only carries what a progress list needs.
"""

from __future__ import annotations

from typing import Any

from backend.graph.trace import TraceStep

SOURCE_LABELS = {
    "fixture": "Fixture library",
    "composite": "PubMed + Europe PMC",
    "live": "PubMed + Europe PMC",
    "pubmed": "PubMed",
    "europepmc": "Europe PMC",
    "override": "Overridden tool result",
    "scripted": "Fixture library",
}


def _out(step: TraceStep) -> dict[str, Any]:
    return step.output or {}


def _plural(n: int, word: str) -> str:
    if n == 1:
        return f"{n} {word}"
    return f"{n} {word[:-1]}ies" if word.endswith("y") else f"{n} {word}s"


def describe_step(step: TraceStep) -> str:
    """One human line for a trace step."""
    out = _out(step)
    node = step.node
    if step.status == "failed" and node not in ("validate_citations", "check_answer", "review", "researcher", "repair_citations"):
        return f"{node.replace('_', ' ').capitalize()} failed: {step.error or step.output_summary}"[:200]
    if node == "intake":
        if out.get("in_scope") is False:
            return f"Out of scope ({out.get('scope_category', '?').replace('_', ' ')}): referred to a professional"
        return "Question in scope"
    if node == "referral":
        return "Referral sent; no research needed"
    if node == "planner":
        n = len(out.get("sub_questions") or [])
        return f"Planned {_plural(n, 'sub-question')}"
    if node == "plan_checkpoint":
        decision = (out.get("decision") or {}).get("action")
        n = len(out.get("sub_questions") or [])
        return f"Plan edited: {_plural(n, 'sub-question')}" if decision == "edit" else "Plan approved"
    if node == "dispatch":
        tasks = out.get("tasks") or []
        return f"Researching {_plural(len(tasks), 'sub-question')}" if tasks else "Research skipped (budget)"
    if node == "researcher":
        sq = step.sub_question_id or "?"
        if any(c.tool == "findings_store" and c.state == "reused" for c in step.tool_calls):
            kept = (out.get("status") or {}).get("findings_kept", 0)
            return f"{sq}: reused {_plural(kept, 'finding')} from an earlier run"
        status = out.get("status") or {}
        tools = [c.tool.split(":", 1)[1] for c in step.tool_calls if c.tool.startswith("search:")]
        source = SOURCE_LABELS.get(tools[0], tools[0]) if tools else "Search"
        seen, kept = status.get("records_seen", 0), status.get("findings_kept", 0)
        again = f" (pass {step.attempt})" if step.attempt > 1 else ""
        line = f"{source}: {_plural(seen, 'study')} → {kept} kept · {sq}{again}"
        state = status.get("state")
        return line if state in (None, "ok") else f"{line} ({state.replace('_', ' ')})"
    if node == "review":
        decision = out.get("decision")
        if decision == "send_back":
            return f"Review: sending back {', '.join(out.get('send_back_ids') or [])} for another pass"
        if decision == "fail":
            return f"Review: {step.output_summary}"
        return "Review: research complete"
    if node == "personalize":
        n = len((out.get("downweighted") or {}))
        return f"Personalized: down-weighted {_plural(n, 'finding')} from other populations" if n else "Personalized: every population matches"
    if node == "position_mapper":
        if step.status == "skipped":
            return "Position mapping skipped"
        pm = out.get("position_map") or out
        positions = pm.get("positions") or []
        status = (pm.get("status") or "?").replace("_", " ")
        return f"Positions mapped: {status}, {_plural(len(positions), 'side')}"
    if node == "advocate":
        kind = "opening" if step.attempt == 1 else "rebuttal"
        return f"Debate: round {step.attempt} · {step.position_id or '?'} {kind}"
    if node == "debate_review":
        return f"Debate: round {step.attempt} done" if step.attempt else "Debate reviewed"
    if node == "judge":
        if step.status == "skipped":
            return "Judge skipped"
        return f"Verdict: {(out.get('status') or '?').replace('_', ' ')}, confidence {out.get('confidence', '?')}"
    if node == "writer":
        return "Answer rewritten" + (" (escalated)" if step.escalated else "") if step.attempt > 1 else "Answer drafted"
    if node == "validate_citations":
        if step.status == "skipped":
            return "Citations: nothing to check"
        issues = len(out.get("issues") or [])
        checked = f"{out.get('claims_checked', 0)} claims and {out.get('numbers_checked', 0)} numbers"
        after = {"autofix": " after autofix", "repaired": " after repair"}.get(out.get("stage") or "draft", "")
        if issues == 0:
            return f"Citations: {checked} backed{after}"
        return f"Citations: {_plural(issues, 'problem')} in {checked}{after}"
    if node == "repair_citations":
        return "Citations repaired" if step.status == "ok" else f"Citation repair failed: {step.error or step.output_summary}"[:200]
    if node == "check_answer":
        decision = out.get("decision")
        return {"accept": "Answer accepted", "rewrite": "Answer sent back for one rewrite"}.get(
            decision or "", f"Answer rejected: {step.output_summary}"[:200]
        )
    if node == "finalize":
        return f"Done: {(out.get('status') or '?')}"
    return step.output_summary[:200]


def step_event(step: TraceStep, totals: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the progress stream shows for one trace step (the full step is in the trace)."""
    data = {
        "step_id": step.step_id,
        "index": step.index,
        "node": step.node,
        "agent": step.agent,
        "message": describe_step(step),
        "sub_question_id": step.sub_question_id,
        "position_id": step.position_id,
        "attempt": step.attempt,
        "status": step.status,
        "summary": step.output_summary,
        "decision": step.decision,
        "model": step.model,
        "tokens": step.input_tokens + step.output_tokens,
        "cost_usd": step.cost_usd,
        "latency_ms": step.latency_ms,
        "searches": sum(1 for c in step.tool_calls if c.tool.startswith("search:") and c.state != "refused"),
        "reused": any(c.tool == "findings_store" and c.state == "reused" for c in step.tool_calls),
        "replayed_calls": sum(1 for c in step.llm_calls if c.replayed),
    }
    if totals is not None:
        data["totals"] = totals
    return data

"""Eval reports: one JSON (for the gate, the frontier and the UI) and one Markdown summary."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.harness import EvalConfig
from evals.metrics import NO_ANSWER, STATUSES, ItemMetrics, aggregate

REPORT_SCHEMA = 1


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def representative(cfg: EvalConfig, judge: str) -> bool:
    return cfg.search_backend == "live" and cfg.llm == "anthropic" and judge == "llm"


def build_report(
    rows: list[ItemMetrics], cfg: EvalConfig, *, judge: str, label: str | None = None, notes: list[str] | None = None
) -> dict[str, Any]:
    notes = list(notes or [])
    rep = representative(cfg, judge)
    if cfg.search_backend == "fixture":
        notes.insert(
            0,
            "FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in "
            "backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage "
            "on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline "
            "and measures cost, latency, scope, calibration mechanics and the citation gate.",
        )
    if cfg.llm == "fake":
        notes.insert(0, "OFFLINE MODE: FakeLLM pipeline and deterministic judges; checks the harness and gate only.")
    if judge == "heuristic":
        notes.append("Heuristic judge: agreement and faithfulness are deterministic proxies, not model judgments.")
    return {
        "schema": REPORT_SCHEMA,
        "label": label or cfg.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "representative": rep,
        "config": cfg.model_dump() | {"name": cfg.name, "budget": cfg.budget().model_dump()},
        "judge": judge,
        "notes": notes,
        "aggregate": aggregate(rows),
        "items": [m.model_dump(mode="json") for m in rows],
    }


def _f(x: Any, pct: bool = False, money: bool = False) -> str:
    if x is None:
        return "–"
    if money:
        return f"${x:.4f}"
    if pct:
        return f"{100 * x:.0f}%"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def to_markdown(report: dict[str, Any]) -> str:
    a = report["aggregate"]
    cfg = report["config"]
    lines = [f"# Eval report: {report['label']}", ""]
    if not report["representative"]:
        lines += ["> **Not representative.** " + " ".join(report["notes"][:1]), ""]
    lines += [
        f"- Generated {report['generated_at']} at `{report.get('git_sha') or '?'}`; judge: `{report['judge']}`",
        f"- Search backend `{cfg['search_backend']}`, LLM `{cfg['llm']}`, budget cap ${cfg['max_usd']:g} per run"
        + (f", max {cfg['max_sub_questions']} sub-questions" if cfg.get("max_sub_questions") else ""),
        f"- Items: {a['items']} ({a['in_scope']} in scope, {a['out_of_scope']} out of scope), harness errors: {a['errors']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer rate (in scope) | {_f(a['answer_rate'], pct=True)} |",
        f"| Agreement (strict mean, no answer = 0) | {_f(a['agreement']['mean'])} |",
        f"| Agreement (answered only) | {_f(a['agreement']['answered_mean'])} |",
        f"| Faithfulness (claim-finding pairs) | {_f(a['faithfulness']['pair_mean'], pct=True)} of {a['faithfulness']['pairs']} pairs |",
        f"| Claims supported by a cited finding | {_f(a['faithfulness']['claim_support_mean'], pct=True)} |",
        f"| Debate calibration (exact / acceptable) | {_f(a['calibration']['exact_accuracy'], pct=True)} / {_f(a['calibration']['acceptable_accuracy'], pct=True)} (n={a['calibration']['n']}) |",
        f"| Key-study coverage | {_f(a['coverage']['micro'], pct=True)} ({a['coverage']['found']}/{a['coverage']['total']}) |",
        f"| Scope: out-of-scope referred / in-scope false referrals | {_f(a['scope']['out_of_scope_referred'], pct=True)} / {a['scope']['in_scope_false_referrals']} |",
        f"| Cost per run: mean / p50 / p90 | {_f(a['cost_usd']['mean'], money=True)} / {_f(a['cost_usd']['p50'], money=True)} / {_f(a['cost_usd']['p90'], money=True)} |",
        f"| Cost per in-scope run / per answered run (mean) | {_f(a['cost_usd']['mean_in_scope'], money=True)} / {_f(a['cost_usd']['mean_answered'], money=True)} |",
        f"| Latency: mean / p50 / p90 | {_f(a['latency_s']['mean'])}s / {_f(a['latency_s']['p50'])}s / {_f(a['latency_s']['p90'])}s |",
        f"| Total run spend / judge spend (cached judgments at their original cost) | {_f(a['cost_usd']['total'], money=True)} / {_f(a['cost_usd']['judge_total'], money=True)} |",
        "",
        "## Citation gate",
        "",
    ]
    g = a["citation_gate"]
    lines += [
        f"- First-draft pass rate: {_f(g['first_draft_pass_rate'], pct=True)} of {g['drafted']} drafted runs",
        f"- Citation repair used: {_f(g['repair_rate'], pct=True)} (repair passed: {_f(g['repair_pass_rate'], pct=True)});"
        f" escalated rewrite used: {_f(g['rewrite_rate'], pct=True)}; final pass: {_f(g['final_pass_rate'], pct=True)};"
        f" runs failed with citations_failed: {g['citations_failed']}",
        f"- First-draft issue kinds: {', '.join(f'{k} {v}' for k, v in sorted(g['first_draft_issue_kinds'].items())) or 'none'}",
        "",
        "## Debate calibration (expected rows, actual columns)",
        "",
        "| expected \\ actual | " + " | ".join((*STATUSES, NO_ANSWER)) + " |",
        "|---|" + "---|" * (len(STATUSES) + 1),
    ]
    for e, row in a["calibration"]["confusion_matrix"].items():
        lines.append(f"| {e} | " + " | ".join(str(row[c]) for c in (*STATUSES, NO_ANSWER)) + " |")
    lines += [
        "",
        f"Outcomes: {a['outcomes']}; failures: {a['failure_reasons'] or 'none'}",
        "",
        "## Items",
        "",
        "| Item | Outcome | Status (exp → act) | Agreement | Faithfulness | Coverage | Gate (1st draft) | Cost | Latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in report["items"]:
        gate = m["gate"]
        first = "–" if gate["first_draft_ok"] is None else ("pass" if gate["first_draft_ok"] else "fail")
        if gate["repaired"]:
            first += " → repaired" + ("" if gate["repair_ok"] else " (failed)")
        if gate["rewrite_used"]:
            first += " → rewrite"
        outcome = m["outcome"] + (f" ({m['failure_reason']})" if m["failure_reason"] else "")
        if not m["in_scope_expected"]:
            outcome += f" [{m['scope_category'] or '?'}{'' if m['scope_correct'] else ', WRONG'}]"
        elif m["referred"]:
            outcome += " [FALSE REFERRAL]"
        status = f"{m['expected_status'] or '–'} → {m['actual_status'] or '–'}" if m["in_scope_expected"] else "–"
        cov = f"{len(m['coverage_found'])}/{m['coverage_total']}" if m["coverage_total"] else "–"
        lines.append(
            f"| {m['item_id']} | {outcome} | {status} | {_f(m['agreement'])} | {_f(m['faithfulness'], pct=True)} | "
            f"{cov} | {first} | {_f(m['cost_usd'], money=True)} | {m['latency_s']:.0f}s |"
        )
    if report["notes"]:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    jp.write_text(json.dumps(report, indent=2) + "\n")
    mp.write_text(to_markdown(report))
    return jp, mp

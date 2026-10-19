"""Replay the writer on cached eval runs to measure the citation gate's first-draft pass rate.

A full run costs ~$0.10-0.20; the writer alone ~$0.03. This takes the writer's inputs
(intake, findings, position map, verdict) from cached runs, calls the writer `--samples`
times per run with the current prompt (or one from a git revision), and reports how often
the draft passes the gate as written, and after the deterministic autofix.

    backend/.venv/bin/python -m evals.writer_replay --tag fixture-anthropic-usd0.2 --samples 2
    backend/.venv/bin/python -m evals.writer_replay --tag ... --prompt-rev HEAD~1   # the old prompt
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from backend.agents import writer as writer_mod
from backend.citations import autofix_citations, validate_answer
from backend.graph.builder import RunResult
from backend.llm import AnthropicLLM
from evals.harness import DEFAULT_CACHE
from evals.run_eval import load_env


def prompt_at(rev: str) -> str:
    """SYSTEM_PROMPT of backend/agents/writer.py at a git revision."""
    src = subprocess.run(
        ["git", "show", f"{rev}:backend/agents/writer.py"], capture_output=True, text=True, check=True,
        cwd=Path(__file__).parent.parent,
    ).stdout
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SYSTEM_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"no SYSTEM_PROMPT in writer.py at {rev}")


def replay_one(result: RunResult) -> dict[str, Any]:
    assert result.intake is not None
    llm = AnthropicLLM()
    answer = writer_mod.WriterAgent(llm).run(
        result.intake, result.findings, positions=result.position_map, verdict=result.verdict
    )
    question = " ".join(filter(None, [result.intake.original_question, result.intake.scoped_question]))
    profile = result.intake.profile
    first = validate_answer(answer, result.findings, question=question, profile=profile)
    fixed, notes = autofix_citations(answer, result.findings, question=question, profile=profile)
    after = validate_answer(fixed, result.findings, question=question, profile=profile) if notes else first
    return {
        "ok": first.ok,
        "autofix_ok": after.ok,
        "autofixed": len(notes),
        "issues": dict(Counter(i.kind for i in first.issues)),
        "issues_after_autofix": dict(Counter(i.kind for i in after.issues)),
        "cost_usd": llm.total_cost_usd,
    }


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="cached run configuration (evals/.cache/runs/<tag>)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--prompt-rev", help="git revision to take the writer prompt from (default: working tree)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    prompt = prompt_at(args.prompt_rev) if args.prompt_rev else writer_mod.SYSTEM_PROMPT
    results = []
    for p in sorted((args.cache_dir / "runs" / args.tag).glob("*.json")):
        case = json.loads(p.read_text())
        if case.get("result") and case["result"].get("findings") and (case["result"].get("intake") or {}).get("in_scope"):
            results.append((case["item_id"], RunResult.model_validate(case["result"])))
    if not results:
        print("no cached in-scope runs with findings", file=sys.stderr)
        return 2
    jobs = [(item_id, r) for item_id, r in results for _ in range(args.samples)]
    original = writer_mod.SYSTEM_PROMPT
    writer_mod.SYSTEM_PROMPT = prompt  # WriterAgent.run reads the module constant at call time
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(lambda j: {"item_id": j[0]} | replay_one(j[1]), jobs))
    finally:
        writer_mod.SYSTEM_PROMPT = original
    n = len(rows)
    summary = {
        "prompt": args.prompt_rev or "working tree",
        "drafts": n,
        "first_draft_pass_rate": round(sum(r["ok"] for r in rows) / n, 4),
        "after_autofix_pass_rate": round(sum(r["autofix_ok"] for r in rows) / n, 4),
        "issue_kinds": dict(sum((Counter(r["issues"]) for r in rows), Counter())),
        "issue_kinds_after_autofix": dict(sum((Counter(r["issues_after_autofix"]) for r in rows), Counter())),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

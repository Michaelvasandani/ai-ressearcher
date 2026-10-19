"""Run the golden set through Evidence Coach and report agreement, calibration, faithfulness,
coverage, scope, cost and latency.

Examples (from the repo root):

    # smoke subset, real models, fixture search (offline research), $0.20 cap per run
    backend/.venv/bin/python -m evals.run_eval --tags smoke --budget 0.20

    # the full golden set against PubMed / Europe PMC (needs network): the representative numbers
    SEARCH_BACKEND=live backend/.venv/bin/python -m evals.run_eval --budget 0.50 --concurrency 3

    # recompute metrics from cached runs only (no new runs), then apply the CI floors
    backend/.venv/bin/python -m evals.run_eval --tags ci --cached-only --gate

    # deterministic offline mode: FakeLLM pipeline + heuristic judges, no network, no spend
    backend/.venv/bin/python -m evals.run_eval --llm fake --judge heuristic --gate

Runs are cached per configuration under evals/.cache/runs/<tag>/ and judge calls under
evals/.cache/judge/, so re-running the command only pays for what is missing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from evals.gate import THRESHOLDS_PATH, check, load_thresholds, profile_for
from evals.golden import GOLDEN_PATH, GoldenItem, load_golden, select
from evals.harness import DEFAULT_CACHE, EvalConfig, RunCache, RunCase, run_items
from evals.judges import HeuristicJudge, Judge, LLMJudge
from evals.metrics import item_metrics
from evals.report import build_report, write_report

ROOT = Path(__file__).parent
REPO = ROOT.parent
DEFAULT_OUT = ROOT / "reports"


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / "backend" / ".env", override=False)
    except ImportError:  # pragma: no cover
        pass


def make_judge(name: str, cache: RunCache | None, llm: Any = None) -> Judge | None:
    if name == "none":
        return None
    if name == "heuristic":
        return HeuristicJudge()
    if llm is None:
        from backend.llm import AnthropicLLM

        llm = AnthropicLLM()
    return LLMJudge(llm, cache_dir=cache.judge_dir() if cache else None)


def evaluate(
    items: list[GoldenItem],
    cfg: EvalConfig,
    *,
    judge: Judge | None,
    cache: RunCache | None,
    concurrency: int = 2,
    rerun: bool = False,
    cached_only: bool = False,
    max_total_usd: float | None = None,
    label: str | None = None,
    notes: list[str] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[RunCase]]:
    by_id = {i.id: i for i in items}

    def progress(case: RunCase, from_cache: bool) -> None:
        if not verbose:
            return
        r = case.result
        what = (r.outcome.status if r and r.outcome else "error") + (" (cached)" if from_cache else "")
        cost = r.spend.cost_usd if r else 0.0
        print(f"  {case.item_id:<32} {what:<20} ${cost:.4f} {case.wall_s:6.1f}s", flush=True)

    cases = run_items(
        items, cfg, cache=cache, concurrency=concurrency, rerun=rerun, cached_only=cached_only,
        max_total_usd=max_total_usd, on_done=progress,
    )
    rows = [item_metrics(by_id[c.item_id], c, judge) for c in cases]
    report = build_report(rows, cfg, judge=judge.name if judge else "none", label=label, notes=notes)
    return report, cases


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    ap.add_argument("--ids", help="comma-separated golden ids")
    ap.add_argument("--tags", help="comma-separated tags (smoke, ci, contested, out_of_scope, fixture_topic)")
    ap.add_argument("--sample", type=int, help="seeded random sample of the selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0.20, help="max_usd per run (default 0.20)")
    ap.add_argument("--max-sub-questions", type=int)
    ap.add_argument("--search-backend", choices=["fixture", "live"], default=None,
                    help="default: $SEARCH_BACKEND or fixture")
    ap.add_argument("--llm", choices=["anthropic", "fake"], default="anthropic")
    ap.add_argument("--judge", choices=["llm", "heuristic", "none"], default=None,
                    help="default: llm with --llm anthropic, heuristic with --llm fake")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="ignore cached runs")
    ap.add_argument("--cached-only", action="store_true", help="never run; report on cached runs only")
    ap.add_argument("--tag", help="cache / report name for this configuration")
    ap.add_argument("--max-total-usd", type=float, help="stop starting runs once new spend reaches this")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", help="report file stem (default: the configuration tag)")
    ap.add_argument("--label")
    ap.add_argument("--gate", action="store_true", help="exit 1 if a floor in thresholds.json is missed")
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))
    args = parse_args(argv)
    backend = args.search_backend or (os.environ.get("SEARCH_BACKEND") or "fixture").strip().lower()
    cfg = EvalConfig(
        search_backend=backend, llm=args.llm, max_usd=args.budget,
        max_sub_questions=args.max_sub_questions, tag=args.tag,
    )
    items = select(
        load_golden(args.golden),
        ids=args.ids.split(",") if args.ids else None,
        tags=args.tags.split(",") if args.tags else None,
        sample=args.sample,
        seed=args.seed,
    )
    if not items:
        print("no golden items selected", file=sys.stderr)
        return 2
    cache = None if args.no_cache else RunCache(args.cache_dir)
    judge = make_judge(args.judge or ("llm" if args.llm == "anthropic" else "heuristic"), cache)
    print(f"eval {cfg.name}: {len(items)} items, judge={judge.name if judge else 'none'}", flush=True)
    report, _ = evaluate(
        items, cfg, judge=judge, cache=cache, concurrency=args.concurrency, rerun=args.rerun,
        cached_only=args.cached_only, max_total_usd=args.max_total_usd, label=args.label,
    )
    jp, mp = write_report(report, args.out_dir, args.name or cfg.name)
    a = report["aggregate"]
    print(
        f"answer rate {a['answer_rate']}, agreement {a['agreement']['mean']}, "
        f"faithfulness {a['faithfulness']['pair_mean']}, calibration {a['calibration']['acceptable_accuracy']}, "
        f"first-draft gate pass {a['citation_gate']['first_draft_pass_rate']}, "
        f"cost mean ${a['cost_usd']['mean']}, total ${a['cost_usd']['total']} (+ judge ${a['cost_usd']['judge_total']})"
    )
    print(f"wrote {jp} and {mp}")
    if args.gate:
        failures = check(report, load_thresholds(args.thresholds))
        if failures:
            print(f"eval gate FAILED ({profile_for(report)}):")
            for f in failures:
                print(f"  - {f}")
            return 1
        print(f"eval gate passed ({profile_for(report)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

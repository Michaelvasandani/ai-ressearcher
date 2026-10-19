"""Cost-quality frontier: the same golden items at several per-run budget caps.

    backend/.venv/bin/python -m evals.frontier --tags smoke --budgets 0.05,0.20,1.00

Runs (or loads from the cache) one eval per cap, then writes:
- `<out>/frontier.json`: one point per cap (mean cost per run, answer rate, agreement,
  faithfulness, calibration, latency) plus the per-cap report paths. The UI can render it.
- `<out>/frontier.png`: quality (0-1) against mean cost per run, one line per metric.
- `<out>/frontier-usd<cap>.json/.md`: the full report for each cap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.golden import GOLDEN_PATH, load_golden, select
from evals.harness import DEFAULT_CACHE, EvalConfig, RunCache
from evals.run_eval import DEFAULT_OUT, evaluate, load_env, make_judge
from evals.report import write_report

# Categorical slots 1-3 of the reference palette (blue, orange, aqua), in fixed order.
SERIES = [
    ("agreement", "Agreement (strict)", "#2a78d6", "o"),
    ("faithfulness", "Faithfulness", "#eb6834", "s"),
    ("answer_rate", "Answer rate", "#1baf7a", "^"),
]


def point(budget: float, report: dict[str, Any]) -> dict[str, Any]:
    a = report["aggregate"]
    return {
        "max_usd": budget,
        "cost_mean_usd": a["cost_usd"]["mean_in_scope"],  # x axis: mean cost of an in-scope run
        "cost_mean_answered_usd": a["cost_usd"]["mean_answered"],
        "cost_p90_usd": a["cost_usd"]["p90"],
        "answer_rate": a["answer_rate"],
        "agreement": a["agreement"]["mean"],
        "agreement_answered": a["agreement"]["answered_mean"],
        "faithfulness": a["faithfulness"]["pair_mean"],
        "calibration_acceptable": a["calibration"]["acceptable_accuracy"],
        "first_draft_pass_rate": a["citation_gate"]["first_draft_pass_rate"],
        "latency_mean_s": a["latency_s"]["mean"],
        "outcomes": a["outcomes"],
        "failure_reasons": a["failure_reasons"],
        "items": a["items"],
    }


def plot(points: list[dict[str, Any]], path: Path, title: str, subtitle: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3dd"
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    fig.patch.set_facecolor("white")
    xs = [p["cost_mean_usd"] or 0.0 for p in points]
    for key, label, color, marker in SERIES:
        ys = [p[key] if p[key] is not None else float("nan") for p in points]
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7, label=label,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    last_x, level = None, 0
    for x, p in sorted(zip(xs, points), key=lambda t: t[0]):
        # One hairline per budget cap, labelled at the top; staggered when caps land close together.
        level = level + 1 if last_x is not None and x - last_x < 0.15 * max(xs) else 0
        last_x = x
        ax.axvline(x, color=grid, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.annotate(f"cap ${p['max_usd']:.2f}", (x, 1.2), xytext=(0, -9 - 10 * level), textcoords="offset points",
                    ha="center", fontsize=8, color=muted)
    ax.set_xlabel("Mean cost per in-scope run (USD)", color=ink, fontsize=9)
    ax.set_ylabel("Score (0-1)", color=ink, fontsize=9)
    ax.set_ylim(-0.03, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0, max(xs) * 1.12 if xs and max(xs) > 0 else 1)
    ax.grid(True, color=grid, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, labelcolor=ink)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=ink, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=8, color=muted)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    ap.add_argument("--ids")
    ap.add_argument("--tags", default="smoke")
    ap.add_argument("--budgets", default="0.05,0.20,1.00")
    ap.add_argument("--search-backend", choices=["fixture", "live"], default="fixture")
    ap.add_argument("--llm", choices=["anthropic", "fake"], default="anthropic")
    ap.add_argument("--judge", choices=["llm", "heuristic", "none"])
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--max-total-usd", type=float, help="per cap: stop starting runs once new spend reaches this")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    items = select(
        load_golden(args.golden), ids=args.ids.split(",") if args.ids else None,
        tags=args.tags.split(",") if args.tags else None,
    )
    cache = RunCache(args.cache_dir)
    judge = make_judge(args.judge or ("llm" if args.llm == "anthropic" else "heuristic"), cache)
    points, reports = [], {}
    for budget in [float(b) for b in args.budgets.split(",")]:
        cfg = EvalConfig(search_backend=args.search_backend, llm=args.llm, max_usd=budget)
        print(f"budget ${budget:g}: {len(items)} items ({cfg.name})", flush=True)
        report, _ = evaluate(
            items, cfg, judge=judge, cache=cache, concurrency=args.concurrency,
            cached_only=args.cached_only, max_total_usd=args.max_total_usd,
        )
        jp, _ = write_report(report, args.out_dir, f"frontier-usd{budget:g}")
        reports[f"{budget:g}"] = jp.name
        points.append(point(budget, report))
    representative = args.search_backend == "live" and args.llm == "anthropic"
    note = (
        "live search" if representative
        else f"{args.search_backend} search backend, NOT representative of answer quality"
    )
    out = {
        "schema": 1,
        "tags": args.tags,
        "items": [i.id for i in items],
        "search_backend": args.search_backend,
        "llm": args.llm,
        "representative": representative,
        "note": note,
        "points": points,
        "reports": reports,
    }
    (args.out_dir / "frontier.json").write_text(json.dumps(out, indent=2) + "\n")
    plot(
        points, args.out_dir / "frontier.png", "Cost-quality frontier",
        f"{len(items)} golden items ({args.tags or 'all'}), one run each per budget cap; {note}",
    )
    for p in points:
        print(
            f"  cap ${p['max_usd']:<5g} cost ${p['cost_mean_usd']}  answer {p['answer_rate']}  "
            f"agreement {p['agreement']}  faithfulness {p['faithfulness']}"
        )
    print(f"wrote {args.out_dir / 'frontier.json'} and {args.out_dir / 'frontier.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

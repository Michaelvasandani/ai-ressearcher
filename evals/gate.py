"""CI gate: compare a report's aggregates with the floors in `thresholds.json`.

    python -m evals.gate evals/reports/ci.json            # exit 1 if any floor is missed

Floors are per profile: `offline` (FakeLLM pipeline, deterministic judges: checks the
harness itself), `fixture` (real models, synthetic library) and `live` (real models,
PubMed / Europe PMC). A metric that was not measured (None) fails its floor: a gate
that cannot see the number must not pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"


def profile_for(report: dict[str, Any]) -> str:
    cfg = report["config"]
    return "offline" if cfg["llm"] == "fake" else cfg["search_backend"]


def metric(aggregate: dict[str, Any], path: str) -> Any:
    cur: Any = aggregate
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check(report: dict[str, Any], thresholds: dict[str, Any], profile: str | None = None) -> list[str]:
    """Failures, one line each; empty = pass."""
    profile = profile or profile_for(report)
    rules = thresholds["profiles"].get(profile)
    if rules is None:
        return [f"no thresholds for profile {profile!r}"]
    failures: list[str] = []
    agg = report["aggregate"]
    for path, floor in rules.get("min", {}).items():
        value = metric(agg, path)
        if value is None:
            failures.append(f"{path}: not measured (floor {floor})")
        elif value < floor:
            failures.append(f"{path}: {value:.4g} < floor {floor}")
    for path, ceiling in rules.get("max", {}).items():
        value = metric(agg, path)
        if value is None:
            failures.append(f"{path}: not measured (ceiling {ceiling})")
        elif value > ceiling:
            failures.append(f"{path}: {value:.4g} > ceiling {ceiling}")
    return failures


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path)
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    ap.add_argument("--profile", help="override the profile picked from the report's config")
    args = ap.parse_args(argv)
    report = json.loads(args.report.read_text())
    failures = check(report, load_thresholds(args.thresholds), args.profile)
    profile = args.profile or profile_for(report)
    if failures:
        print(f"eval gate FAILED ({profile}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"eval gate passed ({profile})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

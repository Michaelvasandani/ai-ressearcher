"""Phase 7 eval harness: golden set, judges, metrics, report, the CI gate and the offline mode.

Everything here is offline and deterministic: the pipeline runs on FakeLLM (`--llm fake`),
judges are the heuristic ones or FakeLLM, and the committed fixture-smoke runs are re-scored
from their cached judgments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.llm import FakeLLM, LLMError
from backend.schemas import DISCLAIMER, CoachAnswer, Position
from backend.tests.conftest import make_finding
from evals import gate
from evals.golden import load_golden, select
from evals.harness import EvalConfig, RunCache, run_items
from evals.judges import (
    AGREEMENT_AGENT,
    FAITHFULNESS_AGENT,
    AgreementJudgment,
    FaithfulnessJudgment,
    HeuristicJudge,
    LLMJudge,
    claim_pairs,
)
from evals.metrics import aggregate, confusion_matrix, coverage, item_metrics
from evals.report import build_report, to_markdown

EVALS = Path(__file__).resolve().parents[2] / "evals"
SMOKE = EVALS / "reports" / "fixture-smoke"

# --- Golden set ---------------------------------------------------------------------------


def test_golden_set_shape():
    items = load_golden()
    assert 30 <= len(items) <= 50
    oos = [i for i in items if not i.in_scope]
    assert len(oos) >= 5 and all(i.expected_status is None for i in oos)
    assert {i.expected_scope_category for i in oos} >= {"injury", "pregnancy", "medication", "eating_disorder", "medical_condition"}
    assert sum(i.expected_status == "genuinely_contested" for i in items) >= 2
    assert sum("contested" in i.tags for i in items) >= 6
    for i in items:
        if i.in_scope:
            assert i.reference.recommendation and i.source and i.key_studies, i.id
        for s in i.key_studies:
            assert s.pmid is None or re.fullmatch(r"\d{6,8}", s.pmid), (i.id, s.pmid)
            assert s.doi is None or s.doi.startswith("10."), (i.id, s.doi)
    assert sum(len(i.scored_studies()) for i in items) >= 40  # trusted identifiers for coverage


def test_select_by_tags_ids_and_seeded_sample():
    items = load_golden()
    assert [i.id for i in select(items, tags=["ci"])] == ["vol-sets-hypertrophy", "protein-daily", "creatine-safety", "oos-pregnancy"]
    assert [i.id for i in select(items, ids=["protein-cut"])] == ["protein-cut"]
    a, b = select(items, sample=5, seed=3), select(items, sample=5, seed=3)
    assert [i.id for i in a] == [i.id for i in b] and len(a) == 5
    with pytest.raises(KeyError):
        select(items, ids=["nope"])


# --- Judges and per-item metrics -------------------------------------------------------------

F1 = make_finding("sq1:PMID:27433992", claim="10 or more weekly sets beat fewer than 5.", snippet="Ten or more weekly sets per muscle produced larger gains.")
F2 = make_finding("sq2:10.1136/bjsports-2017-097608", claim="Protein gains plateaued at 1.62 g/kg/day.", snippet="Plateau at 1.62 g/kg/day.")


def coach_answer(**over) -> CoachAnswer:
    data = dict(
        status="leaning",
        recommendation="Do 10 or more hard sets per muscle each week [1].",
        confidence="moderate",
        why="Ten or more weekly sets produced larger gains [1]. Protein beyond 1.62 g/kg/day added little [2].",
        positions=[Position(stance="More volume, more growth", supporting_findings=[F1.id], evidence_grade="high", best_counterargument="x")],
        what_would_change_this="x", applies_to_you="x", practical_protocol="Spread the sets out.", whats_new=None,
        cited_finding_ids=[F1.id, F2.id], disclaimer=DISCLAIMER,
    )
    data.update(over)
    return CoachAnswer(**data)


def test_claim_pairs_follow_markers_and_positions():
    pairs, total = claim_pairs(coach_answer(), [F1, F2])
    assert [(p.field, p.finding_id) for p in pairs] == [
        ("recommendation", F1.id), ("why", F1.id), ("why", F2.id), ("positions[1].stance", F1.id)
    ]
    assert total == 4 and "[" not in pairs[0].claim


def test_llm_judge_scores_caches_and_counts_skipped_pairs_as_unsupported(tmp_path):
    item = select(load_golden(), ids=["vol-sets-hypertrophy"])[0]
    llm = FakeLLM(
        {
            AGREEMENT_AGENT: AgreementJudgment(
                rating="mostly_agrees", numbers_consistent="yes", contradictions=[], missing_key_points=["upper range"], rationale="ok"
            ),
            # Pair 4 is not judged: it must count as unsupported.
            FAITHFULNESS_AGENT: FaithfulnessJudgment(
                judgments=[
                    {"pair_id": 1, "label": "supports", "reason": "r"},
                    {"pair_id": 2, "label": "supports", "reason": "r"},
                    {"pair_id": 3, "label": "partially_supports", "reason": "r"},
                ]
            ),
        }
    )
    judge = LLMJudge(llm, cache_dir=tmp_path)
    agr = judge.agreement(item, coach_answer())
    faith = judge.faithfulness(coach_answer(), [F1, F2])
    assert agr.score == 0.75 and agr.rating == "mostly_agrees" and agr.cost_usd > 0
    assert faith.score == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert faith.claim_support == 0.5  # 4 claims: recommendation and the first why sentence are supported
    req = llm.calls_for(AGREEMENT_AGENT)[0]
    assert "Reference recommendation" in req.user and "0.37" in req.user and "[1]" not in req.user
    # Same requests again: answered from the cache, no model call, same original cost reported.
    again = LLMJudge(FakeLLM({}), cache_dir=tmp_path)
    assert again.agreement(item, coach_answer()).cost_usd == agr.cost_usd
    assert again.faithfulness(coach_answer(), [F1, F2]).score == faith.score


def test_heuristic_judge_is_deterministic():
    item = select(load_golden(), ids=["vol-sets-hypertrophy"])[0]
    j = HeuristicJudge()
    a = j.agreement(item, coach_answer(why="0.37% per set [1]."))
    assert a.score == 1.0  # both key numbers (10 and 0.37) appear
    assert j.agreement(item, coach_answer(recommendation="Do 3 sets [1].", why="x [1].")).score == 0.0
    f = j.faithfulness(coach_answer(), [F1, F2])
    assert f.score is not None and f.pairs[0]["label"] == "supports"


def test_coverage_matches_pmid_and_doi():
    item = select(load_golden(), ids=["protein-daily"])[0]
    found, total = coverage(item, [F1, F2])
    assert total == 3 and found == ["28698222"]  # matched by DOI (F2's source id); PMIDs too:
    item2 = select(load_golden(), ids=["vol-sets-hypertrophy"])[0]
    assert coverage(item2, [F1])[0] == ["27433992"]


# --- The harness end to end, offline ---------------------------------------------------------


def test_offline_run_eval_end_to_end_passes_the_gate(tmp_path, capsys):
    from evals import run_eval

    code = run_eval.main(["--llm", "fake", "--no-cache", "--out-dir", str(tmp_path), "--gate"])
    out = capsys.readouterr().out
    assert code == 0, out
    report = json.loads((tmp_path / "fixture-fake-usd0.2.json").read_text())
    a = report["aggregate"]
    assert report["judge"] == "heuristic" and report["representative"] is False
    assert a["answer_rate"] == 1.0 and a["scope"]["out_of_scope_referred"] == 1.0 and a["errors"] == 0
    assert a["citation_gate"]["first_draft_pass_rate"] == 1.0
    md = (tmp_path / "fixture-fake-usd0.2.md").read_text()
    assert "OFFLINE MODE" in md and "| vol-sets-hypertrophy |" in md


def test_cache_round_trip_and_cached_only(tmp_path):
    items = select(load_golden(), ids=["creatine-safety", "oos-pregnancy"])
    cfg = EvalConfig(llm="fake", max_usd=0.2)
    cache = RunCache(tmp_path)
    first = run_items(items, cfg, cache=cache, concurrency=2)
    assert [c.item_id for c in first] == ["creatine-safety", "oos-pregnancy"]
    assert cache.path(cfg, "creatine-safety").exists()
    seen: list[bool] = []
    again = run_items(items, cfg, cache=cache, cached_only=True, on_done=lambda c, cached: seen.append(cached))
    assert seen == [True, True] and again[0].result.answer == first[0].result.answer
    assert run_items(items, EvalConfig(llm="fake", max_usd=0.5), cache=cache, cached_only=True) == []


def test_a_writer_regression_fails_the_offline_gate(tmp_path):
    """The gate's purpose: a pipeline that stops producing valid answers must fail it."""
    from backend.tests import graph_fakes as gf
    from evals import offline
    from evals.run_eval import evaluate

    original = offline._writer
    offline._writer = lambda: gf.writer_response(unbacked_number="17")  # every draft fails the citation gate
    try:
        items = select(load_golden(), tags=["ci"])
        report, _ = evaluate(items, EvalConfig(llm="fake"), judge=HeuristicJudge(), cache=None, verbose=False)
    finally:
        offline._writer = original
    failures = gate.check(report, gate.load_thresholds())
    assert any(f.startswith("answer_rate") for f in failures)
    assert any(f.startswith("citation_gate.citations_failed") for f in failures)


# --- The gate ------------------------------------------------------------------------------------


def _report(**agg) -> dict:
    base = {"agreement": {"mean": 0.8}, "faithfulness": {"pair_mean": 0.9}, "answer_rate": 1.0, "errors": 0}
    base.update(agg)
    return {"config": {"llm": "anthropic", "search_backend": "live"}, "aggregate": base}


def test_gate_floors_ceilings_and_unmeasured_metrics():
    t = {"profiles": {"live": {"min": {"agreement.mean": 0.5, "faithfulness.pair_mean": 0.8}, "max": {"errors": 0}}}}
    assert gate.check(_report(), t) == []
    assert gate.check(_report(agreement={"mean": 0.4}), t) == ["agreement.mean: 0.4 < floor 0.5"]
    assert gate.check(_report(faithfulness={"pair_mean": None}), t) == ["faithfulness.pair_mean: not measured (floor 0.8)"]
    assert gate.check(_report(errors=2), t) == ["errors: 2 > ceiling 0"]
    assert gate.check(_report(), t, profile="fixture") == ["no thresholds for profile 'fixture'"]
    assert gate.profile_for({"config": {"llm": "fake", "search_backend": "live"}}) == "offline"


def test_gate_cli_exit_codes(tmp_path, capsys):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(_report(**{"scope": {"out_of_scope_referred": 1.0, "in_scope_false_referrals": 0}})))
    bad.write_text(json.dumps(_report(agreement={"mean": 0.1})))
    assert gate.main([str(good)]) == 0
    assert gate.main([str(bad)]) == 1
    assert "agreement.mean" in capsys.readouterr().out


def test_thresholds_cover_agreement_and_faithfulness_everywhere():
    t = gate.load_thresholds()
    for profile in ("offline", "fixture", "live"):
        assert {"agreement.mean", "faithfulness.pair_mean"} <= set(t["profiles"][profile]["min"])


# --- The committed fixture smoke run, re-scored from its cache ---------------------------------


class _NoCalls(FakeLLM):
    def _call(self, request):  # every judgment must come from the committed cache
        raise LLMError(f"unexpected judge call: {request.agent}")


def test_committed_smoke_report_recomputes_from_cached_runs():
    # Judgments are cached by a hash of the judge prompt: after changing evals/judges.py prompts,
    # re-run the smoke eval and refresh evals/reports/fixture-smoke/ (see evals/README.md).
    committed = json.loads((SMOKE / "after-fix-usd0.2.json").read_text())
    items = select(load_golden(), tags=["smoke"])
    cache = RunCache(SMOKE / "cache")
    cfg = EvalConfig(search_backend="fixture", llm="anthropic", max_usd=0.2)
    cases = run_items(items, cfg, cache=cache, cached_only=True)
    assert len(cases) == len(items)
    judge = LLMJudge(_NoCalls(), cache_dir=cache.judge_dir())
    by_id = {i.id: i for i in items}
    rows = [item_metrics(by_id[c.item_id], c, judge) for c in cases]
    agg = aggregate(rows)
    for key in ("answer_rate", "agreement", "faithfulness", "calibration", "citation_gate", "scope"):
        assert agg[key] == committed["aggregate"][key], key
    assert gate.check(committed, gate.load_thresholds()) == []  # the fix passes the fixture floors...
    before = json.loads((SMOKE / "before-fix-usd0.2.json").read_text())
    assert gate.check(before, gate.load_thresholds())  # ...and the Phase 6 behaviour would have failed them
    report = build_report(rows, cfg, judge="llm")
    assert "FIXTURE BACKEND, NOT REPRESENTATIVE" in to_markdown(report)
    assert confusion_matrix(rows)["leaning"]["leaning"] >= 1


def test_frontier_offline_writes_json_and_png(tmp_path):
    pytest.importorskip("matplotlib")
    from evals import frontier

    assert frontier.main(["--llm", "fake", "--tags", "ci", "--budgets", "0.05,0.2", "--out-dir", str(tmp_path),
                          "--cache-dir", str(tmp_path / "cache")]) == 0
    data = json.loads((tmp_path / "frontier.json").read_text())
    assert [p["max_usd"] for p in data["points"]] == [0.05, 0.2] and data["representative"] is False
    assert (tmp_path / "frontier.png").stat().st_size > 10_000

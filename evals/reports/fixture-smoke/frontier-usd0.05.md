# Eval report: fixture-anthropic-usd0.05

> **Not representative.** FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline and measures cost, latency, scope, calibration mechanics and the citation gate.

- Generated 2026-10-19T19:33:44+00:00 at `086b1ab`; judge: `llm`
- Search backend `fixture`, LLM `anthropic`, budget cap $0.05 per run
- Items: 6 (4 in scope, 2 out of scope), harness errors: 0

## Summary

| Metric | Value |
|---|---|
| Answer rate (in scope) | 0% |
| Agreement (strict mean, no answer = 0) | 0.00 |
| Agreement (answered only) | – |
| Faithfulness (claim-finding pairs) | – of 0 pairs |
| Claims supported by a cited finding | – |
| Debate calibration (exact / acceptable) | – / – (n=0) |
| Key-study coverage | 0% (0/9) |
| Scope: out-of-scope referred / in-scope false referrals | 100% / 0 |
| Cost per run: mean / p50 / p90 | $0.0278 / $0.0392 / $0.0410 |
| Cost per in-scope run / per answered run (mean) | $0.0416 / – |
| Latency: mean / p50 / p90 | 19.38s / 25.52s / 30.21s |
| Total run spend / judge spend (cached judgments at their original cost) | $0.1668 / $0.0000 |

## Citation gate

- First-draft pass rate: 0% of 4 drafted runs
- Citation repair used: 100% (repair passed: 0%); escalated rewrite used: 0%; final pass: 0%; runs failed with citations_failed: 0
- First-draft issue kinds: unbacked_number 8, uncited_claim 1, uncited_number 1

## Debate calibration (expected rows, actual columns)

| expected \ actual | consensus | leaning | genuinely_contested | no_answer |
|---|---|---|---|---|
| consensus | 0 | 0 | 0 | 1 |
| leaning | 0 | 0 | 0 | 3 |
| genuinely_contested | 0 | 0 | 0 | 0 |

Outcomes: {'failed': 4, 'referred': 2}; failures: {'budget_exhausted': 4}

## Items

| Item | Outcome | Status (exp → act) | Agreement | Faithfulness | Coverage | Gate (1st draft) | Cost | Latency |
|---|---|---|---|---|---|---|---|---|
| vol-sets-hypertrophy | failed (budget_exhausted) | leaning → – | – | – | 0/2 | fail → repaired (failed) | $0.0468 | 32s |
| failure-hypertrophy | failed (budget_exhausted) | leaning → – | – | – | 0/2 | fail → repaired (failed) | $0.0410 | 30s |
| protein-cut | failed (budget_exhausted) | leaning → – | – | – | 0/3 | fail → repaired (failed) | $0.0392 | 26s |
| creatine-safety | failed (budget_exhausted) | consensus → – | – | – | 0/2 | fail → repaired (failed) | $0.0394 | 27s |
| oos-injury-rehab | referred [injury] | – | – | – | – | – | $0.0003 | 2s |
| oos-pregnancy | referred [pregnancy] | – | – | – | – | – | $0.0000 | 0s |

## Notes

- FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline and measures cost, latency, scope, calibration mechanics and the citation gate.

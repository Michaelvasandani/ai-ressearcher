# Eval report: fixture-anthropic-usd1

> **Not representative.** FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline and measures cost, latency, scope, calibration mechanics and the citation gate.

- Generated 2026-10-19T19:33:44+00:00 at `086b1ab`; judge: `llm`
- Search backend `fixture`, LLM `anthropic`, budget cap $1 per run
- Items: 6 (4 in scope, 2 out of scope), harness errors: 0

## Summary

| Metric | Value |
|---|---|
| Answer rate (in scope) | 100% |
| Agreement (strict mean, no answer = 0) | 0.69 |
| Agreement (answered only) | 0.69 |
| Faithfulness (claim-finding pairs) | 74% of 81 pairs |
| Claims supported by a cited finding | 62% |
| Debate calibration (exact / acceptable) | 75% / 100% (n=4) |
| Key-study coverage | 0% (0/9) |
| Scope: out-of-scope referred / in-scope false referrals | 100% / 0 |
| Cost per run: mean / p50 / p90 | $0.0748 / $0.0698 / $0.1139 |
| Cost per in-scope run / per answered run (mean) | $0.1121 / $0.1121 |
| Latency: mean / p50 / p90 | 34.71s / 39.58s / 54.69s |
| Total run spend / judge spend (cached judgments at their original cost) | $0.4487 / $0.0348 |

## Citation gate

- First-draft pass rate: 50% of 4 drafted runs
- Citation repair used: 25% (repair passed: 100%); escalated rewrite used: 0%; final pass: 100%; runs failed with citations_failed: 0
- First-draft issue kinds: unbacked_number 2, uncited_number 1

## Debate calibration (expected rows, actual columns)

| expected \ actual | consensus | leaning | genuinely_contested | no_answer |
|---|---|---|---|---|
| consensus | 1 | 0 | 0 | 0 |
| leaning | 1 | 2 | 0 | 0 |
| genuinely_contested | 0 | 0 | 0 | 0 |

Outcomes: {'answered': 1, 'degraded': 3, 'referred': 2}; failures: none

## Items

| Item | Outcome | Status (exp → act) | Agreement | Faithfulness | Coverage | Gate (1st draft) | Cost | Latency |
|---|---|---|---|---|---|---|---|---|
| vol-sets-hypertrophy | answered | leaning → leaning | 0.75 | 64% | 0/2 | fail | $0.1885 | 70s |
| failure-hypertrophy | degraded | leaning → leaning | 0.75 | 81% | 0/2 | pass | $0.1139 | 55s |
| protein-cut | degraded | leaning → consensus | 0.25 | 85% | 0/3 | fail → repaired | $0.0762 | 42s |
| creatine-safety | degraded | consensus → consensus | 1.00 | 69% | 0/2 | pass | $0.0698 | 40s |
| oos-injury-rehab | referred [injury] | – | – | – | – | – | $0.0003 | 3s |
| oos-pregnancy | referred [pregnancy] | – | – | – | – | – | $0.0000 | 0s |

## Notes

- FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline and measures cost, latency, scope, calibration mechanics and the citation gate.

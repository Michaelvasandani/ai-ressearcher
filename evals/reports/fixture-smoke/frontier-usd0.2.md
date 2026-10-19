# Eval report: fixture-anthropic-usd0.2

> **Not representative.** FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline and measures cost, latency, scope, calibration mechanics and the citation gate.

- Generated 2026-10-19T19:33:44+00:00 at `086b1ab`; judge: `llm`
- Search backend `fixture`, LLM `anthropic`, budget cap $0.2 per run
- Items: 6 (4 in scope, 2 out of scope), harness errors: 0

## Summary

| Metric | Value |
|---|---|
| Answer rate (in scope) | 100% |
| Agreement (strict mean, no answer = 0) | 0.69 |
| Agreement (answered only) | 0.69 |
| Faithfulness (claim-finding pairs) | 66% of 88 pairs |
| Claims supported by a cited finding | 54% |
| Debate calibration (exact / acceptable) | 75% / 100% (n=4) |
| Key-study coverage | 0% (0/9) |
| Scope: out-of-scope referred / in-scope false referrals | 100% / 0 |
| Cost per run: mean / p50 / p90 | $0.0685 / $0.0727 / $0.1055 |
| Cost per in-scope run / per answered run (mean) | $0.1027 / $0.1027 |
| Latency: mean / p50 / p90 | 34.21s / 40.54s / 53.05s |
| Total run spend / judge spend (cached judgments at their original cost) | $0.4111 / $0.0353 |

## Citation gate

- First-draft pass rate: 50% of 4 drafted runs
- Citation repair used: 50% (repair passed: 100%); escalated rewrite used: 0%; final pass: 100%; runs failed with citations_failed: 0
- First-draft issue kinds: unbacked_number 3, uncited_number 2

## Debate calibration (expected rows, actual columns)

| expected \ actual | consensus | leaning | genuinely_contested | no_answer |
|---|---|---|---|---|
| consensus | 1 | 0 | 0 | 0 |
| leaning | 1 | 2 | 0 | 0 |
| genuinely_contested | 0 | 0 | 0 | 0 |

Outcomes: {'degraded': 4, 'referred': 2}; failures: none

## Items

| Item | Outcome | Status (exp → act) | Agreement | Faithfulness | Coverage | Gate (1st draft) | Cost | Latency |
|---|---|---|---|---|---|---|---|---|
| vol-sets-hypertrophy | degraded | leaning → leaning | 0.75 | 48% | 0/2 | fail → repaired | $0.1425 | 61s |
| failure-hypertrophy | degraded | leaning → leaning | 0.75 | 62% | 0/2 | pass | $0.1055 | 53s |
| protein-cut | degraded | leaning → consensus | 0.25 | 85% | 0/3 | fail → repaired | $0.0900 | 49s |
| creatine-safety | degraded | consensus → consensus | 1.00 | 78% | 0/2 | pass | $0.0727 | 41s |
| oos-injury-rehab | referred [injury] | – | – | – | – | – | $0.0003 | 2s |
| oos-pregnancy | referred [pregnancy] | – | – | – | – | – | $0.0000 | 0s |

## Notes

- FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline and measures cost, latency, scope, calibration mechanics and the citation gate.

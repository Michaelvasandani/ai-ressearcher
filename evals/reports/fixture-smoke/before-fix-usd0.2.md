# Eval report: Before the citation fix (writer prompt at bbc8daa, no repair): fixture backend, $0.20 cap

> **Not representative.** FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline and measures cost, latency, scope, calibration mechanics and the citation gate.

- Generated 2026-10-19T19:33:40+00:00 at `086b1ab`; judge: `llm`
- Search backend `fixture`, LLM `anthropic`, budget cap $0.2 per run
- Items: 6 (4 in scope, 2 out of scope), harness errors: 0

## Summary

| Metric | Value |
|---|---|
| Answer rate (in scope) | 25% |
| Agreement (strict mean, no answer = 0) | 0.25 |
| Agreement (answered only) | 1.00 |
| Faithfulness (claim-finding pairs) | 67% of 21 pairs |
| Claims supported by a cited finding | 47% |
| Debate calibration (exact / acceptable) | 100% / 100% (n=1) |
| Key-study coverage | 0% (0/9) |
| Scope: out-of-scope referred / in-scope false referrals | 100% / 0 |
| Cost per run: mean / p50 / p90 | $0.0909 / $0.0706 / $0.1409 |
| Cost per in-scope run / per answered run (mean) | $0.1362 / $0.1409 |
| Latency: mean / p50 / p90 | 38.76s / 38.15s / 55.37s |
| Total run spend / judge spend (cached judgments at their original cost) | $0.5452 / $0.0085 |

## Citation gate

- First-draft pass rate: 0% of 4 drafted runs
- Citation repair used: 0% (repair passed: –); escalated rewrite used: 50%; final pass: 25%; runs failed with citations_failed: 3
- First-draft issue kinds: unbacked_number 1, uncited_claim 8, uncited_number 9

## Debate calibration (expected rows, actual columns)

| expected \ actual | consensus | leaning | genuinely_contested | no_answer |
|---|---|---|---|---|
| consensus | 1 | 0 | 0 | 0 |
| leaning | 0 | 0 | 0 | 3 |
| genuinely_contested | 0 | 0 | 0 | 0 |

Outcomes: {'failed': 3, 'answered': 1, 'referred': 2}; failures: {'citations_failed': 3}

## Items

| Item | Outcome | Status (exp → act) | Agreement | Faithfulness | Coverage | Gate (1st draft) | Cost | Latency |
|---|---|---|---|---|---|---|---|---|
| vol-sets-hypertrophy | failed (citations_failed) | leaning → – | – | – | 0/2 | fail → rewrite | $0.2084 | 84s |
| failure-hypertrophy | failed (citations_failed) | leaning → – | – | – | 0/2 | fail | $0.1250 | 50s |
| protein-cut | failed (citations_failed) | leaning → – | – | – | 0/3 | fail | $0.0706 | 38s |
| creatine-safety | answered | consensus → consensus | 1.00 | 67% | 0/2 | fail → rewrite | $0.1409 | 55s |
| oos-injury-rehab | referred [injury] | – | – | – | – | – | $0.0003 | 5s |
| oos-pregnancy | referred [pregnancy] | – | – | – | – | – | $0.0000 | 0s |

## Notes

- FIXTURE BACKEND, NOT REPRESENTATIVE: research used the small synthetic library in backend/tools/fixtures/studies.json (5 topics, fake FIXTURE-* ids). Agreement and coverage on real questions need SEARCH_BACKEND=live with network access; this run proves the pipeline and measures cost, latency, scope, calibration mechanics and the citation gate.

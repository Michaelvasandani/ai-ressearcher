# Evidence Coach evals

Scores the coach against a golden set of questions with reference answers from ISSN / ACSM / NSCA
position stands and well-known meta-analyses, and gates CI on the result.

Run everything from the repo root with the backend venv (`backend/.venv/bin/python`; the
`dev` extra adds matplotlib for the frontier plot).

## Golden set (`golden/golden.json`)

47 items: 40 in scope (training volume, frequency, proximity to failure, load, rest, programming,
protein, nutrient timing, creatine, body composition) and 7 out of scope (injury, pregnancy,
medication, eating disorder, diagnosis, medical condition, off topic) that must be referred.
Each item has:

- the question and an optional `profile` (a `UserProfile`);
- `expected_status` (`consensus` / `leaning` / `genuinely_contested`) plus `acceptable_statuses` for
  borderline items, for debate calibration;
- `reference`: the recommendation, key points and key numbers, paraphrased from the `source`;
- `key_studies` with PMID / DOI and `id_confidence`: `verified` (confirmed by web search while
  writing the set), `high`, `medium` (recalled), `unverified` (not trusted; excluded from coverage);
- `notes` on uncertainty. Numbers appear only where the source states them.
- `tags`: `smoke` (4 in scope + 2 out of scope, all on topics the fixture library covers), `ci`
  (the CI subset), `contested`, `out_of_scope`, `fixture_topic`.

`golden.py` validates the file (`load_golden`) and selects items (`select`: ids, tags, seeded sample).

## Metrics (`metrics.py`, `judges.py`)

| Metric | How |
|---|---|
| Agreement | Sonnet judge, fixed rubric (`agrees` 1 / `mostly_agrees` 0.75 / `partially_agrees` 0.25 / `disagrees` 0), structured output. Strict mean counts an in-scope item without an answer as 0. |
| Debate calibration | expected vs actual status: confusion matrix, exact and acceptable accuracy, false debates, missed debates |
| Faithfulness | Haiku judge per (claim sentence, cited finding) pair: supports 1 / partially 0.5 / does not 0; also the share of claims supported by at least one cited finding. Pairs come from the `[n]` markers and position stances. |
| Coverage | golden key studies (trusted ids) whose PMID / DOI is among the run's findings |
| Scope | out-of-scope items referred with an acceptable category; in-scope false referrals |
| Cost, latency | the run's metered spend and wall-clock time (mean, p50, p90), plus judge spend |
| Citation gate | first-draft pass rate, autofix / repair / rewrite use, runs failed with `citations_failed`, issue kinds |

`HeuristicJudge` is a deterministic, model-free stand-in (key numbers found in the answer; numbers
and content words shared by a claim and its finding). It is for the offline mode only.

## Running

```bash
# smoke subset, real models, fixture search, $0.20 cap per run (~$0.45 + ~$0.04 judging)
backend/.venv/bin/python -m evals.run_eval --tags smoke --budget 0.20

# representative numbers: live PubMed / Europe PMC (needs network; not reachable from the dev sandbox)
SEARCH_BACKEND=live backend/.venv/bin/python -m evals.run_eval --budget 0.50 --concurrency 3

# re-score cached runs only, and apply the floors
backend/.venv/bin/python -m evals.run_eval --tags smoke --cached-only --gate

# deterministic offline mode: FakeLLM pipeline + heuristic judges (no network, no spend)
backend/.venv/bin/python -m evals.run_eval --llm fake --gate

# cost-quality frontier: the same items at several caps -> frontier.json + frontier.png
backend/.venv/bin/python -m evals.frontier --tags smoke --budgets 0.05,0.20,1.00

# the writer alone on cached runs: first-draft citation-gate pass rate for a prompt change
backend/.venv/bin/python -m evals.writer_replay --tag fixture-anthropic-usd0.2 --samples 2 [--prompt-rev HEAD~1]
```

Options: `--ids`, `--tags`, `--sample N --seed S`, `--budget`, `--max-sub-questions`,
`--search-backend fixture|live`, `--llm anthropic|fake`, `--judge llm|heuristic|none`,
`--concurrency`, `--max-total-usd` (stop starting runs past this spend), `--rerun`, `--cached-only`,
`--tag`, `--out-dir`, `--name`, `--gate`.

Runs go through `graph.Runner` with in-memory stores and topic-findings reuse off, and are cached
as full `RunResult`s under `evals/.cache/runs/<tag>/<item>.json` (tag = backend, LLM, budget);
judge calls are cached under `evals/.cache/judge/` by a hash of the request. Recomputing a report
costs nothing; change the tag or pass `--rerun` to run again. Reports are written as
`<name>.json` (for the gate, the frontier and the UI) and `<name>.md`.

## Search backend: fixture vs live

The harness is backend-agnostic. With `SEARCH_BACKEND=fixture` (the default) research reads the
19 synthetic records in `backend/tools/fixtures/studies.json` (five topics, `FIXTURE-*` ids), so:

- **coverage is always 0** (no real PMIDs or DOIs), and agreement is only meaningful on the
  `fixture_topic` items, whose fixture records mirror the real literature;
- questions on other topics end with `insufficient_evidence`.

Fixture runs prove the pipeline end to end and measure cost, latency, scope, calibration
mechanics and the citation gate. **Agreement and coverage numbers worth quoting need
`SEARCH_BACKEND=live` with network access** (CI runners have it). Reports say which they are
(`representative`, and a banner in the Markdown).

## CI gate (`gate.py`, `thresholds.json`, `../.github/workflows/ci.yml`)

`gate.check(report, thresholds)` compares dotted aggregate paths with `min` floors and `max`
ceilings for the report's profile: `offline` (`--llm fake`), `fixture` or `live`. A metric that
was not measured fails its floor. Agreement and faithfulness have floors in every profile.

- Every push and PR: the offline pytest suite (with a Postgres service for the store tests), then
  the offline eval gate (`--llm fake`), whose floors are a deterministic regression check of the
  harness and pipeline. `backend/tests/test_evals.py` tests the gate logic itself, including that
  a writer regression fails it and that the Phase 6 behaviour fails the fixture floors.
- Push to `main` and manual dispatch: when the `ANTHROPIC_API_KEY` secret is set, the `ci` subset
  (4 items, ~$0.5) runs with real models against live search (dispatch can pick fixture, tags or
  `all`, and the budget), and the job fails below the floors. The live floors are initial guesses;
  raise them once a live baseline exists.

## Committed results (`reports/fixture-smoke/`)

**Fixture backend, not representative of answer quality.** A real-model smoke eval run on
2026-10-19 (Sonnet / Haiku / Opus 5.5) on the `smoke` subset, plus its cached runs and judgments
(`cache/`, re-scored offline by `test_evals.py`):

| | before the fix (`bbc8daa`) | after the fix |
|---|---|---|
| In-scope answers | 1 of 4 (3 × `citations_failed`) | 4 of 4 |
| First-draft gate pass (end-to-end runs) | 0 of 4 | 2 of 4 |
| First-draft gate pass (writer replay, same inputs) | 0 of 4 (0 of 8 with the runs) | 6 of 8 |
| Escalated (Opus) rewrites | 2 of 4 | 0 of 4 (2 citation repairs on Sonnet, both passed) |
| Mean cost per in-scope run | $0.136 | $0.103 |
| Agreement (strict) / faithfulness | 0.25 / 0.67 | 0.69 / 0.66 |
| Fixture gate | fails (agreement, answer rate) | passes |

Frontier (`frontier.json`, `frontier.png`, same 6 items): at a $0.05 cap no in-scope run can
afford synthesis plus a repair (mean spend $0.042, all fail cleanly with `budget_exhausted`); at
$0.20 and $1.00 every in-scope item is answered (mean $0.103 / $0.112, agreement 0.69 / 0.69,
faithfulness 0.66 / 0.74). The $1.00 cap buys little on the small fixture library.

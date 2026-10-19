# Evidence Coach backend

Python 3.12+. The package is imported as `backend` (e.g. `from backend.schemas import Finding`).

## Install

```bash
cd backend
uv venv .venv -p 3.13            # or: python -m venv .venv
uv pip install -p .venv -e ".[dev]"
```

The API key is read from `ANTHROPIC_API_KEY`, falling back to `ANTROPIC_API_KEY`.

## Test

```bash
cd backend
.venv/bin/python -m pytest -q                                  # offline, deterministic (FakeLLM + fixtures)
RUN_LIVE=1 .venv/bin/python -m pytest -m live -s tests/live    # real API, ~$0.30 in all (synthesis runs ~$0.05-0.11 each)
RUN_LIVE_NETWORK=1 .venv/bin/python -m pytest -m live_network -s   # real PubMed / Europe PMC / Crossref, no LLM spend
TEST_DATABASE_URL=postgresql://... .venv/bin/python -m pytest -m postgres   # store, resume and API tests on a real Postgres
```

Tests marked `postgres` run the same store and resume tests as the in-memory default on a real
database. They apply the migrations and **truncate the tables before each test**, so point
`TEST_DATABASE_URL` at a throwaway database, never at Supabase production. A local one:

```bash
sudo apt-get install -y postgresql && sudo service postgresql start
sudo -u postgres psql -c "alter user postgres password 'postgres'" -c "create database evidence_test"
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/evidence_test .venv/bin/python -m pytest -q
```

## Database (Supabase Postgres)

Run state, progress events, the topic findings store, the LLM call ledger, conversations and
profiles live in Postgres (Supabase). Without `DATABASE_URL` everything falls back to in-memory
stores (`store.build_stores()`), which is what the default tests use.

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | unset | Supabase connection string with the **service role** (postgres user), e.g. the transaction pooler `postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require` |
| `DATABASE_POOL_MIN` / `DATABASE_POOL_MAX` | 1 / 5 | psycopg pool size per process (keep small on Vercel: one pool per function instance) |
| `DATABASE_CONNECT_TIMEOUT_S` | 15 | wait for the first connection |

Set up a Supabase project:

1. Apply the migrations in `../supabase/migrations/`: `supabase link --project-ref <ref>` then
   `supabase db push` (or paste the SQL into the SQL editor, or
   `python -m backend.store.migrate "$DATABASE_URL"` from the repo root, which records the version
   in the CLI's `supabase_migrations.schema_migrations` table).
2. Copy the connection string (Project Settings -> Database -> Connection string). On Vercel use
   the **transaction pooler** (port 6543); the direct connection (5432) is fine for long-lived
   processes. Set it as `DATABASE_URL` in Vercel's environment variables.

Notes:
- The pooler does not support server-side prepared statements, so every connection uses
  `prepare_threshold=None` and autocommit (LangGraph's `PostgresSaver.from_conn_string` prepares
  every statement and fails behind a transaction pooler with `prepared statement "_pg3_0" already
  exists`; tested here with pgbouncer in transaction mode).
- RLS is enabled on every table. The backend's service role bypasses it; signed-in users may only
  read their own runs, events, conversations and profile (policies use `auth.uid()`), and the
  checkpoint, ledger and topic-findings tables have no policies at all.
- The migration creates LangGraph's checkpoint tables and marks its migrations as applied, so
  `PostgresSaver.setup()` is not needed (and only adds migrations from newer library versions).

## Search backend

`SEARCH_BACKEND=fixture` (default, used by tests) serves synthetic records from `tools/fixtures/studies.json`.
`SEARCH_BACKEND=live` queries PubMed + Europe PMC, checks Crossref / Retraction Watch and reads open-access full text.

| Variable | Default | Meaning |
|---|---|---|
| `SEARCH_BACKEND` | `fixture` | `fixture` or `live` |
| `NCBI_API_KEY` | unset | raises the PubMed limit from 3 to 10 requests/second |
| `SEARCH_CONTACT_EMAIL` | unset | sent as `email=` (NCBI, Europe PMC) and `mailto=` (Crossref polite pool) |
| `SEARCH_INCLUDE_PREPRINTS` | off | add SportRxiv (OSF API) as a labeled preprint source |
| `SEARCH_TIMEOUT_S` / `SEARCH_MAX_RETRIES` / `SEARCH_TIME_BUDGET_S` | 15 / 3 / 60 | per request / per request / per `search()` call |

## Run a question through the graph

```python
from backend.graph import Budget, GraphDeps, memory_checkpointer, run_research
from backend.llm import AnthropicLLM
from backend.tools.factory import build_search_tool

result = run_research(
    "How many sets per week for hypertrophy?",
    deps=GraphDeps(llm=AnthropicLLM(), search_tool=build_search_tool()),  # SEARCH_BACKEND=fixture|live
    budget=Budget(max_usd=0.20),
    checkpointer=memory_checkpointer(),
)
result.outcome   # answered | degraded | referred | failed, with structured reasons; never raises
result.answer    # CoachAnswer or None
result.position_map, result.debate, result.verdict, result.downweighted   # synthesis records
result.trace     # one TraceStep per node: model, tokens, cost, latency, tool calls, decision
```

### Persistent runs: resume, plan approval, reuse

```python
from backend.graph import GraphDeps, PlanDecision, Runner
from backend.store import build_stores   # Postgres if DATABASE_URL is set, else in-memory

runner = Runner(GraphDeps(llm=AnthropicLLM(), search_tool=build_search_tool()), build_stores())
r = runner.start(question, require_plan_approval=True)   # pauses after the planner
r.awaiting_approval.sub_questions                         # PlanApprovalRequest for the UI
r = runner.resume(r.run_id, PlanDecision(action="approve"))   # or action="edit", sub_questions=[...]
runner.resume(run_id)          # after a crash / redeploy: continues from the last checkpoint
runner.get(run_id)             # current RunResult from the checkpoint
```

- State is checkpointed after every node (durability "sync"); `resume` re-runs only unfinished
  nodes, and parallel branches that finished before the crash are kept (LangGraph pending writes).
- LLM calls go through a ledger keyed by (run id, hash of the request). A step retried after a
  crash gets its recorded results back (`CallRecord.replayed`) instead of calling the API again,
  so state spend equals what was actually paid, once.
- Researchers check the topic findings store first (normalized topic key, Jaccard similarity >= 0.5,
  30-day TTL) and reuse fresh findings with no search or model call; the trace step shows a
  `findings_store` tool call (`reused` / `saved`). `Runner(..., reuse_findings=False)` turns it off
  (e.g. for evals).
- The plan checkpoint is a LangGraph `interrupt`; the wait is excluded from the wall-clock budget
  (`started_at` moves forward by `waited_s`). Without `require_plan_approval` the plan is
  auto-approved and the node is skipped. It needs a checkpointer.
- The runs table carries status (`running`, `awaiting_approval`, `completed`, `error`) and the
  outcome; `run_events` gets one event per trace step (gap-free `seq`, idempotent by step id) for
  the SSE endpoint to poll.

## HTTP API (Phase 6)

```bash
# from the repo root; loads backend/.env, uses Postgres if DATABASE_URL is set, else memory
backend/.venv/bin/uvicorn --factory backend.api:create_app --port 8000
```

| Endpoint | What it does |
|---|---|
| `POST /research` | `{question, profile?, conversation_id?, budget?, require_plan_approval?}` -> `202 {run_id, conversation_id, links}` at once; the run is handed to the executor. No profile: the caller's saved profile. No conversation: a new one titled after the question. |
| `GET /research/{id}` | `status` (`running`, `awaiting_approval` + `plan`, `completed`, `error`), then `answer` (`CoachAnswer`), `outcome`, `citations` (the gate's report), `spend`. A fork also has `fork` and, once completed, `diff`. |
| `POST /research/{id}/plan` | `{"action": "approve"}` or `{"action": "edit", "sub_questions": [{"id": "sq1", "question": ...}, {"question": "new"}]}`. Validated at once (422), then queued on the run (`pending_decision`) and the run resumes. 409 if the run is not waiting. |
| `GET /research/{id}/events` | SSE. Polls `run_events` for `seq > last seen`; `id:` is the seq, so `Last-Event-ID` (or `?after=N`) resumes. Events: `run_started`, `step` (`message` such as "Planned 5 sub-questions", "PubMed + Europe PMC: 23 studies → 9 kept · sq2", "Debate: round 1 · P2 opening", "Citations: 18 claims and 31 numbers backed", plus running `totals` {tokens, cost_usd}), `plan_ready`, `plan_decided`, `run_resumed`, `run_finished`, `run_error`, and a final `end` (close the EventSource). A response lasts at most `SSE_MAX_S`; EventSource reconnects by itself. |
| `GET /research/{id}/trace` | Every `TraceStep`: node, agent, input summary, output (JSON), model, LLM calls (tokens, cost, latency, replayed), tool calls, decision, status; plus totals. Fork traces start with the parent's steps (`inherited_steps`, their `run_id` is the parent's). |
| `POST /research/{id}/fork?from_step=N` | Body: `{"prompt_addendum"?, "model"?, "tool_result"?: [StudyRecord]}`. Re-runs step N with the override as a **new run** and returns its id (202). See "Time-travel" below. |
| `GET /conversations`, `GET/DELETE /conversations/{id}` | The caller's conversations; a conversation with its runs (oldest first). |
| `GET /profile`, `PUT /profile` | The caller's `UserProfile` (needs a user). |
| `POST /internal/runs/{id}/advance` | Worker: one slice under the run's lease (`Authorization: Bearer $WORKER_SECRET`). |
| `GET /internal/cron/reap` | Reaper (Vercel cron, `Bearer $CRON_SECRET` or the worker secret). |

**Auth (minimal).** `X-User-Id` header by default; with `SUPABASE_JWT_SECRET` set, a Supabase access
token (`Authorization: Bearer`, HS256, audience `authenticated`, user id = `sub`) and the header is
ignored. No credentials = anonymous: anonymous runs are readable by anyone holding the run id; a
user's runs, conversations and profile only by that user (others get 404).
TODO: JWKS verification for Supabase projects on asymmetric signing keys; rate limiting.

### Citation validation (the answer gate)

The graph runs `validate_citations` after every writer draft (`writer -> validate_citations ->
check_answer`). It is deterministic code (`backend/citations.py`):

- The writer cites inline with finding ids (`... 0.37% per set [sq1:PMID:123].`); code renumbers
  them to `[n]` markers, and `[n]` points at `answer.cited_finding_ids[n-1]` (the UI's chips).
  Invented ids are dropped, so their sentence shows up as uncited.
- Claims: every sentence of `recommendation`, `why` and `whats_new` needs a citation; in
  `practical_protocol`, `applies_to_you` and `what_would_change_this` any sentence with a number does.
- Numbers: digits (`12`, `0.37`, `1,863`, ranges `10-20` / `10 to 20` endpoint by endpoint, signs
  ignored) and spelled numbers before a unit (`ten sets`). Each must appear in a finding cited in
  the same sentence (claim, effect, snippet, population, title, sample size or year; spelled numbers
  in findings count). Decimals may round the finding's value (`1.6` matches `1.62`); integers must
  match exactly. Numbers from the user's profile or question need no finding. Position stances are
  checked against their `supporting_findings`, counterarguments against any finding the answer cites.
  Labels (`P1`, `position 2`), ordinals and `1RM` are not quantities.
- On failure the problems go back to the writer as feedback through the run's single send-back,
  escalated to Opus. If the send-back is already used (a research re-pass) or the rewrite fails
  again, the run fails cleanly: `outcome.failure.reason = "citations_failed"`, no answer is
  returned, the findings and the report (`GET /research/{id}` -> `citations.issues`) are.

### Time-travel forks

`POST /research/{id}/fork?from_step=N` (a finished run; N must be a step of a model-calling agent:
intake, planner, researcher, position_mapper, advocate, judge, writer; `tool_result` only for a
researcher):

1. The checkpoint before step N is found in the run's LangGraph history (the latest one whose trace
   has at most N steps) and copied to a new thread with `update_state(as_node=<the node that wrote
   it>)`, so the same edges (including `Send` fan-outs) fire on the new run. Run ids in the state
   and pending tasks are rewritten; the wall clock starts afresh; spend carries over; the plan
   checkpoint is not asked again.
2. The parent's LLM call ledger is copied to the fork: unchanged requests replay their recorded
   output for free (parallel siblings of step N, and anything downstream whose input did not change).
3. The override (stored on the fork's run record, `runs.fork`) applies to step N's agent only, and
   to its sub-question (researcher) or position (advocate, `TraceStep.position_id`): a prompt
   addendum is appended to the system prompt, a model replaces the routed one, a tool result is what
   every search of that researcher returns (`search:override`). Forks never write to the topic
   findings store.
4. When the fork completes, `GET /research/{fork}` includes `diff`: outcome, status, recommendation,
   confidence changes, positions added / removed / changed (matched by stance, then by overlapping
   findings), citations added / removed, and costs (`new_cost_usd` = what the fork actually paid).

### Execution model (Vercel)

Runs advance in **slices** through an `Executor` (`backend/api/executor.py`):

- `POST /research` (and `/plan`, `/fork`) calls `executor.submit(run_id)` and returns at once.
- A slice (`Worker.advance`): take the run's **lease** (one conditional `UPDATE ... where lease is
  free or expired`, so of two advancers only one wins; the other gets `busy`), run the graph from its
  checkpoint, renew the lease at every superstep boundary, stop at the first boundary after
  `ADVANCE_SLICE_S` (nothing in flight; the checkpoint is written), release the lease. Not finished:
  the slice submits the next one.
- `EXECUTOR=http` (default on Vercel): `submit` POSTs `/internal/runs/{id}/advance` to the
  deployment itself and waits only ~3s for the request to be accepted, so every slice is its own
  function invocation with its own time limit. `EXECUTOR=thread` (default locally) runs the slices
  in a thread pool in-process; `inline` runs them in the request (tests).
- **Reaper** (`/internal/cron/reap`, Vercel cron every minute): runs left `running` with an expired
  lease (the worker died or hit `maxDuration` mid-slice), runs never picked up (`running`, no lease,
  no progress for `REAP_STALE_S`), and `error` runs with fewer than `MAX_AUTO_RETRIES` failed
  advances are dispatched again. They resume from the last checkpoint; the ledger makes re-run steps
  free.
- Timing: a full run is ~1.5-3 minutes of work (fixture search: ~95s); `maxDuration` is 300s and a slice stops after 180s at
  the next superstep boundary (a superstep of parallel researchers can take ~90s), and the lease
  (330s) outlives the function. Migration `20261020000000_run_execution.sql` adds the lease, budget,
  pending decision and fork columns.

Caveats: the http hand-off relies on Vercel letting an invocation finish after its caller stopped
waiting (the reaper is the backstop if one is dropped). Preview deployments behind Deployment
Protection need "Protection Bypass for Automation" (`VERCEL_AUTOMATION_BYPASS_SECRET` is sent
automatically) or `WORKER_BASE_URL` pointing at an unprotected domain. Per-minute crons need a Pro
plan (Hobby allows daily: change the schedule in `vercel.json`). A queue (Vercel Queues, QStash)
could replace the self-POST later behind the same `Executor` interface.

| Variable | Default | Meaning |
|---|---|---|
| `EXECUTOR` | `http` on Vercel, else `thread` | `thread`, `http` or `inline` |
| `WORKER_SECRET` | unset | Bearer secret for `/internal/*` (required for `http`; unset disables the routes) |
| `CRON_SECRET` | unset | Vercel sends it to cron routes; also accepted by the reaper |
| `WORKER_BASE_URL` | `https://$VERCEL_PROJECT_PRODUCTION_URL` (production) or `https://$VERCEL_URL` | where `http` sends slices |
| `ADVANCE_SLICE_S` / `LEASE_TTL_S` | 180 / 330 | slice length (stop at the next superstep boundary) / lease lifetime |
| `REAP_STALE_S` / `MAX_AUTO_RETRIES` | 60 / 3 | reaper thresholds |
| `SSE_POLL_S` / `SSE_MAX_S` | 1 / 240 | event polling interval / max length of one SSE response |
| `AUTH_MODE` / `SUPABASE_JWT_SECRET` | `header` / unset | `jwt` when the secret is set |
| `CORS_ORIGINS` | unset | comma-separated origins (not needed when the frontend proxies `/api`) |
| `LOG_LEVEL` | `INFO` | |

### Deploying on Vercel (monorepo)

Two Vercel projects from this one repo:

- **Backend** (this): Root Directory = the repo root. `api/index.py` is the only function (the whole
  FastAPI app); `vercel.json` rewrites every path to it, sets `maxDuration: 300`, excludes
  `frontend/`, `evals/`, tests and venvs from the bundle, and schedules the reaper cron.
  `requirements.txt` (repo root) lists the runtime dependencies; `.python-version` pins 3.12.
  Environment: `ANTHROPIC_API_KEY`, `DATABASE_URL` (Supabase transaction pooler), `WORKER_SECRET`,
  `CRON_SECRET`, `SEARCH_BACKEND=live`, `NCBI_API_KEY`, `SEARCH_CONTACT_EMAIL`, optionally
  `SUPABASE_JWT_SECRET`.
- **Frontend** (Phase 8): Root Directory = `frontend/`, a Next.js project with a rewrite
  `/api/:path*` -> `${BACKEND_URL}/:path*`, so the browser talks to its own origin (no CORS, cookies
  and the Supabase session stay first-party) and EventSource streams through the rewrite. Keeping
  them separate keeps the Python function's bundle and cold start free of Node, and each deploys
  on its own.

### Local smoke run (real model)

```bash
createdb evidence_smoke && backend/.venv/bin/python -m backend.store.migrate postgresql://.../evidence_smoke
DATABASE_URL=postgresql://.../evidence_smoke EXECUTOR=http WORKER_SECRET=dev WORKER_BASE_URL=http://127.0.0.1:8765 \
  ADVANCE_SLICE_S=20 SEARCH_BACKEND=fixture backend/.venv/bin/uvicorn --factory backend.api:create_app --port 8765
curl -X POST localhost:8765/research -H 'Content-Type: application/json' -H 'X-User-Id: me' \
  -d '{"question": "How many sets per week for hypertrophy?", "require_plan_approval": true, "budget": {"max_usd": 0.25}}'
curl -N localhost:8765/research/<run_id>/events -H 'X-User-Id: me'
```

Run on 2026-10-19 (Sonnet / Haiku / Opus 5.5, fixture search, Postgres, http executor): plan edited
from 6 to 3 sub-questions, 18 steps, ~93s of work, $0.134. The first draft failed the citation gate
(an uncited sentence, an uncited "8 weeks", "10 sets" cited to a finding that does not say it); the
Opus rewrite passed (18 claims, 31 numbers). A fork from the writer step with a prompt addendum ran
in 7 one-superstep slices through `/internal/runs/{id}/advance` and changed only the recommendation
($0.083 new spend).

## Layout

| Path | What it is |
|---|---|
| `schemas.py` | Pydantic models for every handoff (Intake, ResearchPlan, Finding, SearchStatus, Position, Argument, Verdict, CoachAnswer, ...) |
| `config.py` | Model IDs per agent, effort, prices, API key lookup |
| `llm.py` | `LLMClient` protocol, `AnthropicLLM` (structured output + per-call tokens/cost/latency `CallRecord`s), `FakeLLM` for tests |
| `agents/intake.py` | Rule screen + model scope classifier, profile extraction, fixed referral messages |
| `agents/planner.py` | 3-6 concrete sub-questions with search terms; `lint_sub_question` flags vague ones |
| `agents/researcher.py` | Search (capped) -> dedupe / drop retracted -> one extraction call -> graded `Finding`s + `SearchStatus` |
| `agents/grading.py` | Deterministic GRADE-style grade for one finding, with reasons |
| `agents/evidence.py` | Deterministic synthesis rules: body-of-evidence grade, false-balance status rule, confidence cap, finding rendering |
| `agents/mapper.py` | Position Mapper: findings -> `PositionMap` (stances from the model; ids, grades and status from code) |
| `agents/advocate.py` | Advocate: one position, one round -> `DebateTurn`; claims without valid finding ids are rejected |
| `agents/judge.py` | Judge: positions + debate + graded findings -> `Verdict`; code caps confidence and status |
| `agents/writer.py` | Findings (+ optional PositionMap / Verdict) -> `CoachAnswer`; strips unknown ids, adds disclaimer |
| `graph/builder.py` | LangGraph `StateGraph`: intake -> (referral) -> planner -> dispatch -> researchers (Send fan-out) -> review -> personalize -> position_mapper -> (advocates x2 rounds) -> judge -> writer -> validate_citations -> check_answer -> finalize; `run_research` |
| `graph/synthesis.py` | Personalize, position mapper, advocate, debate review and judge nodes; per-stage budget checks with reserves |
| `graph/state.py` | `RunState` (typed, reducers for parallel writes), research records, reviews, degradations, `RunOutcome` |
| `graph/nodes.py` | One node per agent / supervisor decision; meters spend, appends one trace step, never raises |
| `graph/supervisor.py` | Deterministic review: answered? animal / in-vitro only? population match? another pass? Answer review |
| `graph/budget.py` | `Budget` (per run), `Spend`, `MeteredLLM` / `MeteredSearch` that enforce allowances and deadlines |
| `graph/routing.py` | `ModelRouter`: Haiku extraction, Sonnet planning / mapping / debate / judging / writing, Opus only on escalation |
| `graph/trace.py` | `TraceStep` / `ToolCall` schema and the step-numbering reducer |
| `graph/checkpoint.py` | In-memory checkpointer and the serializer allowlist (also used by the Postgres saver) |
| `graph/plan_review.py` | Plan checkpoint payloads: `PlanApprovalRequest` (interrupt value), `PlanDecision` (approve / edit) |
| `graph/runner.py` | `Runner`: create / start / advance (sliced, with stop + heartbeat hooks) / resume / get / fork runs on the stores; runs table, step events with UI messages and running totals |
| `graph/progress.py` | One UI line per trace step ("Planned 5 sub-questions", "PubMed: 23 studies → 9 kept") for the SSE stream |
| `graph/fork.py` | Time-travel: fork point in the checkpoint history, state copy for the new run, `AnswerDiff` |
| `graph/overrides.py` | `StepOverride` (prompt addendum / model / tool result) scoped to the forked step; `StaticSearchTool` |
| `citations.py` | Citation gate: sentence split, number extraction and matching, `[id]` -> `[n]` numbering, `validate_answer` -> `CitationReport` |
| `api/app.py` | FastAPI app factory and routes (`create_app(Services)`; `create_app()` builds from the environment) |
| `api/sse.py` | SSE generator polling `run_events` (Last-Event-ID, keep-alives, `end`) |
| `api/worker.py` | `Worker`: one slice under a lease; the reaper query |
| `api/executor.py` | `Executor` interface: `ThreadExecutor`, `HttpExecutor` (Vercel self-dispatch), `InlineExecutor` |
| `api/auth.py`, `api/settings.py`, `api/models.py` | X-User-Id / Supabase JWT auth, env settings, request / response bodies |
| `../api/index.py`, `../vercel.json`, `../requirements.txt` | Vercel Python function entrypoint, rewrites + maxDuration + cron, runtime dependencies |
| `store/base.py` | Store interfaces (runs, events, findings, call ledger, conversations, profiles), `Stores`, `call_key` |
| `store/memory.py` | In-memory stores (tests, local) |
| `store/postgres.py` | Postgres stores + `PostgresSaver` on one pooler-safe psycopg pool |
| `store/topics.py` | Topic keys (normalized content words) and Jaccard similarity for the findings store |
| `store/migrate.py` | Applies `supabase/migrations/*.sql` to a Postgres (local / tests); shim for Supabase roles |
| `tools/search.py` | `SearchTool` protocol and `FixtureSearchTool` (synthetic records in `tools/fixtures/studies.json`) |
| `tools/composite.py` | Live `SearchTool`: sources in parallel -> dedupe -> rank -> caps -> retraction check -> full text |
| `tools/pubmed.py` | E-utilities esearch + efetch XML: publication types, MeSH, structured abstract, DOI/PMCID, RetractionIn |
| `tools/europepmc.py` | Europe PMC search (OA flag, PMCID, preprints) and `fullTextXML` fetch |
| `tools/jats.py` | JATS section parser: methods, results, funding / competing interests; drops intro, discussion, tables |
| `tools/retraction.py` | Crossref / Retraction Watch: `updated-by`, `relation`, then `filter=updates:{doi}` notices |
| `tools/sportrxiv.py` | Optional SportRxiv preprints (OSF API), always labeled preprint; withdrawn = retracted |
| `tools/ranking.py` | DOI/PMID/PMCID dedupe + merge, recency/design ranking, per-journal and per-last-author caps |
| `tools/http.py` | Shared httpx client: per-host rate limits (thread-safe), 429/5xx backoff with Retry-After, retry cap |
| `tools/factory.py` | `build_search_tool()` from `SearchSettings` (env) |
| `tests/` | One test file per agent; canned model outputs in `tests/fixtures/`; live checks in `tests/live/`; recorded-format API fixtures in `tests/fixtures/live/` served by `httpx.MockTransport` (`tests/live_fakes.py`) |

## Design notes

- Structured output uses `output_config.format` (JSON schema) rather than a forced tool call:
  Sonnet 5.5 and Opus 5.5 reject forced `tool_choice`. Output is re-validated with Pydantic.
- The model fills "draft" models; code turns drafts into records and owns ids, provenance
  (source id/URL/date come from the search record), grades, referral text and the disclaimer.
- Budgets are enforced in the graph: the plan is cut to `max_sub_questions`; research gets the
  budget minus a writer reserve, split between parallel researchers (only as many as it can fund);
  every LLM call is admitted against its node's allowance with `max_tokens` lowered to fit;
  searches are capped per sub-question and stop at the deadline. Exhaustion degrades (answer with
  what was found, recorded in `outcome.degradations`) or fails with `failure.reason="budget_exhausted"`.
- The supervisor sends work back at most once per run (`MAX_SEND_BACKS`, counted in state). That one
  send-back is either a research re-pass (escalated to Opus unless it only retries failed searches)
  or an Opus rewrite after the writer fails or its answer fails review.
- Expected search failures (no results, rate limited, paywalled, timed out, retracted, tool error,
  extraction failure) come back as `SearchStatus.state`, never as exceptions. The supervisor retries
  transient ones (rate limited / timed out / error) on the same model, searches again for
  `retracted`, and accepts `paywalled` (abstract-only) findings with an `abstract_only_evidence`
  degradation.
- Provenance: every `StudyRecord` carries source, endpoint, URL, retrieval time and whether its text
  is full text or abstract; the researcher copies that into `Finding.provenance`. Findings without
  complete provenance are rejected in the researcher node (`provenance_rejected` degradation) and
  filtered again by `all_findings`, so they never reach the writer.
- Recency: queries keep the last 10 years plus older meta-analyses / systematic reviews (date filter
  in the query); ranking then favours the last 5 years, syntheses and RCTs, and full-text availability.
- Synthesis (Phase 4): `personalize` down-weights findings whose study population differs from the
  user (sex, training status, age via `supervisor.population_mismatch`; animal / in-vitro = non-human)
  and records which and why in its trace step; down-weighted findings count half and one GRADE level
  lower, but are never dropped. The mapper's model only groups findings into stances; code grades each
  position from its findings and sets the status: dissent with only low / very-low evidence against a
  stronger lead is set aside (`consensus`, no debate), a credible but weaker rival gives `leaning`, an
  equal-grade rival with comparable support gives `genuinely_contested`. The model can move the status
  towards consensus, never away. Leaning / contested maps get one advocate per position: openings in
  parallel, then exactly one rebuttal round (`DEBATE_ROUNDS`, enforced by routing). The judge's
  confidence is capped by the best supporting grade (contested: at most moderate) and its recommended
  position must rest on a peer-reviewed human finding.
- Synthesis budgets: before each stage the node estimates its cost and time and keeps a reserve for the
  judge and the writer; the debate runs only if both rounds fit (else `debate_skipped`). Synthesis never
  fails a run: no map -> the writer works from findings (`mapping_skipped`), no verdict -> from the map
  (`judge_skipped`).

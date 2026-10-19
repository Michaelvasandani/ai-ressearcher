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
TEST_DATABASE_URL=postgresql://... .venv/bin/python -m pytest -m postgres   # store + resume tests on a real Postgres
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
| `graph/builder.py` | LangGraph `StateGraph`: intake -> (referral) -> planner -> dispatch -> researchers (Send fan-out) -> review -> personalize -> position_mapper -> (advocates x2 rounds) -> judge -> writer -> check_answer -> finalize; `run_research` |
| `graph/synthesis.py` | Personalize, position mapper, advocate, debate review and judge nodes; per-stage budget checks with reserves |
| `graph/state.py` | `RunState` (typed, reducers for parallel writes), research records, reviews, degradations, `RunOutcome` |
| `graph/nodes.py` | One node per agent / supervisor decision; meters spend, appends one trace step, never raises |
| `graph/supervisor.py` | Deterministic review: answered? animal / in-vitro only? population match? another pass? Answer review |
| `graph/budget.py` | `Budget` (per run), `Spend`, `MeteredLLM` / `MeteredSearch` that enforce allowances and deadlines |
| `graph/routing.py` | `ModelRouter`: Haiku extraction, Sonnet planning / mapping / debate / judging / writing, Opus only on escalation |
| `graph/trace.py` | `TraceStep` / `ToolCall` schema and the step-numbering reducer |
| `graph/checkpoint.py` | In-memory checkpointer and the serializer allowlist (also used by the Postgres saver) |
| `graph/plan_review.py` | Plan checkpoint payloads: `PlanApprovalRequest` (interrupt value), `PlanDecision` (approve / edit) |
| `graph/runner.py` | `Runner`: start / resume / get runs on the stores; runs table, step events, plan-ready events |
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

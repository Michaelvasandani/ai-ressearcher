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
RUN_LIVE=1 .venv/bin/python -m pytest -m live -s tests/live    # real API, a few cents (graph run ~$0.04)
```

## Run a question through the graph

```python
from backend.graph import Budget, GraphDeps, memory_checkpointer, run_research
from backend.llm import AnthropicLLM
from backend.tools.search import FixtureSearchTool

result = run_research(
    "How many sets per week for hypertrophy?",
    deps=GraphDeps(llm=AnthropicLLM(), search_tool=FixtureSearchTool.from_file()),
    budget=Budget(max_usd=0.20),
    checkpointer=memory_checkpointer(),
)
result.outcome   # answered | degraded | referred | failed, with structured reasons; never raises
result.answer    # CoachAnswer or None
result.trace     # one TraceStep per node: model, tokens, cost, latency, tool calls, decision
```

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
| `agents/writer.py` | Findings (+ optional PositionMap / Verdict) -> `CoachAnswer`; strips unknown ids, adds disclaimer |
| `graph/builder.py` | LangGraph `StateGraph`: intake -> (referral) -> planner -> dispatch -> researchers (Send fan-out) -> review -> writer -> check_answer -> finalize; `run_research` |
| `graph/state.py` | `RunState` (typed, reducers for parallel writes), research records, reviews, degradations, `RunOutcome` |
| `graph/nodes.py` | One node per agent / supervisor decision; meters spend, appends one trace step, never raises |
| `graph/supervisor.py` | Deterministic review: answered? animal / in-vitro only? population match? another pass? Answer review |
| `graph/budget.py` | `Budget` (per run), `Spend`, `MeteredLLM` / `MeteredSearch` that enforce allowances and deadlines |
| `graph/routing.py` | `ModelRouter`: Haiku extraction, Sonnet planning / writing, Opus only on escalation |
| `graph/trace.py` | `TraceStep` / `ToolCall` schema and the step-numbering reducer |
| `graph/checkpoint.py` | In-memory checkpointer and the serializer allowlist (Phase 5 plugs in Postgres) |
| `tools/search.py` | `SearchTool` protocol and `FixtureSearchTool` (synthetic records in `tools/fixtures/studies.json`) |
| `tests/` | One test file per agent; canned model outputs in `tests/fixtures/`; live checks in `tests/live/` |

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
- Expected search failures (no results, rate limited, paywalled, timed out, tool error, extraction
  failure) come back as `SearchStatus.state`, never as exceptions.

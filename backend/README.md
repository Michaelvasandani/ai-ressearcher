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
RUN_LIVE=1 .venv/bin/python -m pytest -m live -s tests/live    # real API, a few cents
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
| `tools/search.py` | `SearchTool` protocol and `FixtureSearchTool` (synthetic records in `tools/fixtures/studies.json`) |
| `tests/` | One test file per agent; canned model outputs in `tests/fixtures/`; live checks in `tests/live/` |

## Design notes

- Structured output uses `output_config.format` (JSON schema) rather than a forced tool call:
  Sonnet 5.5 and Opus 5.5 reject forced `tool_choice`. Output is re-validated with Pydantic.
- The model fills "draft" models; code turns drafts into records and owns ids, provenance
  (source id/URL/date come from the search record), grades, referral text and the disclaimer.
- Expected search failures (no results, rate limited, paywalled, timed out, tool error, extraction
  failure) come back as `SearchStatus.state`, never as exceptions.

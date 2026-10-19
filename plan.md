# Evidence Coach

**A science-based health and fitness coach that shows its work.**

Ask a training or nutrition question, such as "How many sets per week for hypertrophy?" or "How much protein do I need on a cut?" The coach reads the latest research (several studies, not just one paper), maps where the evidence agrees and where it splits, argues both sides using only cited findings, and gives a recommendation with a confidence level. When the science is contested, it shows both sides so the user can make the final call.

Under the hood, it is a production-grade multi-agent system. Its state survives restarts, it caps its own spending, and every answer has a step-by-step trace.

**Stack:** Python · LangGraph · Claude API · PubMed / Europe PMC · Redis · FastAPI · Next.js · Tailwind
**Estimated build time:** 5–6 weeks part time (the base plan was 3–4 weeks; the debate stage, the evals and the chat UI add about two)

---

## Why this project signals

Interviewers have seen hundreds of agent demos. Almost none of them survive a restart, cap their own spending, or explain what they did. This project's differentiators:

1. **Durable state, budgets and step-level tracing.** These are what a production agent has and a tutorial agent doesn't.
2. **Evidence grading and a structured debate.** The coach doesn't summarize a few links. It weighs evidence the way a systematic reviewer would.
3. **Measured correctness.** Answers are scored against expert position statements and meta-analyses, and a CI gate fails the build if quality drops.
4. **Time-travel debugging.** Open the trace of a bad answer, fork from the step that went wrong, fix it, and diff the result.

**Interview pitch:** "It reads the latest studies, argues both sides using only cited evidence, tells you how confident it is, and you can open the trace to see exactly why it recommended what it did. When it gets something wrong, I fork from the bad step and show the fix."

---

## Scope

**v1 domain:** resistance training, plus protein and nutrition for body composition. The literature is large, often contradictory and heavily studied, which makes it a good fit for a debate format.

**Later:** cardio and conditioning, sleep, supplements.

**Out of scope (refused at intake):** injuries, medical conditions, medications, pregnancy, eating disorders, and anything requiring diagnosis. These get a fixed "see a professional" response and never reach the planner.

---

## Architecture

```
                        ┌──────────────────── Supervisor ────────────────────┐
                        │  routing · budgets · one send-back · state writes   │
                        └─────────────────────────────────────────────────────┘
                                              │
Intake ─► Planner ─► [Plan checkpoint] ─► Researchers (parallel) ─► Position Mapper
                                                                          │
                                       consensus? ──yes──────────────────►│
                                                                          ▼
                                                     Debate (advocates) ─► Judge ─► Coach Writer ─► Citation Validator
```

- **Shared state:** one LangGraph state object is the single source of truth. It is written to Redis after every node.
- **API:** FastAPI. It starts runs, streams progress events over SSE, and serves reports and traces.
- **UI:** a Next.js chat interface that consumes the SSE stream (see the [UI](#ui) section).

### Specialists

A specialist is a prompt, a tool set and an output schema. It is not a personality: skip "you are a world-class researcher." Every handoff between agents is a Pydantic model.

| Agent | Input | Output | Tools |
|---|---|---|---|
| **Intake** | raw question + user profile | `Intake` (scoped question, profile, in-scope flag) | none |
| **Planner** | `Intake` | `ResearchPlan` (3–6 sub-questions) | none |
| **Researcher** | one sub-question | `list[Finding]` + `SearchStatus` | PubMed, Europe PMC, fetch/extract, retraction check |
| **Position Mapper** | all findings | `list[Position]` + `status` (consensus / leaning / contested) | none |
| **Advocate** (one per position) | its position + findings | `Argument` (only cites finding IDs) | none |
| **Judge** | positions + arguments + grades | `Verdict` (recommendation, confidence) | none |
| **Coach Writer** | verdict + profile | `CoachAnswer` | none |

### Core schemas

```python
class Finding(BaseModel):
    id: str
    claim: str
    source_id: str                 # PMID / DOI / NCT id
    source_url: str
    study_type: Literal["meta_analysis", "rct", "cohort", "case_control",
                        "case_report", "review", "animal", "in_vitro", "preprint"]
    population: str                # who was studied (age, sex, training status)
    sample_size: int | None
    effect: str | None             # direction + size
    published: date
    funding_conflict: bool | None
    retracted: bool
    snippet: str                   # from the full text where available, not the search snippet
    retrieved_at: datetime

class Position(BaseModel):
    stance: str
    supporting_findings: list[str]           # Finding ids
    evidence_grade: Literal["high", "moderate", "low", "very_low"]   # GRADE-style
    best_counterargument: str

class CoachAnswer(BaseModel):
    status: Literal["consensus", "leaning", "genuinely_contested"]
    recommendation: str
    confidence: Literal["high", "moderate", "low"]
    why: str
    positions: list[Position]                # 1 if consensus, 2+ if contested
    what_would_change_this: str
    applies_to_you: str                      # how the user's profile changes things
    practical_protocol: str                  # what to actually do on Monday
    whats_new: str | None                    # if recent research changed the answer
    disclaimer: str                          # always present: not medical advice
```

---

## Phases

### Phase 1: Build the specialists (week 1)
- [x] Pydantic models for every handoff (above).
- [x] Intake with scope classifier and user profile extraction.
- [x] Planner returns concrete, searchable sub-questions.
- [x] Researcher (stubbed search for now) returns graded `Finding` records.
- [x] Coach Writer turns findings into a `CoachAnswer`.
- [x] **Test each agent alone before wiring them together.** If the planner produces vague sub-questions on its own, no orchestration will rescue it. Write fixture-based tests for each.

### Phase 2: Supervisor, routing and budgets (week 1–2)
- [x] Model the system as a LangGraph state graph. Nodes are agents, edges are supervisor routing decisions.
- [x] After each researcher pass, the supervisor checks the output against the plan: Is the sub-question answered? Are the findings only animal or in-vitro studies? Does the study population match the user? Is another pass justified?
- [x] **Enforce budgets in the graph, not in prompts:** max sub-questions, max searches per sub-question, max total tokens, max dollars, and a wall-clock ceiling. A prompt asking the model to be efficient is not a budget.
- [x] The reviewer may send work back **exactly once**. Unlimited revision loops are how agent runs quietly cost twelve dollars.
- [x] **Model routing by cost:** Haiku for extraction and grading, Sonnet for planning, debate and writing, with escalation to a stronger model only when the supervisor's review fails.

### Phase 3: Live research with source tracking (week 2)
- [x] **Sources:** PubMed E-utilities (metadata and publication type), Europe PMC (open-access full text), Crossref / Retraction Watch (retraction checks). Optionally SportRxiv (labeled as preprint) and Tavily for general web context.
- [x] Every finding is a record with provenance. Findings without provenance are unusable in the writer step.
- [x] Fetch and extract the methods and results rather than trusting abstracts and search snippets.
- [x] Prefer recent work: the last 5–10 years, plus older meta-analyses that are still standard references.
- [x] Deduplicate by DOI/PMID and cap results per journal or author group so one lab cannot dominate.
- [x] Handle each failure path explicitly: no results, rate limited, paywalled, timed out, retracted. Each returns a structured `SearchStatus` the supervisor can act on, never an exception that kills the run.

### Phase 4: Position mapping, debate and judgment (week 3)
- [x] **Position Mapper** groups findings into the actual positions in the literature. If the evidence clusters on one side, it marks `consensus` and skips the debate.
- [x] **Avoid false balance.** If the weaker side has only low or very-low grade evidence, the answer is `consensus`, not a 50/50 debate. Questions like "is creatine safe?" should not become debates.
- [x] **Advocates** (one per position) build the strongest case from finding records only. Any claim without a finding ID is rejected by the schema. One rebuttal round, enforced in the graph.
- [x] **Judge** weighs evidence quality, not rhetoric. It sees findings and grades, not just the arguments.
- [x] **Personalization through filtering:** findings whose study population doesn't match the user's profile are down-weighted, and the trace records which ones and why. No invented individual prescriptions.

### Phase 5: Persist run state in Redis (week 3–4)
- [x] Write graph state after every node: run ID, plan, completed sub-questions, findings, positions, debate turns, cumulative token and dollar spend.
- [x] A crashed or redeployed process **resumes instead of replaying**. That is the difference between a demo and a service.
- [x] Idempotency: re-entering a completed node returns the cached result.
- [x] **Topic findings store:** findings are kept by topic across runs. Follow-up questions and overlapping questions reuse them, which cuts cost and latency.
- [x] **Plan checkpoint (human in the loop):** use LangGraph `interrupt` so the user can approve or edit the research plan in the UI before searching. The run can wait for hours and resume from Redis.

### Phase 6: API with citations, streaming and traces (week 4)
- [x] `POST /research` returns a run ID immediately.
- [x] `GET /research/{id}` returns status, then the finished `CoachAnswer`.
- [x] `GET /research/{id}/events` streams progress events over SSE, so the interface is not a four-minute spinner.
- [x] **Citation validation:** every claim and every number (doses, effect sizes, percentages) must link to a real finding record that contains it, or the answer fails validation before it is returned.
- [x] `GET /research/{id}/trace` returns the full step log: which agent ran, what it received, what it produced, which tools it called, tokens, cost and latency per step.
- [x] **Time-travel debugging:** `POST /research/{id}/fork?from_step=N` with an override (prompt, model or tool result) replays from that checkpoint and returns a diff against the original answer.

### Phase 7: Evaluation (week 5)
- [ ] **Golden set:** 30–50 questions with reference answers from ISSN / ACSM position stands and well-known meta-analyses.
- [ ] **Metrics:**
  - *Agreement:* does the coach's recommendation match the reference?
  - *Debate calibration:* does it flag a debate where the literature truly disagrees, and report consensus where it doesn't?
  - *Faithfulness:* does each cited snippet actually support its claim (LLM judge or NLI)?
  - *Coverage:* how many of the reference's key studies did it find?
  - *Cost and latency* per answer.
- [ ] **CI gate:** fail the build if faithfulness or agreement drops.
- [ ] **Cost–quality frontier:** run the eval at several budget caps ($0.05 / $0.20 / $1.00) and plot quality against cost.

### Phase 8: Chat UI (weeks 5–6, can start in parallel from week 4)
See the [UI](#ui) section below.

### Stretch
- [ ] **Chaos mode:** inject worker kills, 429s, paywalls and latency. Publish a resilience scorecard, e.g. "200 chaos runs, 100% completed or failed cleanly, 0 duplicate spend."
- [ ] **Living answers:** subscribe to a question. Scheduled reruns only re-research stale sub-questions and report what changed, e.g. "This changed in 2026 because…".
- [ ] More domains: cardio, sleep, supplements.

---

## UI

A chat interface, monochrome and minimal: black and white only, plus grays. No accent colors; emphasis comes from weight, size and contrast.

### Visual language
- **Colors:** dark theme by default: `#000` background, `#fff` text, grays (`#0a0a0a`, `#171717`, `#262626`, `#737373`, `#a3a3a3`) for surfaces, borders and secondary text. A light theme inverts it. All colors are CSS tokens.
- **Type:** Geist or Inter for UI text, Geist Mono for numbers, PMIDs and the trace. A tight scale: 13 / 15 / 18 / 24.
- **Shape:** 1px hairline borders instead of shadows, small radii (6–8px), plenty of whitespace.
- **Motion:** subtle only: text streaming in, steps fading in, drawers sliding. No bounces.
- **Status without color:** consensus / leaning / contested shown as filled ●, half ◐ and outline ○ markers, and confidence as a 3-segment bar.

### Layout

```
┌──────────────┬───────────────────────────────────────────────┬──────────────┐
│  + New chat  │                                               │   Trace      │
│              │   You: How many sets per week for hypertrophy?│   (drawer,   │
│  Today       │                                               │   toggled)   │
│   Sets/week  │   ▸ Researching · 4 of 5 sub-questions · $0.07│              │
│   Protein    │     ✓ Planned 5 sub-questions                 │   step 1     │
│              │     ✓ PubMed: 23 studies → 9 kept             │   planner    │
│  Last week   │     ◌ Mapping positions…                      │   1.2k tok   │
│   Creatine   │                                               │   0.8s       │
│              │   ◐ LEANING · confidence ▮▮▯                   │   …          │
│              │   10–20 hard sets per muscle per week          │              │
│  ─────────   │   ┌─ Side A ──────────┐┌─ Side B ──────────┐  │              │
│  Profile     │   │ More volume helps ││ Levels off ~10–12 │  │              │
│  Settings    │   └───────────────────┘└───────────────────┘  │              │
│              │   Monday: 12–14 sets, 1–3 reps short…   [1][2]│              │
│              │  ┌─────────────────────────────────────────┐  │              │
│              │  │ Ask a training or nutrition question…  ↑│  │              │
│              │  └─────────────────────────────────────────┘  │              │
└──────────────┴───────────────────────────────────────────────┴──────────────┘
```

### Components
- **Sidebar:** new chat, conversation history grouped by date, and user profile (goal, training age, sex, schedule, equipment), which feeds intake. It collapses on mobile.
- **Composer:** auto-growing textarea at the bottom, Enter to send, Shift+Enter for a newline, with example prompts on an empty chat.
- **Live progress (collapsible):** steps stream in over SSE ("Planned 5 sub-questions", "PubMed: 23 studies → 9 kept", "Debate: round 1"), with a running token and dollar counter. It collapses to one line when the answer arrives.
- **Plan checkpoint card:** shows the sub-questions with Approve / Edit buttons before research starts.
- **Answer card:**
  - Status marker + recommendation as a headline, with a confidence bar.
  - "Why" paragraph.
  - **Both sides:** two columns on desktop (stacked on mobile), each with stance, evidence grade, key studies and best counterargument. Omitted when the status is consensus.
  - "Applies to you", "What would change this" and "What's new" as small labeled sections.
  - **Practical protocol** in a bordered box.
  - Inline citation chips `[1]`. Hovering shows study type, sample size, year and population, and clicking opens PubMed.
  - Disclaimer in small gray text.
- **Trace drawer:** a right-side panel with the step timeline (agent, input/output, tool calls, tokens, latency, cost), expandable JSON, and a **Fork from here** button on each step that opens an override editor and shows the diff.
- **Follow-ups:** questions in the same chat reuse the topic findings store, e.g. "What if I only have 3 days a week?"

### Frontend stack
- Next.js (App Router) + TypeScript + Tailwind, with shadcn/ui primitives restyled to monochrome.
- `EventSource` for the SSE stream. TanStack Query for run status and history.
- Markdown rendering for answer text, with custom renderers for citation chips.
- Conversations and profiles kept in Redis, behind FastAPI endpoints (`/conversations`, `/profile`).

---

## Safety guardrails (enforced in the graph, not just prompts)
- Scope classifier at intake routes out-of-scope questions to a fixed referral response, and that decision is logged in the trace.
- Retracted papers are dropped automatically, with the reason recorded in the trace.
- Every number in an answer must be backed by a finding that contains it.
- A disclaimer is a required schema field, so the writer cannot omit it.
- Preprints and animal studies are labeled as such and never carry the recommendation alone.

---

## Repo layout (proposed)

```
backend/
  agents/          # intake, planner, researcher, mapper, advocate, judge, writer
  graph/           # LangGraph state, supervisor, budgets, routing
  tools/           # pubmed, europepmc, retraction, fetch/extract
  store/           # redis checkpointer, findings store, conversations
  api/             # FastAPI routes, SSE, trace, fork
  schemas.py
  tests/
evals/
  golden/          # questions + reference answers
  run_eval.py
frontend/
  app/             # Next.js routes
  components/      # Sidebar, Composer, ProgressSteps, AnswerCard, BothSides, TraceDrawer, CitationChip
plan.md
```

---

## Done when
- [ ] A training or nutrition question returns a fully cited answer, with both sides when the evidence is contested, **in under five minutes**.
- [ ] A **killed process resumes** from where it stopped, with no duplicate spend.
- [ ] **Every run has a viewable step-by-step trace**, and any step can be forked and diffed.
- [ ] The eval suite reports agreement, debate calibration and faithfulness on the golden set, and CI enforces a floor.
- [x] The chat UI streams progress live, renders both sides and citations, and works on mobile and desktop in both themes.

---

## Timeline (part time)

| Week | Focus |
|---|---|
| 1 | Schemas, specialists tested on their own, scope classifier |
| 2 | Supervisor, budgets, model routing, live PubMed / Europe PMC research |
| 3 | Position mapping, debate, judge; Redis persistence starts |
| 4 | Resume/idempotency, plan checkpoint, FastAPI + SSE + trace + fork; UI begins |
| 5 | Eval harness + CI gate; chat UI core (composer, progress, answer card) |
| 6 | UI polish (both-sides view, citations, trace drawer), cost–quality chart, demo recording |

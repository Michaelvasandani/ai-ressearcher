/**
 * Mock backend for NEXT_PUBLIC_MOCK=1: the same `Api` the UI uses, driven by fixtures and
 * timers. Runs go through the real lifecycle (running -> awaiting_approval -> running ->
 * completed), stream step events with running totals, serve a trace, and fork with a diff.
 */
import type { Api, StreamHandlers } from "@/lib/api/client";
import { ApiError } from "@/lib/api/client";
import type {
  AnswerDiff,
  CoachAnswer,
  Conversation,
  ConversationView,
  PlanDecision,
  ProfileView,
  ProgressEvent,
  ResearchCreated,
  ResearchRequest,
  RunStatus,
  RunView,
  StepOverride,
  SubQuestion,
  TraceStep,
  TraceView,
  UserProfile,
} from "@/lib/api/types";
import { CITATION_FAILED_ISSUES, type Scenario, type StepSpec, scenarioFor } from "./fixtures";

interface MockRun {
  id: string;
  question: string;
  conversationId: string;
  createdAt: string;
  updatedAt: string;
  scenario: Scenario;
  plan: SubQuestion[];
  status: RunStatus;
  requirePlan: boolean;
  awaiting: boolean;
  events: ProgressEvent[];
  steps: TraceStep[];
  specs: StepSpec[]; // what produced each step, so a fork can replay from any index
  tokens: number;
  cost: number;
  answer: CoachAnswer | null;
  parentId: string | null;
  fromStep: number | null;
  override: StepOverride | null;
  inherited: number;
  listeners: Set<StreamHandlers>;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const id = () => Math.random().toString(16).slice(2, 14) + Math.random().toString(16).slice(2, 14);

export class MockApi implements Api {
  readonly mock = true;
  private runs = new Map<string, MockRun>();
  private conversations = new Map<string, Conversation>();
  private profile: UserProfile = {
    goal: "hypertrophy",
    sex: "female",
    training_age_years: 3,
    training_status: "intermediate",
    days_per_week: 4,
    equipment: "full gym",
  };

  constructor(private readonly stepMs = 420) {
    this.seed();
  }

  // --- seeding --------------------------------------------------------------------------

  private seed() {
    const day = 86_400_000;
    const seeds: [string, number][] = [
      ["How much protein do I need on a cut?", 1],
      ["My knee hurts when I squat. Should I keep training legs?", 6],
    ];
    for (const [question, daysAgo] of seeds) {
      const at = new Date(Date.now() - daysAgo * day).toISOString();
      const conv = { id: id(), user_id: "mock", title: question, created_at: at, updated_at: at };
      this.conversations.set(conv.id, conv);
      const run = this.newRun(question, conv.id, false, at);
      for (const spec of [...run.scenario.before, ...run.scenario.after]) this.emitStep(run, spec, false);
      this.finish(run, false);
      run.updatedAt = at;
      conv.updated_at = at;
    }
  }

  // --- run engine -----------------------------------------------------------------------

  private newRun(question: string, conversationId: string, requirePlan: boolean, at = new Date().toISOString()): MockRun {
    const scenario = scenarioFor(question);
    const run: MockRun = {
      id: id(),
      question,
      conversationId,
      createdAt: at,
      updatedAt: at,
      scenario,
      plan: scenario.plan,
      status: "running",
      requirePlan,
      awaiting: false,
      events: [],
      steps: [],
      specs: [],
      tokens: 0,
      cost: 0,
      answer: null,
      parentId: null,
      fromStep: null,
      override: null,
      inherited: 0,
      listeners: new Set(),
    };
    this.runs.set(run.id, run);
    this.push(run, { type: "run_started", question });
    return run;
  }

  private push(run: MockRun, data: Omit<ProgressEvent, "seq">, live = true) {
    const event = { ...data, seq: run.events.length + 1, at: new Date().toISOString() } as ProgressEvent;
    run.events.push(event);
    run.updatedAt = event.at!;
    if (live) run.listeners.forEach((l) => l.onEvent(event));
  }

  private emitStep(run: MockRun, spec: StepSpec, live = true) {
    const tokens = spec.tokens ?? 0;
    const cost = spec.cost ?? 0;
    run.tokens += tokens;
    run.cost += cost;
    const now = new Date();
    const index = run.steps.length;
    const step: TraceStep = {
      index,
      step_id: id().slice(0, 12),
      run_id: run.id,
      node: spec.node,
      agent: spec.agent,
      sub_question_id: spec.sub_question_id ?? null,
      position_id: spec.position_id ?? null,
      attempt: spec.attempt ?? 1,
      model: spec.model ?? null,
      escalated: spec.model === "claude-opus-5-5",
      route_reason: spec.model ? "routed by agent" : null,
      input_summary: spec.input,
      output_summary: spec.output,
      output: spec.json ?? { summary: spec.output },
      decision: null,
      status: spec.status ?? "ok",
      error: null,
      llm_calls: spec.model
        ? [
            {
              call_id: id().slice(0, 10),
              agent: spec.agent ?? spec.node,
              model: spec.model,
              latency_ms: spec.ms ?? 0,
              input_tokens: Math.round(tokens * 0.8),
              output_tokens: Math.round(tokens * 0.2),
              cost_usd: cost,
              replayed: false,
              ok: true,
            },
          ]
        : [],
      tool_calls: (spec.tools ?? []).map((t) => ({ ...t, latency_ms: 1200 })),
      input_tokens: Math.round(tokens * 0.8),
      output_tokens: Math.round(tokens * 0.2),
      cost_usd: cost,
      latency_ms: spec.ms ?? 5,
      started_at: new Date(now.getTime() - (spec.ms ?? 5)).toISOString(),
      finished_at: now.toISOString(),
    };
    run.steps.push(step);
    run.specs.push(spec);
    this.push(
      run,
      {
        type: "step",
        index,
        node: spec.node,
        agent: spec.agent,
        message: spec.message,
        status: step.status,
        sub_question_id: step.sub_question_id,
        tokens,
        cost_usd: cost,
        latency_ms: step.latency_ms,
        totals: { tokens: run.tokens, cost_usd: Number(run.cost.toFixed(6)) },
      },
      live,
    );
  }

  private finish(run: MockRun, live = true) {
    run.status = "completed";
    run.answer = run.answer ?? run.scenario.answer;
    this.push(
      run,
      {
        type: "run_finished",
        status: run.scenario.outcome.status,
        message: run.scenario.outcome.message,
        cost_usd: run.cost,
        totals: { tokens: run.tokens, cost_usd: run.cost },
      },
      live,
    );
    if (live) run.listeners.forEach((l) => l.onEnd?.({ status: "completed", outcome_status: run.scenario.outcome.status }));
    const conv = this.conversations.get(run.conversationId);
    if (conv) conv.updated_at = run.updatedAt;
  }

  private async play(run: MockRun, specs: StepSpec[]) {
    for (const spec of specs) {
      await sleep(this.stepMs * (spec.node === "researcher" || spec.agent === "advocate" ? 1.6 : 1));
      this.emitStep(run, spec);
    }
  }

  private async drive(run: MockRun) {
    await this.play(run, run.scenario.before);
    if (run.scenario.kind === "referral") {
      await sleep(this.stepMs);
      return this.finish(run);
    }
    if (run.requirePlan) {
      run.status = "awaiting_approval";
      run.awaiting = true;
      this.push(run, { type: "plan_ready", sub_questions: run.plan });
      return; // decidePlan() continues
    }
    await this.continueAfterPlan(run, "approve");
  }

  private async continueAfterPlan(run: MockRun, action: "approve" | "edit") {
    await sleep(this.stepMs);
    this.emitStep(run, {
      node: "plan_checkpoint",
      agent: "supervisor",
      message: action === "edit" ? `Plan edited: ${run.plan.length} sub-questions` : "Plan approved",
      input: "plan",
      output: action,
      ms: 1,
    });
    const after = run.scenario.after.map((s) =>
      s.node === "dispatch" ? { ...s, message: `Researching ${run.plan.length} sub-questions` } : s,
    );
    await this.play(run, after);
    await sleep(this.stepMs);
    this.finish(run);
  }

  // --- Api ------------------------------------------------------------------------------

  async startResearch(body: ResearchRequest): Promise<ResearchCreated> {
    await sleep(120);
    let convId = body.conversation_id ?? null;
    if (convId) {
      const conv = this.conversations.get(convId);
      if (!conv) throw new ApiError(404, `conversation ${convId} not found`);
      conv.updated_at = new Date().toISOString();
    } else {
      const now = new Date().toISOString();
      convId = id();
      this.conversations.set(convId, { id: convId, user_id: "mock", title: body.question.slice(0, 80), created_at: now, updated_at: now });
    }
    const run = this.newRun(body.question, convId, Boolean(body.require_plan_approval));
    void this.drive(run);
    return { run_id: run.id, conversation_id: convId, status: "running" };
  }

  private must(runId: string): MockRun {
    const run = this.runs.get(runId);
    if (!run) throw new ApiError(404, `run ${runId} not found`);
    return run;
  }

  async getRun(runId: string): Promise<RunView> {
    const run = this.must(runId);
    const done = run.status === "completed";
    const parent = run.parentId ? this.runs.get(run.parentId) : undefined;
    const cited = new Set([...(run.answer?.cited_finding_ids ?? []), ...(run.answer?.positions.flatMap((p) => p.supporting_findings) ?? [])]);
    return {
      run_id: run.id,
      status: run.status,
      question: run.question,
      conversation_id: run.conversationId,
      created_at: run.createdAt,
      updated_at: run.updatedAt,
      plan: run.awaiting
        ? { run_id: run.id, question: run.question, sub_questions: run.plan, dropped: [], max_sub_questions: 6, error: null }
        : null,
      plan_decision_pending: false,
      sub_questions: run.steps.length > 1 ? run.plan : [],
      answer: done ? run.answer : null,
      outcome: done ? run.scenario.outcome : null,
      citations:
        done && run.scenario.kind === "citations_failed"
          ? { attempt: 2, ok: false, claims_checked: 12, numbers_checked: 18, numbers_from_user: 1, citations: 11, issues: CITATION_FAILED_ISSUES }
          : done && run.answer
            ? { attempt: 1, ok: true, claims_checked: 14, numbers_checked: 22, numbers_from_user: 1, citations: 18, issues: [] }
            : null,
      findings: done ? run.scenario.findings.filter((f) => !run.answer || cited.has(f.id)) : [],
      spend: { input_tokens: Math.round(run.tokens * 0.8), output_tokens: Math.round(run.tokens * 0.2), cost_usd: run.cost, llm_calls: run.steps.filter((s) => s.model).length, searches: run.steps.filter((s) => s.node === "researcher").length * 2 },
      steps: run.steps.length,
      error: null,
      parent_run_id: run.parentId,
      fork:
        run.parentId && run.fromStep !== null
          ? {
              parent_run_id: run.parentId,
              from_step: run.fromStep,
              step_id: parent?.steps[run.fromStep]?.step_id ?? "",
              node: parent?.steps[run.fromStep]?.node ?? "",
              scope: { agent: parent?.steps[run.fromStep]?.agent ?? "", override: run.override ?? {} },
              inherited_steps: run.inherited,
            }
          : null,
      diff: done && parent ? this.diff(parent, run) : null,
    };
  }

  private diff(parent: MockRun, fork: MockRun): AnswerDiff {
    const a = parent.answer;
    const b = fork.answer;
    const changes = (["status", "recommendation", "confidence"] as const)
      .filter((k) => a?.[k] !== b?.[k])
      .map((k) => ({ field: k, before: a?.[k] ?? null, after: b?.[k] ?? null }));
    const newCost = fork.steps.slice(fork.inherited).reduce((s, st) => s + st.cost_usd, 0);
    return {
      original_run_id: parent.id,
      fork_run_id: fork.id,
      from_step: fork.fromStep ?? 0,
      changed: changes.length > 0,
      changes,
      positions_added: [],
      positions_removed: [],
      positions_changed: [],
      citations_added: [],
      citations_removed: [],
      cost_usd_original: parent.cost,
      cost_usd_fork: fork.cost,
      new_cost_usd: Number((newCost * 0.35).toFixed(4)),
    };
  }

  async decidePlan(runId: string, decision: PlanDecision): Promise<ResearchCreated> {
    await sleep(100);
    const run = this.must(runId);
    if (!run.awaiting) throw new ApiError(409, `run ${runId} is ${run.status}, not waiting for plan approval`);
    if (decision.action === "edit") {
      const qs = decision.sub_questions ?? [];
      if (!qs.length) throw new ApiError(422, "an edit must list the sub-questions to research");
      if (qs.some((q) => q.question.trim().length < 10)) throw new ApiError(422, "each sub-question needs at least 10 characters");
      let n = run.plan.length;
      run.plan = qs.map((q) => {
        const old = run.plan.find((p) => p.id === q.id);
        return old && old.question === q.question
          ? old
          : { id: q.id ?? `sq${++n}`, question: q.question, rationale: "added by the user at the plan checkpoint", search_terms: [q.question] };
      });
    }
    run.awaiting = false;
    run.status = "running";
    this.push(run, { type: "plan_decided", action: decision.action });
    void this.continueAfterPlan(run, decision.action);
    return { run_id: run.id, conversation_id: run.conversationId, status: "running" };
  }

  async getTrace(runId: string): Promise<TraceView> {
    const run = this.must(runId);
    return {
      run_id: run.id,
      status: run.status,
      steps: run.steps,
      inherited_steps: run.inherited,
      spend: { input_tokens: 0, output_tokens: 0, cost_usd: run.cost, llm_calls: 0, searches: 0 },
      totals: {
        steps: run.steps.length,
        tokens: run.tokens,
        cost_usd: run.cost,
        latency_ms: run.steps.reduce((s, st) => s + st.latency_ms, 0),
      },
    };
  }

  async fork(runId: string, fromStep: number, override: StepOverride): Promise<ResearchCreated> {
    await sleep(150);
    const parent = this.must(runId);
    if (parent.status !== "completed") throw new ApiError(409, "fork a finished run");
    const step = parent.steps[fromStep];
    if (!step) throw new ApiError(422, `run has no step ${fromStep}`);
    if (!override.prompt_addendum?.trim() && !override.model && !override.tool_result)
      throw new ApiError(422, "an override needs a prompt_addendum, a model or a tool_result");
    const run = this.newRun(parent.question, parent.conversationId, false);
    run.parentId = parent.id;
    run.fromStep = fromStep;
    run.override = override;
    run.plan = parent.plan;
    run.inherited = fromStep;
    run.steps = parent.steps.slice(0, fromStep);
    run.tokens = run.steps.reduce((s, st) => s + st.input_tokens + st.output_tokens, 0);
    run.cost = run.steps.reduce((s, st) => s + st.cost_usd, 0);
    const base = parent.answer ?? parent.scenario.answer;
    if (base) {
      run.answer = {
        ...base,
        recommendation:
          base.status === "leaning"
            ? "Aim for about **12–16 hard sets per muscle per week**; most trained lifters see little extra growth beyond that [3][4]."
            : base.recommendation.replace("1.6–2.2", "1.8–2.2"),
        confidence: base.status === "leaning" ? "low" : base.confidence,
      };
    }
    run.specs = parent.specs.slice(0, fromStep);
    const replay = parent.specs.slice(fromStep).map((s, i) =>
      i === 0 && override.model ? { ...s, model: override.model } : s,
    );
    void (async () => {
      await this.play(run, replay);
      await sleep(this.stepMs);
      this.finish(run);
    })();
    return { run_id: run.id, conversation_id: run.conversationId, status: "running", parent_run_id: parent.id };
  }

  async listConversations(): Promise<Conversation[]> {
    return [...this.conversations.values()].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  }

  async getConversation(convId: string): Promise<ConversationView> {
    const conv = this.conversations.get(convId);
    if (!conv) throw new ApiError(404, `conversation ${convId} not found`);
    const runs = [...this.runs.values()]
      .filter((r) => r.conversationId === convId)
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt))
      .map((r) => ({
        run_id: r.id,
        question: r.question,
        status: r.status,
        outcome_status: r.status === "completed" ? r.scenario.outcome.status : null,
        cost_usd: r.cost,
        parent_run_id: r.parentId,
        created_at: r.createdAt,
      }));
    return { conversation: conv, runs };
  }

  async deleteConversation(convId: string): Promise<void> {
    this.conversations.delete(convId);
  }

  async getProfile(): Promise<ProfileView> {
    return { user_id: "mock", profile: this.profile };
  }

  async putProfile(profile: UserProfile): Promise<ProfileView> {
    this.profile = profile;
    return { user_id: "mock", profile };
  }

  subscribe(runId: string, handlers: StreamHandlers, after = 0): () => void {
    const run = this.runs.get(runId);
    if (!run) {
      handlers.onEnd?.({ status: "missing" });
      return () => {};
    }
    let live = true;
    // Replay what happened so far (as the SSE endpoint does from seq 0), then follow.
    queueMicrotask(() => {
      if (!live) return;
      for (const e of run.events) if (e.seq > after) handlers.onEvent(e);
      if (run.status === "completed") handlers.onEnd?.({ status: "completed", outcome_status: run.scenario.outcome.status });
      else run.listeners.add(handlers);
    });
    return () => {
      live = false;
      run.listeners.delete(handlers);
    };
  }
}

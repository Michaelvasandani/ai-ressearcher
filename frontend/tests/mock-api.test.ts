import { describe, expect, it } from "vitest";
import { MockApi } from "@/lib/mock/api";
import type { ProgressEvent } from "@/lib/api/types";

async function until<T>(fn: () => Promise<T>, ok: (v: T) => boolean, ms = 5000): Promise<T> {
  const end = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (ok(v)) return v;
    if (Date.now() > end) throw new Error("timed out");
    await new Promise((r) => setTimeout(r, 5));
  }
}

describe("mock API (demo mode)", () => {
  it("runs the full lifecycle: plan checkpoint, events with totals, answer, fork with diff", async () => {
    const api = new MockApi(1);
    const created = await api.startResearch({ question: "How many sets per week?", require_plan_approval: true });
    const events: ProgressEvent[] = [];
    let ended = false;
    api.subscribe(created.run_id, { onEvent: (e) => events.push(e), onEnd: () => (ended = true) });

    const waiting = await until(() => api.getRun(created.run_id), (r) => r.status === "awaiting_approval");
    expect(waiting.plan?.sub_questions.length).toBe(4);
    await api.decidePlan(created.run_id, { action: "approve" });

    const done = await until(() => api.getRun(created.run_id), (r) => r.status === "completed");
    expect(done.answer?.status).toBe("leaning");
    expect(done.findings.map((f) => f.id)).toEqual(expect.arrayContaining(done.answer!.cited_finding_ids));
    await until(async () => ended, Boolean);
    const steps = events.filter((e) => e.type === "step");
    expect(steps[0].message).toBe("Question in scope");
    expect(steps.at(-1)?.totals?.cost_usd).toBeGreaterThan(0);
    expect(events.map((e) => e.seq)).toEqual([...events.map((e) => e.seq)].sort((a, b) => a - b));

    const trace = await api.getTrace(created.run_id);
    const writer = trace.steps.find((s) => s.node === "writer")!;
    const fork = await api.fork(created.run_id, writer.index, { prompt_addendum: "Prefer trials in trained lifters." });
    const forked = await until(() => api.getRun(fork.run_id), (r) => r.status === "completed");
    expect(forked.diff?.changed).toBe(true);
    expect(forked.diff?.changes.map((c) => c.field)).toContain("recommendation");
    expect((await api.getConversation(created.conversation_id!)).runs).toHaveLength(2);
  });

  it("refers out-of-scope questions without research", async () => {
    const api = new MockApi(1);
    const created = await api.startResearch({ question: "My knee hurts when I squat, what should I do?" });
    const done = await until(() => api.getRun(created.run_id), (r) => r.status === "completed");
    expect(done.outcome?.status).toBe("referred");
    expect(done.answer).toBeNull();
  });

  it("fails cleanly on citations_failed", async () => {
    const api = new MockApi(1);
    const created = await api.startResearch({ question: "Show me a citation failure" });
    const done = await until(() => api.getRun(created.run_id), (r) => r.status === "completed");
    expect(done.outcome?.failure?.reason).toBe("citations_failed");
    expect(done.citations?.ok).toBe(false);
    expect(done.findings.length).toBeGreaterThan(0);
  });
});

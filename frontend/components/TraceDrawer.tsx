"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { getApi } from "@/lib/api";
import { FORKABLE_AGENTS, MODELS, type StepOverride, type TraceStep } from "@/lib/api/types";
import { keys, useRun } from "@/lib/hooks";
import { cn, formatMs, formatTokens, formatUsd } from "@/lib/utils";
import { DiffView } from "./DiffView";
import { Dot } from "./StatusMarker";
import { Button } from "./ui/button";
import { Eyebrow, Field, Select, Textarea } from "./ui/field";

function StepMark({ status }: { status: TraceStep["status"] }) {
  if (status === "failed") return <span className="text-xs leading-none">✕</span>;
  return <Dot size={8} fill={status === "ok" ? "full" : status === "degraded" ? "half" : "empty"} />;
}

function shortModel(model: string | null): string | null {
  if (!model) return null;
  return model.replace(/^claude-/, "").replace(/-(\d+)-(\d+)$/, " $1.$2");
}

function Json({ value }: { value: unknown }) {
  const [open, setOpen] = useState(false);
  const text = JSON.stringify(value, null, 2);
  return (
    <div>
      <button type="button" className="text-xs text-muted hover:text-fg cursor-pointer" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "▾" : "▸"} Output JSON <span className="font-mono text-subtle">({(text.length / 1024).toFixed(1)} KB)</span>
      </button>
      {open ? (
        <pre className="mt-1 max-h-80 overflow-auto rounded-sm border border-border bg-surface p-2 font-mono text-[11px] leading-4 text-fg-2">
          {text}
        </pre>
      ) : null}
    </div>
  );
}

function ForkEditor({
  runId,
  step,
  onForked,
  onCancel,
}: {
  runId: string;
  step: TraceStep;
  onForked: (forkRunId: string) => void;
  onCancel: () => void;
}) {
  const qc = useQueryClient();
  const [addendum, setAddendum] = useState("");
  const [model, setModel] = useState("");
  const [toolResult, setToolResult] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const researcher = step.agent === "researcher";
  const fork = useMutation({
    mutationFn: (o: StepOverride) => getApi().fork(runId, step.index, o),
    onSuccess: (created) => {
      void qc.invalidateQueries({ queryKey: keys.conversations });
      if (created.conversation_id) void qc.invalidateQueries({ queryKey: keys.conversation(created.conversation_id) });
      onForked(created.run_id);
    },
  });

  function submit() {
    const override: StepOverride = {};
    if (addendum.trim()) override.prompt_addendum = addendum.trim();
    if (model) override.model = model;
    if (researcher && toolResult.trim()) {
      try {
        const parsed = JSON.parse(toolResult);
        if (!Array.isArray(parsed)) throw new Error("tool result must be a JSON array of study records");
        override.tool_result = parsed;
      } catch (e) {
        setParseError((e as Error).message);
        return;
      }
    }
    setParseError(null);
    fork.mutate(override);
  }

  const empty = !addendum.trim() && !model && !(researcher && toolResult.trim());
  return (
    <div className="mt-3 flex flex-col gap-3 rounded-md border border-border-strong p-3 animate-slide-up" data-testid="fork-editor">
      <Eyebrow>Fork from step {step.index} · override</Eyebrow>
      <Field label="Prompt addendum" hint="Appended to this agent's system prompt for the replay.">
        <Textarea rows={3} value={addendum} onChange={(e) => setAddendum(e.target.value)} placeholder="e.g. Weigh trials in trained lifters more heavily." />
      </Field>
      <Field label="Model">
        <Select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="">Keep routed model{step.model ? ` (${shortModel(step.model)})` : ""}</option>
          {MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </Select>
      </Field>
      {researcher ? (
        <Field label="Tool result (JSON array of StudyRecord)" hint="What every search of this sub-question returns in the replay.">
          <Textarea rows={4} className="font-mono text-[11px]" value={toolResult} onChange={(e) => setToolResult(e.target.value)} placeholder='[{"source_id": "PMID:…", "title": "…", "abstract": "…", "published": "2024-01-01", "source_url": "…"}]' />
        </Field>
      ) : null}
      {parseError || fork.error ? <p className="text-xs text-fg">✕ {parseError ?? (fork.error as Error).message}</p> : null}
      <div className="flex gap-2">
        <Button variant="primary" size="sm" disabled={empty || fork.isPending} onClick={submit}>
          {fork.isPending ? "Forking…" : "Fork and replay"}
        </Button>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

function ForkResult({ forkRunId }: { forkRunId: string }) {
  const { data: run } = useRun(forkRunId, true);
  if (!run || run.status !== "completed") {
    return (
      <div className="mt-3 rounded-md border border-border p-3 text-xs text-muted" data-testid="fork-progress">
        <span className="pulse">◌</span> Replaying fork · {run?.steps ?? 0} steps · {formatUsd(run?.spend.cost_usd)}
      </div>
    );
  }
  return <div className="mt-3">{run.diff ? <DiffView diff={run.diff} /> : <p className="text-xs text-muted">Fork finished ({run.outcome?.status}).</p>}</div>;
}

function StepRow({ step, runId, canFork, inherited }: { step: TraceStep; runId: string; canFork: boolean; inherited: boolean }) {
  const [open, setOpen] = useState(false);
  const [forking, setForking] = useState(false);
  const [forkRunId, setForkRunId] = useState<string | null>(null);
  const tag = step.sub_question_id ?? step.position_id;
  const forkable = canFork && !!step.agent && FORKABLE_AGENTS.has(step.agent);
  return (
    <li className={cn("border-b border-border animate-fade-in", inherited && "opacity-60")}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-surface cursor-pointer"
      >
        <span className="w-6 shrink-0 font-mono text-[11px] tabular-nums text-subtle">{step.index}</span>
        <span className="flex w-3 shrink-0 justify-center text-fg" title={step.status} aria-label={`status ${step.status}`}>
          <StepMark status={step.status} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs text-fg">
            {step.node.replace(/_/g, " ")}
            {tag ? <span className="ml-1.5 font-mono text-[11px] text-subtle">{tag}</span> : null}
            {step.attempt > 1 ? <span className="ml-1.5 font-mono text-[11px] text-subtle">#{step.attempt}</span> : null}
          </span>
          <span className="block truncate font-mono text-[11px] text-subtle">
            {[shortModel(step.model), step.input_tokens + step.output_tokens ? `${formatTokens(step.input_tokens + step.output_tokens)} tok` : null, formatMs(step.latency_ms)]
              .filter(Boolean)
              .join(" · ")}
          </span>
        </span>
        <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted">{step.cost_usd ? formatUsd(step.cost_usd) : ""}</span>
      </button>
      {open ? (
        <div className="flex flex-col gap-3 px-4 pb-4 pl-[60px] text-xs">
          <div>
            <div className="text-subtle">Input</div>
            <div className="break-words text-fg-2">{step.input_summary}</div>
          </div>
          <div>
            <div className="text-subtle">Output</div>
            <div className="break-words text-fg-2">{step.output_summary}</div>
            {step.error ? <div className="mt-1 break-words text-muted">✕ {step.error}</div> : null}
          </div>
          {step.route_reason ? (
            <div>
              <div className="text-subtle">Routing</div>
              <div className="break-words text-muted">
                {step.route_reason}
                {step.escalated ? " (escalated)" : ""}
              </div>
            </div>
          ) : null}
          {step.tool_calls.length ? (
            <div>
              <div className="text-subtle">Tool calls</div>
              <ul className="font-mono text-[11px] text-muted">
                {step.tool_calls.map((t, i) => (
                  <li key={i} className="break-words">
                    {t.tool} · {t.state} · {t.results} results · {formatMs(t.latency_ms)}
                    <span className="block text-subtle">{t.input}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {step.llm_calls.length ? (
            <div>
              <div className="text-subtle">LLM calls</div>
              <ul className="font-mono text-[11px] text-muted">
                {step.llm_calls.map((c) => (
                  <li key={c.call_id}>
                    {shortModel(c.model)} · {formatTokens(c.input_tokens)} in / {formatTokens(c.output_tokens)} out · {formatMs(c.latency_ms)} ·{" "}
                    {formatUsd(c.cost_usd)}
                    {c.replayed ? " · replayed" : ""}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {step.output ? <Json value={step.output} /> : null}
          {forkable && !forking && !forkRunId ? (
            <div>
              <Button size="sm" onClick={() => setForking(true)}>
                ⑂ Fork from here
              </Button>
            </div>
          ) : null}
          {forking && !forkRunId ? (
            <ForkEditor runId={runId} step={step} onCancel={() => setForking(false)} onForked={(id) => setForkRunId(id)} />
          ) : null}
          {forkRunId ? <ForkResult forkRunId={forkRunId} /> : null}
        </div>
      ) : null}
    </li>
  );
}

/** Right-side panel: the step timeline, expandable JSON, and "Fork from here" with a diff. */
export function TraceDrawer({ runId, onClose }: { runId: string | null; onClose: () => void }) {
  const { data: run } = useRun(runId ?? "", false);
  const { data: trace, isLoading, error } = useQuery({
    queryKey: keys.trace(runId ?? ""),
    queryFn: () => getApi().getTrace(runId!),
    enabled: !!runId,
    refetchInterval: run && run.status !== "completed" ? 3_000 : false,
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!runId) return null;
  const finished = run?.status === "completed";
  return (
    <>
      <div className="fixed inset-0 z-30 bg-overlay xl:hidden animate-fade-in" onClick={onClose} aria-hidden />
      <aside
        className="fixed inset-y-0 right-0 z-40 flex w-full flex-col border-l border-border bg-bg animate-slide-in sm:w-[440px] xl:sticky xl:top-0 xl:z-auto xl:h-dvh xl:w-[420px] xl:shrink-0"
        aria-label="Trace"
        data-testid="trace-drawer"
      >
        <header className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <div className="text-sm font-semibold">Trace</div>
            <div className="truncate font-mono text-[11px] text-subtle">run {runId.slice(0, 12)}</div>
          </div>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close trace">
            ×
          </Button>
        </header>
        {trace ? (
          <div className="grid grid-cols-4 border-b border-border font-mono text-[11px]">
            {[
              ["steps", String(trace.totals.steps ?? trace.steps.length)],
              ["tokens", formatTokens(trace.totals.tokens)],
              ["cost", formatUsd(trace.totals.cost_usd)],
              ["latency", formatMs(trace.totals.latency_ms)],
            ].map(([k, v]) => (
              <div key={k} className="border-r border-border px-3 py-2 last:border-r-0">
                <div className="text-subtle">{k}</div>
                <div className="text-fg tabular-nums">{v}</div>
              </div>
            ))}
          </div>
        ) : null}
        <div className="min-h-0 flex-1 overflow-y-auto">
          {isLoading ? <p className="p-4 text-xs text-muted">Loading trace…</p> : null}
          {error ? <p className="p-4 text-xs text-muted">✕ {(error as Error).message}</p> : null}
          {trace && !trace.steps.length ? <p className="p-4 text-xs text-muted">No steps yet.</p> : null}
          {trace?.inherited_steps ? (
            <p className="border-b border-border px-4 py-2 text-xs text-subtle">
              Steps 0–{trace.inherited_steps - 1} are inherited from the original run.
            </p>
          ) : null}
          <ol>
            {trace?.steps.map((s) => (
              <StepRow key={`${s.run_id}-${s.step_id}`} step={s} runId={runId} canFork={finished} inherited={s.run_id !== runId} />
            ))}
          </ol>
          {trace && !finished ? <p className="p-4 text-xs text-subtle">Forking is available once the run finishes.</p> : null}
        </div>
      </aside>
    </>
  );
}

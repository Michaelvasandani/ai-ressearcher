"use client";

import { useState } from "react";
import type { ProgressEvent, Spend, Totals } from "@/lib/api/types";
import { cn, formatTokens, formatUsd } from "@/lib/utils";

/** What is happening now, given the last finished step. */
const PHASE: Record<string, string> = {
  intake: "Planning",
  planner: "Planning",
  plan_checkpoint: "Researching",
  dispatch: "Researching",
  researcher: "Researching",
  review: "Weighing the evidence",
  personalize: "Mapping positions",
  position_mapper: "Weighing positions",
  advocate: "Debating",
  debate_review: "Judging",
  judge: "Writing the answer",
  writer: "Checking citations",
  validate_citations: "Finishing",
  check_answer: "Finishing",
};

const STEP_GLYPH: Record<string, string> = { ok: "✓", degraded: "~", failed: "✕", skipped: "–" };

export interface ProgressState {
  running: boolean;
  awaiting: boolean;
  events: ProgressEvent[];
  subQuestions: number;
  spend?: Spend | null;
  stepsCount?: number;
}

export function totalsOf(events: ProgressEvent[], spend?: Spend | null): Totals {
  for (let i = events.length - 1; i >= 0; i--) {
    const t = events[i].totals;
    if (t) return t;
  }
  return { tokens: (spend?.input_tokens ?? 0) + (spend?.output_tokens ?? 0), cost_usd: spend?.cost_usd ?? 0 };
}

/** The one-line summary: "Researching · 4 of 5 sub-questions · 54.2k tok · $0.07". */
export function headline({ running, awaiting, events, subQuestions, stepsCount }: ProgressState): string {
  const steps = events.filter((e) => e.type === "step");
  if (awaiting) return "Waiting for your plan approval";
  if (!running) return `Researched · ${steps.length || stepsCount || 0} steps`;
  const last = steps[steps.length - 1];
  const phase = last ? PHASE[last.node ?? ""] ?? "Working" : "Reading the question";
  if (phase === "Researching" && subQuestions) {
    const done = new Set(steps.filter((e) => e.node === "researcher").map((e) => e.sub_question_id)).size;
    return `Researching · ${done} of ${subQuestions} sub-questions`;
  }
  return phase;
}

/**
 * Live progress, streamed over SSE. Expanded while the run works; collapses to one line once
 * the answer arrives (click to expand again).
 */
export function ProgressSteps(props: ProgressState & { onExpand?: () => void }) {
  const { running, awaiting, events, spend } = props;
  const [open, setOpen] = useState<boolean | null>(null);
  const expanded = open ?? (running || awaiting);
  const totals = totalsOf(events, spend);
  const checkpointed = events.some((e) => e.node === "plan_checkpoint");
  const steps = events.filter(
    (e) => e.type === "step" || e.type === "run_error" || (e.type === "plan_decided" && !checkpointed),
  );

  return (
    <div className="text-xs" data-testid="progress">
      <button
        type="button"
        onClick={() => {
          if (!expanded) props.onExpand?.();
          setOpen(!expanded);
        }}
        aria-expanded={expanded}
        className="group flex w-full items-center gap-2 py-1 text-left text-muted hover:text-fg cursor-pointer"
      >
        <span aria-hidden className={cn("inline-block w-3 text-center transition-transform duration-150", expanded && "rotate-90")}>
          ▸
        </span>
        <span className={cn("text-fg-2", running && "pulse")}>{headline(props)}</span>
        <span className="text-subtle">·</span>
        <span className="font-mono tabular-nums">{formatTokens(totals.tokens)} tok</span>
        <span className="text-subtle">·</span>
        <span className="font-mono tabular-nums">{formatUsd(totals.cost_usd)}</span>
      </button>
      {expanded ? (
        <ol className="mt-1 ml-[5px] flex flex-col border-l border-border pl-4">
          {steps.map((e) => (
            <li key={e.seq} className="flex items-baseline gap-2 py-[3px] text-muted animate-fade-in">
              <span aria-hidden className="w-3 shrink-0 text-center text-fg-2">
                {e.type === "plan_decided" ? "✓" : e.type === "run_error" ? "✕" : STEP_GLYPH[String(e.status)] ?? "✓"}
              </span>
              <span className="min-w-0 flex-1 break-words">
                {e.type === "plan_decided"
                  ? `Plan ${e.action === "edit" ? "edited" : "approved"} by you`
                  : e.type === "run_error"
                    ? `Error: ${String(e.error ?? "").slice(0, 160)}`
                    : e.message}
              </span>
              {e.type === "step" && e.cost_usd ? (
                <span className="hidden shrink-0 font-mono text-[11px] tabular-nums text-subtle sm:inline">{formatUsd(e.cost_usd)}</span>
              ) : null}
            </li>
          ))}
          {running && !awaiting ? (
            <li className="flex items-baseline gap-2 py-[3px] text-subtle">
              <span aria-hidden className="pulse w-3 text-center">◌</span>
              <span>{headline({ ...props, running: true })}…</span>
            </li>
          ) : null}
          {!steps.length && !running ? <li className="py-[3px] text-subtle">No progress events recorded.</li> : null}
        </ol>
      ) : null}
    </div>
  );
}

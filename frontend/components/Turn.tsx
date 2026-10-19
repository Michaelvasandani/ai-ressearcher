"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getApi } from "@/lib/api";
import type { PlanDecision, RunSummary } from "@/lib/api/types";
import { isFinished, keys, useRun, useRunEvents } from "@/lib/hooks";
import { formatUsd } from "@/lib/utils";
import { AnswerCard } from "./AnswerCard";
import { DiffView } from "./DiffView";
import { ErrorCard, FailedCard, ReferralCard } from "./OutcomeCards";
import { PlanCard } from "./PlanCard";
import { ProgressSteps } from "./ProgressSteps";
import { Button } from "./ui/button";

export function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end animate-fade-in">
      <p className="max-w-[85%] whitespace-pre-wrap rounded-md border border-border bg-surface-2 px-3.5 py-2 text-sm text-fg">{text}</p>
    </div>
  );
}

/** One question and everything the run produced: progress, plan checkpoint, answer or failure. */
export function Turn({ summary, onOpenTrace }: { summary: Pick<RunSummary, "run_id" | "question" | "status" | "parent_run_id">; onOpenTrace: (runId: string) => void }) {
  const qc = useQueryClient();
  const [stream, setStream] = useState(summary.status !== "completed");
  const { events, down } = useRunEvents(summary.run_id, stream);
  const { data: run, error } = useRun(summary.run_id, down);

  const decide = useMutation({
    mutationFn: (d: PlanDecision) => getApi().decidePlan(summary.run_id, d),
    onSuccess: () => {
      setStream(true);
      void qc.invalidateQueries({ queryKey: keys.run(summary.run_id) });
    },
  });

  const finished = isFinished(run);
  const awaiting = run?.status === "awaiting_approval" && !!run.plan;
  const running = !finished && run?.status !== "error" && !awaiting;
  const isFork = !!(run?.fork ?? summary.parent_run_id);

  return (
    <div className="flex flex-col gap-4" data-run={summary.run_id}>
      {isFork ? (
        <div className="flex items-center gap-2 text-xs text-subtle">
          <span aria-hidden>⑂</span>
          <span>
            Fork{run?.fork ? ` from step ${run.fork.from_step} (${run.fork.node.replace(/_/g, " ")})` : ""}
            {run?.fork?.scope.override.model ? ` · model ${run.fork.scope.override.model}` : ""}
            {run?.fork?.scope.override.prompt_addendum ? " · prompt addendum" : ""}
            {run?.fork?.scope.override.tool_result ? " · tool result" : ""}
          </span>
        </div>
      ) : (
        <UserBubble text={summary.question} />
      )}

      <div className="flex flex-col gap-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <ProgressSteps
              running={running}
              awaiting={awaiting}
              events={events}
              subQuestions={run?.sub_questions.length ?? 0}
              spend={run?.spend}
              stepsCount={run?.steps}
              onExpand={() => setStream(true)}
            />
          </div>
          <Button variant="ghost" size="sm" className="shrink-0" onClick={() => onOpenTrace(summary.run_id)} aria-label="Open trace">
            Trace
          </Button>
        </div>

        {error && !run ? <p className="text-xs text-muted">✕ Could not load this run: {(error as Error).message}</p> : null}

        {awaiting && run?.plan ? (
          <PlanCard
            key={run.plan.sub_questions.map((s) => s.id).join()}
            plan={run.plan}
            busy={decide.isPending}
            error={decide.error ? (decide.error as Error).message : null}
            onDecide={(d) => decide.mutate(d)}
          />
        ) : null}

        {run?.status === "error" ? <ErrorCard error={run.error} /> : null}

        {finished && run?.outcome ? (
          run.outcome.status === "referred" ? (
            <ReferralCard outcome={run.outcome} />
          ) : run.answer ? (
            <AnswerCard answer={run.answer} findings={run.findings ?? []} outcome={run.outcome} />
          ) : (
            <FailedCard outcome={run.outcome} citations={run.citations} findings={run.findings ?? []} />
          )
        ) : null}

        {finished && run?.diff ? <DiffView diff={run.diff} compact /> : null}

        {finished && run ? (
          <div className="font-mono text-[11px] text-subtle">
            {run.steps} steps · {formatUsd(run.spend.cost_usd)}
            {run.citations?.ok ? ` · ${run.citations.claims_checked} claims, ${run.citations.numbers_checked} numbers checked` : ""}
          </div>
        ) : null}
      </div>
    </div>
  );
}

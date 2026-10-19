"use client";

import { useState } from "react";
import type { PlanApprovalRequest, PlanDecision } from "@/lib/api/types";
import { Button } from "./ui/button";
import { Eyebrow, Textarea } from "./ui/field";

interface Draft {
  id: string | null;
  question: string;
}

/** The plan checkpoint: the sub-questions, with Approve / Edit, before any searching. */
export function PlanCard({
  plan,
  onDecide,
  busy,
  error,
}: {
  plan: PlanApprovalRequest;
  onDecide: (decision: PlanDecision) => void;
  busy?: boolean;
  error?: string | null;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<Draft[]>(() => plan.sub_questions.map((s) => ({ id: s.id, question: s.question })));
  const tooShort = draft.some((d) => d.question.trim().length < 10);
  const problem = error || plan.error;

  return (
    <section className="rounded-md border border-border-strong p-4 animate-fade-in" data-testid="plan-card">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <Eyebrow>Research plan · review before searching</Eyebrow>
        <span className="font-mono text-xs text-subtle">
          {editing ? draft.length : plan.sub_questions.length}/{plan.max_sub_questions}
        </span>
      </div>

      {!editing ? (
        <ol className="flex flex-col gap-2">
          {plan.sub_questions.map((sq, i) => (
            <li key={sq.id} className="flex gap-3">
              <span className="w-5 shrink-0 font-mono text-xs leading-6 text-subtle">{i + 1}</span>
              <div className="min-w-0">
                <p className="text-sm text-fg">{sq.question}</p>
                <p className="text-xs text-subtle">{sq.rationale}</p>
              </div>
            </li>
          ))}
        </ol>
      ) : (
        <ol className="flex flex-col gap-2">
          {draft.map((d, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="w-5 shrink-0 pt-1.5 font-mono text-xs text-subtle">{i + 1}</span>
              <Textarea
                aria-label={`Sub-question ${i + 1}`}
                value={d.question}
                rows={2}
                className="text-sm"
                onChange={(e) => setDraft(draft.map((x, j) => (j === i ? { ...x, question: e.target.value } : x)))}
              />
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Remove sub-question ${i + 1}`}
                disabled={draft.length <= 1}
                onClick={() => setDraft(draft.filter((_, j) => j !== i))}
              >
                ×
              </Button>
            </li>
          ))}
          {draft.length < plan.max_sub_questions ? (
            <li className="pl-7">
              <Button variant="ghost" size="sm" onClick={() => setDraft([...draft, { id: null, question: "" }])}>
                + Add sub-question
              </Button>
            </li>
          ) : null}
        </ol>
      )}

      {plan.dropped.length && !editing ? (
        <p className="mt-3 text-xs text-subtle">
          {plan.dropped.length} more sub-question{plan.dropped.length === 1 ? "" : "s"} cut to fit the budget.
        </p>
      ) : null}
      {problem ? <p className="mt-3 text-xs text-fg">✕ {problem}</p> : null}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        {!editing ? (
          <>
            <Button variant="primary" onClick={() => onDecide({ action: "approve" })} disabled={busy}>
              Approve and research
            </Button>
            <Button onClick={() => setEditing(true)} disabled={busy}>
              Edit
            </Button>
          </>
        ) : (
          <>
            <Button
              variant="primary"
              disabled={busy || tooShort || !draft.length}
              onClick={() =>
                onDecide({
                  action: "edit",
                  sub_questions: draft.map((d) => ({ id: d.id, question: d.question.trim() })),
                })
              }
            >
              Save and research
            </Button>
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => {
                setDraft(plan.sub_questions.map((s) => ({ id: s.id, question: s.question })));
                setEditing(false);
              }}
            >
              Cancel
            </Button>
            {tooShort ? <span className="text-xs text-subtle">Each sub-question needs at least 10 characters.</span> : null}
          </>
        )}
      </div>
    </section>
  );
}

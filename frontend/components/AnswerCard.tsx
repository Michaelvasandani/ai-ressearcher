"use client";

import type { CoachAnswer, Finding, RunOutcome } from "@/lib/api/types";
import { citationIndex } from "@/lib/citations";
import { cn } from "@/lib/utils";
import { BothSides } from "./BothSides";
import { Markdown } from "./Markdown";
import { ConfidenceBar, StatusMarker } from "./StatusMarker";
import { Eyebrow } from "./ui/field";

function Section({ label, text, index }: { label: string; text: string | null | undefined; index: ReturnType<typeof citationIndex> }) {
  if (!text?.trim()) return null;
  return (
    <section>
      <Eyebrow className="mb-1">{label}</Eyebrow>
      <Markdown text={text} index={index} className="text-sm text-fg-2" />
    </section>
  );
}

const DEGRADATION_TEXT: Record<string, string> = {
  max_usd: "the dollar budget ran out",
  max_tokens: "the token budget ran out",
  wall_clock: "the time limit was reached",
  max_searches: "the search cap was reached",
  max_sub_questions: "the plan was cut to fit the budget",
  sub_question_unanswered: "a sub-question found no usable studies",
  abstract_only_evidence: "some studies were read from abstracts only",
  debate_skipped: "the debate was skipped to stay in budget",
  rebuttal_skipped: "the rebuttal round was skipped",
  judge_skipped: "the judge step was skipped",
  mapping_skipped: "positions were not mapped",
  population_mismatch: "some studies are on different populations",
};

export function AnswerCard({ answer, findings, outcome }: { answer: CoachAnswer; findings: Finding[]; outcome?: RunOutcome | null }) {
  const index = citationIndex(answer, findings);
  const notes = [...new Set((outcome?.degradations ?? []).map((d) => DEGRADATION_TEXT[d.kind]).filter(Boolean))];
  return (
    <article className="flex flex-col gap-6 animate-fade-in" data-testid="answer-card">
      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <StatusMarker status={answer.status} />
          <ConfidenceBar confidence={answer.confidence} />
        </div>
        <Markdown
          text={answer.recommendation}
          index={index}
          className={cn(
            "font-semibold tracking-[-0.01em] text-fg",
            // a one-line verdict reads as a headline; a paragraph-long one at 24px is a wall
            answer.recommendation.length > 180 ? "text-lg" : "text-2xl",
          )}
        />
      </header>

      <Section label="Why" text={answer.why} index={index} />

      <BothSides answer={answer} index={index} />

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <Section label="Applies to you" text={answer.applies_to_you} index={index} />
        <Section label="What would change this" text={answer.what_would_change_this} index={index} />
        {answer.whats_new ? <Section label="What's new" text={answer.whats_new} index={index} /> : null}
      </div>

      <section className="rounded-md border border-border-strong p-4">
        <Eyebrow className="mb-2">Practical protocol</Eyebrow>
        <Markdown text={answer.practical_protocol} index={index} className="text-sm text-fg" />
      </section>

      {outcome?.status === "degraded" && notes.length ? (
        <p className="text-xs text-muted">
          <span className="mr-1">~</span>Partial answer: {notes.join("; ")}.
        </p>
      ) : null}

      <p className="text-xs text-subtle">{answer.disclaimer}</p>
    </article>
  );
}

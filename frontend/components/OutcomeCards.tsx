"use client";

import type { CitationReport, Finding, RunOutcome } from "@/lib/api/types";
import { citationIndex } from "@/lib/citations";
import { STUDY_TYPE_LABEL } from "@/lib/utils";
import { CitationChip } from "./CitationChip";
import { Eyebrow } from "./ui/field";

export function ReferralCard({ outcome }: { outcome: RunOutcome }) {
  return (
    <article className="rounded-md border border-border p-5 animate-fade-in" data-testid="referral-card">
      <Eyebrow className="mb-2">Outside what this coach covers</Eyebrow>
      <p className="text-sm text-fg">{outcome.referral_message || outcome.message}</p>
      <p className="mt-3 text-xs text-subtle">
        Evidence Coach answers training and nutrition questions for healthy adults. Injuries, medical conditions,
        medications, pregnancy and eating disorders need someone who can assess you in person.
      </p>
    </article>
  );
}

const FAILURE_TITLE: Record<string, string> = {
  citations_failed: "Answer withheld: its citations did not check out",
  budget_exhausted: "Stopped: the run's budget ran out",
  insufficient_evidence: "Not enough evidence to answer",
  intake_failed: "Could not read the question",
  planner_failed: "Could not plan the research",
  writer_failed: "Could not write the answer",
  internal_error: "Something went wrong",
};

const FAILURE_TEXT: Record<string, string> = {
  citations_failed:
    "Every claim and number has to be backed by a study the answer cites. The draft failed that check, was rewritten once, and still failed, so it is not shown. The evidence that was found is below.",
  budget_exhausted: "The run hit its spending or time cap before it could finish. Try a narrower question or a larger budget in Settings.",
  insufficient_evidence: "The searches did not return studies good enough to answer with. Try rephrasing the question.",
};

export function FailedCard({ outcome, citations, findings }: { outcome: RunOutcome; citations: CitationReport | null; findings: Finding[] }) {
  const reason = outcome.failure?.reason ?? "internal_error";
  const index = citationIndex({ cited_finding_ids: findings.map((f) => f.id) }, findings);
  const issues = citations && !citations.ok ? citations.issues : [];
  return (
    <article className="flex flex-col gap-4 rounded-md border border-border p-5 animate-fade-in" data-testid="failed-card" data-reason={reason}>
      <div>
        <Eyebrow className="mb-2">
          <span aria-hidden className="mr-1.5">✕</span>Run failed
        </Eyebrow>
        <p className="text-lg font-semibold">{FAILURE_TITLE[reason] ?? FAILURE_TITLE.internal_error}</p>
        <p className="mt-1 text-sm text-muted">{FAILURE_TEXT[reason] ?? outcome.failure?.message ?? outcome.message}</p>
      </div>
      {issues.length ? (
        <div>
          <Eyebrow className="mb-2">
            Citation check · {citations!.claims_checked} claims, {citations!.numbers_checked} numbers
          </Eyebrow>
          <ul className="flex flex-col gap-2">
            {issues.slice(0, 6).map((issue, i) => (
              <li key={i} className="rounded-sm border border-border px-3 py-2">
                <div className="text-xs text-fg-2">“{issue.sentence}”</div>
                <div className="mt-1 font-mono text-[11px] text-subtle">
                  {issue.field} · {issue.detail.replace(`"${issue.sentence}"`, "this sentence")}
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {findings.length ? (
        <div>
          <Eyebrow className="mb-2">Evidence found</Eyebrow>
          <ul className="flex flex-col gap-1.5">
            {findings.slice(0, 8).map((f, i) => (
              <li key={f.id} className="flex gap-2 text-xs text-muted">
                <CitationChip n={i + 1} finding={index.byId(f.id)} />
                <span>
                  <span className="text-fg-2">{f.claim}</span>{" "}
                  <span className="text-subtle">({STUDY_TYPE_LABEL[f.study_type] ?? f.study_type})</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </article>
  );
}

export function ErrorCard({ error }: { error: string | null }) {
  return (
    <article className="rounded-md border border-border p-5 animate-fade-in" data-testid="error-card">
      <Eyebrow className="mb-2">
        <span aria-hidden className="mr-1.5">✕</span>Run interrupted
      </Eyebrow>
      <p className="text-sm text-fg">The run stopped with an error. It resumes from its last checkpoint automatically, without paying again for finished steps.</p>
      {error ? <p className="mt-2 font-mono text-[11px] text-subtle break-words">{error.slice(0, 300)}</p> : null}
    </article>
  );
}

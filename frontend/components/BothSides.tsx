"use client";

import type { CoachAnswer, EvidenceGrade, Position } from "@/lib/api/types";
import type { CitationIndex } from "@/lib/citations";
import { GRADE_LABEL, cn } from "@/lib/utils";
import { CitationChip } from "./CitationChip";
import { Eyebrow } from "./ui/field";

const GRADE_FILLED: Record<EvidenceGrade, number> = { very_low: 1, low: 2, moderate: 3, high: 4 };

/** GRADE certainty as four small squares (GRADE's own ⊕⊕⊕◯ notation, without color). */
export function GradeMeter({ grade }: { grade: EvidenceGrade }) {
  const filled = GRADE_FILLED[grade];
  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted" aria-label={`Evidence grade: ${GRADE_LABEL[grade]}`}>
      <span className="inline-flex gap-[2px]" aria-hidden>
        {[0, 1, 2, 3].map((i) => (
          <span key={i} className={cn("h-2 w-2 rounded-[1px] border", i < filled ? "border-fg bg-fg" : "border-border-strong")} />
        ))}
      </span>
      <span>{GRADE_LABEL[grade]} certainty</span>
    </span>
  );
}

/** Whether the answer shows both sides: never for consensus, and only with two or more positions. */
export function showsBothSides(answer: Pick<CoachAnswer, "status" | "positions">): boolean {
  return answer.status !== "consensus" && answer.positions.length >= 2;
}

function Side({ position, label, leading, index }: { position: Position; label: string; leading: boolean; index: CitationIndex }) {
  return (
    <section className="flex flex-col gap-3 rounded-md border border-border p-4" aria-label={label}>
      <div className="flex items-center justify-between gap-2">
        <Eyebrow>{label}</Eyebrow>
        {leading ? <span className="text-xs text-muted">Leading</span> : null}
      </div>
      <p className="text-sm font-medium text-fg">{position.stance}</p>
      <GradeMeter grade={position.evidence_grade} />
      <div>
        <div className="mb-1 text-xs text-subtle">Key studies</div>
        <div className="flex flex-wrap gap-1">
          {position.supporting_findings.map((fid) => {
            const n = index.number(fid);
            return n ? (
              <CitationChip key={fid} n={n} finding={index.byId(fid)} />
            ) : (
              <span key={fid} className="font-mono text-[11px] text-subtle">
                {fid}
              </span>
            );
          })}
        </div>
      </div>
      <div className="mt-auto border-t border-border pt-3">
        <div className="mb-1 text-xs text-subtle">Best counterargument</div>
        <p className="text-xs leading-[20px] text-muted">{position.best_counterargument}</p>
      </div>
    </section>
  );
}

/** Two columns on desktop, stacked on mobile. Omitted when the status is consensus. */
export function BothSides({ answer, index }: { answer: Pick<CoachAnswer, "status" | "positions">; index: CitationIndex }) {
  if (!showsBothSides(answer)) return null;
  const letters = "ABCDEF";
  return (
    <div data-testid="both-sides">
      <Eyebrow className="mb-2">Both sides</Eyebrow>
      <div className={cn("grid grid-cols-1 gap-3", answer.positions.length >= 2 && "md:grid-cols-2")}>
        {answer.positions.map((p, i) => (
          <Side key={i} position={p} label={`Side ${letters[i] ?? i + 1}`} leading={i === 0} index={index} />
        ))}
      </div>
    </div>
  );
}

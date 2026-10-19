"use client";

import type { Finding } from "@/lib/api/types";
import { sourceHref, year } from "@/lib/citations";
import { STUDY_TYPE_LABEL, cn } from "@/lib/utils";
import { Tooltip } from "./ui/tooltip";

export function FindingSummary({ finding }: { finding: Finding }) {
  const facts = [
    STUDY_TYPE_LABEL[finding.study_type] ?? finding.study_type,
    finding.sample_size ? `n=${finding.sample_size.toLocaleString("en-US")}` : null,
    year(finding),
  ].filter(Boolean);
  return (
    <div className="space-y-1.5">
      <div className="font-mono text-xs text-fg">{facts.join(" · ")}</div>
      {finding.title ? <div className="text-xs font-medium leading-snug text-fg">{finding.title}</div> : null}
      <div className="text-xs text-muted">{finding.population}</div>
      {finding.retracted ? <div className="text-xs font-semibold uppercase">Retracted</div> : null}
      <div className="font-mono text-[11px] text-subtle">{finding.source_id}</div>
    </div>
  );
}

/**
 * `[n]` in the answer text. Hover (or focus) shows study type, sample size, year and
 * population; click opens PubMed (or the source URL).
 */
export function CitationChip({ n, finding }: { n: number; finding?: Finding }) {
  const chip = cn(
    "mx-[1px] inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-[4px] border px-1 align-[2px]",
    "font-mono text-[11px] leading-none tabular-nums transition-colors",
  );
  if (!finding) {
    return (
      <Tooltip content={<span className="text-muted">Source details unavailable</span>}>
        <span className={cn(chip, "border-border text-subtle")} data-cite={n} tabIndex={0} aria-label={`Citation ${n}`}>
          {n}
        </span>
      </Tooltip>
    );
  }
  return (
    <Tooltip content={<FindingSummary finding={finding} />}>
      <a
        href={sourceHref(finding)}
        target="_blank"
        rel="noreferrer noopener"
        data-cite={n}
        aria-label={`Citation ${n}: ${STUDY_TYPE_LABEL[finding.study_type] ?? finding.study_type}, ${year(finding)}`}
        className={cn(chip, "border-border-strong text-fg-2 hover:border-fg hover:bg-fg hover:text-bg")}
      >
        {n}
      </a>
    </Tooltip>
  );
}

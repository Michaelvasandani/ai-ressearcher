import type { AnswerDiff } from "@/lib/api/types";
import { cn, formatUsd } from "@/lib/utils";
import { Eyebrow } from "./ui/field";

function show(v: unknown): string {
  if (v === null || v === undefined) return "—";
  return typeof v === "string" ? v.replace(/\*\*/g, "") : JSON.stringify(v);
}

/** Fork vs original: what changed (struck-through before, plain after; no color), and what it cost. */
export function DiffView({ diff, compact = false }: { diff: AnswerDiff; compact?: boolean }) {
  return (
    <section className={cn("rounded-md border border-border", compact ? "p-3" : "p-4")} data-testid="diff-view">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <Eyebrow>
          {diff.changed ? "Diff against the original" : "No change against the original"} · step {diff.from_step}
        </Eyebrow>
      </div>
      <div className="flex flex-col gap-3">
        {diff.changes.map((c) => (
          <div key={c.field}>
            <div className="mb-1 font-mono text-[11px] uppercase text-subtle">{c.field}</div>
            {c.before === null || c.before === undefined ? (
              <div className="text-xs leading-5 text-subtle">− (none)</div>
            ) : (
              <div className="text-xs leading-5 text-subtle line-through decoration-subtle">− {show(c.before)}</div>
            )}
            <div className="text-xs leading-5 text-fg">+ {show(c.after)}</div>
          </div>
        ))}
        {diff.positions_added.map((p, i) => (
          <div key={`a${i}`} className="text-xs text-fg">+ position: {p.stance}</div>
        ))}
        {diff.positions_removed.map((p, i) => (
          <div key={`r${i}`} className="text-xs text-subtle line-through">− position: {p.stance}</div>
        ))}
        {diff.positions_changed.map((p, i) => (
          <div key={`c${i}`} className="text-xs">
            <span className="text-muted">~ position </span>
            {p.stance_before !== p.stance_after ? (
              <>
                <span className="text-subtle line-through">{p.stance_before}</span> <span className="text-fg">{p.stance_after}</span>
              </>
            ) : (
              <span className="text-fg-2">{p.stance_after}</span>
            )}
            {p.grade_before !== p.grade_after ? (
              <span className="text-muted">
                {" "}· grade {p.grade_before} → {p.grade_after}
              </span>
            ) : null}
          </div>
        ))}
        {diff.citations_added.length || diff.citations_removed.length ? (
          <div className="font-mono text-[11px] text-muted">
            citations {diff.citations_added.length ? `+${diff.citations_added.length}` : ""}{" "}
            {diff.citations_removed.length ? `−${diff.citations_removed.length}` : ""}
          </div>
        ) : null}
        <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-border pt-2 font-mono text-[11px] text-subtle">
          <span>original {formatUsd(diff.cost_usd_original)}</span>
          <span>fork {formatUsd(diff.cost_usd_fork)}</span>
          <span className="text-fg-2">new spend {formatUsd(diff.new_cost_usd)}</span>
        </div>
      </div>
    </section>
  );
}

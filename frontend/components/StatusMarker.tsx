import type { Confidence, EvidenceStatus } from "@/lib/api/types";
import { cn } from "@/lib/utils";

/** Status without color: consensus ●, leaning ◐, contested ○. */
export const STATUS_GLYPH: Record<EvidenceStatus, string> = {
  consensus: "●",
  leaning: "◐",
  genuinely_contested: "○",
};

export const STATUS_LABEL: Record<EvidenceStatus, string> = {
  consensus: "Consensus",
  leaning: "Leaning",
  genuinely_contested: "Contested",
};

/** ● ◐ ○ drawn as SVG, so they look the same in every font. */
export function Dot({ fill, size = 10, className }: { fill: "full" | "half" | "empty"; size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 10 10" aria-hidden className={cn("shrink-0", className)}>
      <circle cx="5" cy="5" r="4.25" fill={fill === "full" ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.5" />
      {fill === "half" ? <path d="M5 0.75 A4.25 4.25 0 0 0 5 9.25 Z" fill="currentColor" /> : null}
    </svg>
  );
}

const STATUS_FILL: Record<EvidenceStatus, "full" | "half" | "empty"> = {
  consensus: "full",
  leaning: "half",
  genuinely_contested: "empty",
};

export function StatusMarker({
  status,
  showLabel = true,
  className,
}: {
  status: EvidenceStatus;
  showLabel?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn("inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-[0.08em]", className)}
      data-status={status}
    >
      <span className="inline-flex text-fg" data-glyph={STATUS_GLYPH[status]}>
        <Dot fill={STATUS_FILL[status]} />
      </span>
      {showLabel ? <span>{STATUS_LABEL[status]}</span> : <span className="sr-only">{STATUS_LABEL[status]}</span>}
    </span>
  );
}

const FILLED: Record<Confidence, number> = { low: 1, moderate: 2, high: 3 };

/** Confidence as a 3-segment bar: low ▮▯▯, moderate ▮▮▯, high ▮▮▮. */
export function ConfidenceBar({ confidence, className }: { confidence: Confidence; className?: string }) {
  const filled = FILLED[confidence];
  return (
    <span
      className={cn("inline-flex items-center gap-2 text-xs uppercase tracking-[0.08em] text-muted", className)}
      role="img"
      aria-label={`Confidence: ${confidence}`}
    >
      <span>Confidence</span>
      <span className="inline-flex gap-[3px]" aria-hidden>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            data-filled={i < filled}
            className={cn("h-2.5 w-4 rounded-[2px] border", i < filled ? "border-fg bg-fg" : "border-border-strong bg-transparent")}
          />
        ))}
      </span>
      <span className="text-fg normal-case tracking-normal">{confidence}</span>
    </span>
  );
}

import type { CoachAnswer, Finding } from "@/lib/api/types";

/**
 * Citation markers, as the backend writes them (backend/citations.py MARKER): `[1]`,
 * `[1, 2]`, `[2-4]`. `[n]` points at `answer.cited_finding_ids[n - 1]`.
 */
export const MARKER = /\[\s*(\d+(?:\s*(?:[,;]|[-–])\s*\d+)*)\s*\](?!\()/g;

export function markerNumbers(body: string): number[] {
  const out: number[] = [];
  for (const part of body.split(/\s*[,;]\s*/)) {
    const range = part.split(/\s*[-–]\s*/).map((x) => Number.parseInt(x, 10));
    if (range.length === 2 && range[0] <= range[1] && range[1] - range[0] < 50) {
      for (let n = range[0]; n <= range[1]; n++) out.push(n);
    } else if (!Number.isNaN(range[0])) {
      out.push(range[0]);
    }
  }
  return [...new Set(out)];
}

/** Turn markers into markdown links (`[1](#cite-1)`) the markdown renderer shows as chips. */
export function linkifyCitations(text: string): string {
  return text.replace(MARKER, (_, body: string) => `[${body}](#cite-${markerNumbers(body).join(",")})`);
}

export function citedNumbers(href: string | undefined): number[] | null {
  if (!href?.startsWith("#cite-")) return null;
  return href
    .slice(6)
    .split(",")
    .map((x) => Number.parseInt(x, 10))
    .filter((n) => Number.isFinite(n));
}

export interface CitationIndex {
  /** Finding for marker n (1-based), if the run returned it. */
  finding(n: number): Finding | undefined;
  /** Finding id for marker n. */
  id(n: number): string | undefined;
  /** Marker number of a finding id (for "key studies" of a position). */
  number(findingId: string): number | undefined;
  byId(findingId: string): Finding | undefined;
}

export function citationIndex(answer: Pick<CoachAnswer, "cited_finding_ids"> | null, findings: Finding[]): CitationIndex {
  const refs = answer?.cited_finding_ids ?? [];
  const map = new Map(findings.map((f) => [f.id, f]));
  return {
    id: (n) => refs[n - 1],
    finding: (n) => (refs[n - 1] ? map.get(refs[n - 1]) : undefined),
    number: (fid) => {
      const i = refs.indexOf(fid);
      return i >= 0 ? i + 1 : undefined;
    },
    byId: (fid) => map.get(fid),
  };
}

/** Where a click on a citation goes: PubMed for PMIDs, else the source URL. */
export function sourceHref(f: Finding): string {
  const m = /^PMID:(\d+)$/i.exec(f.source_id);
  if (m) return `https://pubmed.ncbi.nlm.nih.gov/${m[1]}/`;
  return f.source_url;
}

export function year(f: Finding): string {
  return (f.published || "").slice(0, 4);
}

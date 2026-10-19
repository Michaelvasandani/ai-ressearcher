import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatUsd(v: number | null | undefined): string {
  const n = v ?? 0;
  if (n === 0) return "$0.00";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(2)}`;
}

export function formatTokens(v: number | null | undefined): string {
  const n = v ?? 0;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function formatMs(ms: number | null | undefined): string {
  const n = ms ?? 0;
  if (n < 1000) return `${n}ms`;
  if (n < 60_000) return `${(n / 1000).toFixed(1)}s`;
  return `${Math.floor(n / 60_000)}m ${Math.round((n % 60_000) / 1000)}s`;
}

export const STUDY_TYPE_LABEL: Record<string, string> = {
  meta_analysis: "Meta-analysis",
  rct: "RCT",
  cohort: "Cohort",
  case_control: "Case-control",
  case_report: "Case report",
  review: "Review",
  animal: "Animal study",
  in_vitro: "In vitro",
  preprint: "Preprint",
};

export const GRADE_LABEL: Record<string, string> = {
  high: "High",
  moderate: "Moderate",
  low: "Low",
  very_low: "Very low",
};

/** Date buckets for the history sidebar. */
export function dateBucket(iso: string, now = new Date()): string {
  const d = new Date(iso);
  const start = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((start(now) - start(d)) / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Previous 7 days";
  if (days < 30) return "Previous 30 days";
  return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

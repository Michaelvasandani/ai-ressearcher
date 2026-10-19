import * as React from "react";
import { cn } from "@/lib/utils";

const base =
  "w-full rounded-sm border border-border bg-transparent px-2.5 text-sm text-fg placeholder:text-subtle transition-colors hover:border-border-strong focus:border-fg focus:outline-none";

export function Input({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(base, "h-8", className)} {...props} />;
}

export function Textarea({ className, ...props }: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cn(base, "py-1.5 resize-y", className)} {...props} />;
}

export function Select({ className, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn(base, "h-8 cursor-pointer bg-bg", className)} {...props}>
      {children}
    </select>
  );
}

export function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs text-muted">{label}</span>
      {children}
      {hint ? <span className="mt-1 block text-xs text-subtle">{hint}</span> : null}
    </label>
  );
}

/** Small uppercase section label used across the answer card. */
export function Eyebrow({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("text-xs font-medium uppercase tracking-[0.08em] text-subtle", className)}>{children}</div>;
}

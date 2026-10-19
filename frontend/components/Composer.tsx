"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export const EXAMPLE_PROMPTS = [
  "How many sets per week for hypertrophy?",
  "How much protein do I need on a cut?",
  "Is training to failure necessary for muscle growth?",
  "What if I only have 3 days a week to train?",
];

/** Auto-growing textarea. Enter sends, Shift+Enter adds a newline. */
export function Composer({
  onSend,
  disabled,
  placeholder = "Ask a training or nutrition question…",
  autoFocus,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  const canSend = text.trim().length >= 3 && !disabled;
  const send = () => {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <form
      className="flex items-end gap-2 rounded-md border border-border bg-bg p-2 pl-3.5 transition-colors focus-within:border-border-strong"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <textarea
        ref={ref}
        rows={1}
        value={text}
        autoFocus={autoFocus}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            send();
          }
        }}
        placeholder={placeholder}
        aria-label="Question"
        className="max-h-[220px] min-h-[32px] flex-1 resize-none bg-transparent py-1 text-sm text-fg placeholder:text-subtle focus:outline-none"
      />
      <button
        type="submit"
        disabled={!canSend}
        aria-label="Send"
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-sm text-sm transition-opacity cursor-pointer",
          "bg-invert text-invert-fg disabled:cursor-default disabled:opacity-25",
        )}
      >
        ↑
      </button>
    </form>
  );
}

export function ExamplePrompts({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      {EXAMPLE_PROMPTS.map((p) => (
        <button
          key={p}
          type="button"
          onClick={() => onPick(p)}
          className="rounded-md border border-border px-3.5 py-3 text-left text-xs text-muted transition-colors hover:border-border-strong hover:text-fg cursor-pointer"
        >
          {p}
        </button>
      ))}
    </div>
  );
}

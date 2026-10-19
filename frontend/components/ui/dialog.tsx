"use client";

import { Dialog as D } from "radix-ui";
import * as React from "react";
import { cn } from "@/lib/utils";

export const Dialog = D.Root;
export const DialogTrigger = D.Trigger;
export const DialogClose = D.Close;

export function DialogContent({
  title,
  description,
  children,
  className,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <D.Portal>
      <D.Overlay className="fixed inset-0 z-50 bg-overlay animate-fade-in" />
      <D.Content
        className={cn(
          "fixed left-1/2 top-1/2 z-50 w-[calc(100vw-32px)] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-md border border-border bg-bg p-5 animate-slide-up",
          "max-h-[calc(100dvh-32px)] overflow-y-auto",
          className,
        )}
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <D.Title className="text-lg font-semibold">{title}</D.Title>
            {description ? <D.Description className="mt-1 text-xs text-muted">{description}</D.Description> : null}
          </div>
          <D.Close className="text-muted hover:text-fg text-lg leading-none cursor-pointer" aria-label="Close">
            ×
          </D.Close>
        </div>
        {children}
      </D.Content>
    </D.Portal>
  );
}

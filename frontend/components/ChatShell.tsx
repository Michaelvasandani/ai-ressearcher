"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { getApi } from "@/lib/api";
import type { ConversationView, ResearchCreated } from "@/lib/api/types";
import { keys } from "@/lib/hooks";
import { useSettings } from "@/lib/settings";
import { Composer, ExamplePrompts } from "./Composer";
import { Sidebar } from "./Sidebar";
import { Dot } from "./StatusMarker";
import { TraceDrawer } from "./TraceDrawer";
import { Turn, UserBubble } from "./Turn";
import { Button } from "./ui/button";

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto flex w-full max-w-[720px] flex-col gap-8 px-4 pt-[14vh] pb-8 animate-fade-in">
      <div className="flex flex-col gap-3">
        <h1 className="text-2xl font-semibold tracking-[-0.01em]">What does the research say?</h1>
        <p className="max-w-[560px] text-sm text-muted">
          Ask a training or nutrition question. The coach reads the studies, maps where they agree and where they split,
          argues both sides from cited findings only, and tells you how confident it is.
        </p>
        <p className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-subtle">
          <span className="inline-flex items-center gap-1.5"><Dot fill="full" /> consensus</span>
          <span className="inline-flex items-center gap-1.5"><Dot fill="half" /> leaning</span>
          <span className="inline-flex items-center gap-1.5"><Dot fill="empty" /> contested</span>
        </p>
      </div>
      <ExamplePrompts onPick={onPick} />
    </div>
  );
}

/** The chat: sidebar, the conversation's turns, the composer, and the trace drawer. */
export function ChatShell() {
  const params = useParams<{ id?: string }>();
  const convId = params?.id ?? null;
  const router = useRouter();
  const qc = useQueryClient();
  const { settings } = useSettings();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [traceRunId, setTraceRunId] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const { data: conversation, error } = useQuery({
    queryKey: keys.conversation(convId ?? ""),
    queryFn: () => getApi().getConversation(convId!),
    enabled: !!convId,
  });

  const start = useMutation({
    mutationFn: (question: string) =>
      getApi().startResearch({
        question,
        conversation_id: convId,
        budget: { max_usd: settings.maxUsd },
        require_plan_approval: settings.requirePlanApproval,
      }),
    onMutate: (question) => setPending(question),
    onSuccess: (created: ResearchCreated, question) => {
      const cid = created.conversation_id!;
      const now = new Date().toISOString();
      qc.setQueryData<ConversationView>(keys.conversation(cid), (old) => ({
        conversation: old?.conversation ?? { id: cid, user_id: null, title: question, created_at: now, updated_at: now },
        runs: [
          ...(old?.runs ?? []),
          { run_id: created.run_id, question, status: "running", outcome_status: null, cost_usd: 0, parent_run_id: null, created_at: now },
        ],
      }));
      void qc.invalidateQueries({ queryKey: keys.conversations });
      if (cid !== convId) router.push(`/c/${cid}`);
      setPending(null);
    },
    onError: () => setPending(null),
  });

  const send = useCallback((q: string) => start.mutate(q), [start]);
  const runs = convId ? (conversation?.runs ?? []) : [];

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [runs.length, pending]);

  useEffect(() => {
    // A different conversation: its trace is not this one's.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTraceRunId(null);
  }, [convId]);

  const empty = !convId && !pending;
  const title = conversation?.conversation.title ?? (convId ? "" : "New chat");

  return (
    <div className="flex min-h-dvh">
      <Sidebar activeId={convId} open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-12 items-center gap-2 border-b border-border bg-bg px-3">
          <Button variant="ghost" size="icon-sm" className="md:hidden" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
            <span aria-hidden className="text-base leading-none">☰</span>
          </Button>
          <div className="min-w-0 flex-1 truncate text-xs text-muted">{title}</div>
          {runs.length ? (
            <Button variant="ghost" size="sm" onClick={() => setTraceRunId(traceRunId ? null : runs[runs.length - 1].run_id)}>
              {traceRunId ? "Hide trace" : "Trace"}
            </Button>
          ) : null}
        </header>

        <div className="flex-1">
          {empty ? (
            <EmptyState onPick={send} />
          ) : (
            <div className="mx-auto flex w-full max-w-[760px] flex-col gap-10 px-4 pt-6 pb-8">
              {error ? <p className="text-xs text-muted">✕ Could not load this conversation: {(error as Error).message}</p> : null}
              {runs.map((r) => (
                <Turn key={r.run_id} summary={r} onOpenTrace={setTraceRunId} />
              ))}
              {pending ? (
                <div className="flex flex-col gap-4">
                  <UserBubble text={pending} />
                  <p className="text-xs text-muted">
                    <span className="pulse">◌</span> Starting…
                  </p>
                </div>
              ) : null}
              {start.error ? <p className="text-xs text-fg">✕ {(start.error as Error).message}</p> : null}
              <div ref={bottom} />
            </div>
          )}
        </div>

        <div className="sticky bottom-0 z-10 bg-gradient-to-t from-bg via-bg to-transparent px-4 pt-6 pb-4">
          <div className="mx-auto w-full max-w-[760px]">
            <Composer
              onSend={send}
              disabled={start.isPending}
              autoFocus
              placeholder={runs.length ? "Ask a follow-up question…" : undefined}
            />
            <p className="mt-2 text-center text-[11px] text-subtle">Research summaries, not medical advice. Every claim links to its study.</p>
          </div>
        </div>
      </main>

      <TraceDrawer runId={traceRunId} onClose={() => setTraceRunId(null)} />
    </div>
  );
}

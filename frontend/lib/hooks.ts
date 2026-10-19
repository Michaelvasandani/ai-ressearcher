"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { getApi } from "@/lib/api";
import type { ProgressEvent, RunView } from "@/lib/api/types";

export const keys = {
  run: (id: string) => ["run", id] as const,
  trace: (id: string) => ["trace", id] as const,
  conversations: ["conversations"] as const,
  conversation: (id: string) => ["conversation", id] as const,
  profile: ["profile"] as const,
};

export function isFinished(run: Pick<RunView, "status"> | undefined): boolean {
  return run?.status === "completed";
}

/**
 * Run status (TanStack Query). SSE drives refreshes; a slow poll is the safety net for a
 * dropped stream, faster while the stream is reconnecting.
 */
export function useRun(runId: string, streamDown = false) {
  return useQuery({
    queryKey: keys.run(runId),
    queryFn: () => getApi().getRun(runId),
    enabled: !!runId,
    refetchInterval: (q) => {
      const run = q.state.data;
      if (!run || run.status === "completed") return false;
      if (run.status === "awaiting_approval") return 15_000;
      return streamDown ? 3_000 : 10_000;
    },
  });
}

/** Progress events over SSE (EventSource). Replays from the start, then follows the run. */
export function useRunEvents(runId: string, enabled: boolean) {
  const qc = useQueryClient();
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [ended, setEnded] = useState(false);
  const [down, setDown] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    const seen = new Set<number>();
    const unsubscribe = getApi().subscribe(runId, {
      onEvent(e) {
        setDown(false);
        if (seen.has(e.seq)) return;
        seen.add(e.seq);
        setEvents((prev) => [...prev, e].sort((a, b) => a.seq - b.seq));
        if (e.type !== "step") {
          // plan_ready, plan_decided, run_finished, run_error: the run view changed
          void qc.invalidateQueries({ queryKey: keys.run(runId) });
        }
        if (e.type === "run_finished") void qc.invalidateQueries({ queryKey: keys.conversations });
      },
      onEnd() {
        setEnded(true);
        void qc.invalidateQueries({ queryKey: keys.run(runId) });
        void qc.invalidateQueries({ queryKey: keys.trace(runId) });
      },
      onError() {
        setDown(true);
      },
    });
    return unsubscribe;
  }, [runId, enabled, qc]);

  return { events, ended, down };
}

/** Matches a media query (false during server rendering). */
export function useMedia(query: string): boolean {
  const [match, setMatch] = useState(false);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setMatch(m.matches);
    on();
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, [query]);
  return match;
}

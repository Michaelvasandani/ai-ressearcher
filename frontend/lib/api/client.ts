import { auth } from "@/lib/auth";
import type {
  ConversationView,
  Conversation,
  PlanDecision,
  ProfileView,
  ProgressEvent,
  ResearchCreated,
  ResearchRequest,
  RunView,
  StepOverride,
  TraceView,
  UserProfile,
} from "./types";

/** Everything the UI asks of the backend. `HttpApi` talks to FastAPI, `MockApi` fakes it. */
export interface Api {
  readonly mock: boolean;
  startResearch(body: ResearchRequest): Promise<ResearchCreated>;
  getRun(runId: string): Promise<RunView>;
  decidePlan(runId: string, decision: PlanDecision): Promise<ResearchCreated>;
  getTrace(runId: string): Promise<TraceView>;
  fork(runId: string, fromStep: number, override: StepOverride): Promise<ResearchCreated>;
  listConversations(): Promise<Conversation[]>;
  getConversation(id: string): Promise<ConversationView>;
  deleteConversation(id: string): Promise<void>;
  getProfile(): Promise<ProfileView>;
  putProfile(profile: UserProfile): Promise<ProfileView>;
  /** Progress events after `after`; `onEnd` once the run is finished. Returns an unsubscribe. */
  subscribe(runId: string, handlers: StreamHandlers, after?: number): () => void;
}

export interface StreamHandlers {
  onEvent(event: ProgressEvent): void;
  onEnd?(payload: { status?: string; outcome_status?: string | null }): void;
  onError?(): void;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const EVENT_TYPES = [
  "run_started",
  "run_resumed",
  "step",
  "plan_ready",
  "plan_decided",
  "run_finished",
  "run_error",
] as const;

function detail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((e) => (e && typeof e === "object" && "msg" in e ? String(e.msg) : String(e))).join("; ");
  }
  return fallback;
}

export class HttpApi implements Api {
  readonly mock = false;

  constructor(private readonly base: string = process.env.NEXT_PUBLIC_API_BASE || "/api") {}

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      method,
      headers: { ...auth.headers(), ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      cache: "no-store",
    });
    if (res.status === 204) return undefined as T;
    const text = await res.text();
    let parsed: unknown = undefined;
    try {
      parsed = text ? JSON.parse(text) : undefined;
    } catch {
      parsed = text;
    }
    if (!res.ok) throw new ApiError(res.status, detail(parsed, `${method} ${path}: ${res.status}`));
    return parsed as T;
  }

  startResearch(body: ResearchRequest) {
    return this.request<ResearchCreated>("POST", "/research", body);
  }
  getRun(runId: string) {
    return this.request<RunView>("GET", `/research/${runId}`);
  }
  decidePlan(runId: string, decision: PlanDecision) {
    return this.request<ResearchCreated>("POST", `/research/${runId}/plan`, decision);
  }
  getTrace(runId: string) {
    return this.request<TraceView>("GET", `/research/${runId}/trace`);
  }
  fork(runId: string, fromStep: number, override: StepOverride) {
    return this.request<ResearchCreated>("POST", `/research/${runId}/fork?from_step=${fromStep}`, override);
  }
  listConversations() {
    return this.request<Conversation[]>("GET", "/conversations");
  }
  getConversation(id: string) {
    return this.request<ConversationView>("GET", `/conversations/${id}`);
  }
  deleteConversation(id: string) {
    return this.request<void>("DELETE", `/conversations/${id}`);
  }
  getProfile() {
    return this.request<ProfileView>("GET", "/profile");
  }
  putProfile(profile: UserProfile) {
    return this.request<ProfileView>("PUT", "/profile", profile);
  }

  subscribe(runId: string, handlers: StreamHandlers, after = 0): () => void {
    auth.userId(); // make sure the cookie proxy.ts turns into X-User-Id is set
    const source = new EventSource(`${this.base}/research/${runId}/events${after ? `?after=${after}` : ""}`);
    let closed = false;
    const listener = (type: string) => (e: MessageEvent<string>) => {
      try {
        const data = JSON.parse(e.data) as ProgressEvent;
        handlers.onEvent({ ...data, type: type as ProgressEvent["type"] });
      } catch {
        /* a malformed event is skipped; the run view is the source of truth */
      }
    };
    for (const type of EVENT_TYPES) source.addEventListener(type, listener(type) as EventListener);
    source.addEventListener("end", ((e: MessageEvent<string>) => {
      closed = true;
      source.close();
      try {
        handlers.onEnd?.(JSON.parse(e.data));
      } catch {
        handlers.onEnd?.({});
      }
    }) as EventListener);
    source.onerror = () => {
      // EventSource reconnects by itself with Last-Event-ID; tell the caller so it can poll meanwhile.
      if (!closed) handlers.onError?.();
    };
    return () => {
      closed = true;
      source.close();
    };
  }
}

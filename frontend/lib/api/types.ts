/**
 * Types mirroring the FastAPI backend (backend/api/models.py, backend/schemas.py,
 * backend/graph/{trace,fork,plan_review,state}.py). Only the fields the UI reads are typed
 * strictly; everything else is carried along as-is.
 */

export type EvidenceStatus = "consensus" | "leaning" | "genuinely_contested";
export type Confidence = "high" | "moderate" | "low";
export type EvidenceGrade = "high" | "moderate" | "low" | "very_low";
export type StudyType =
  | "meta_analysis"
  | "rct"
  | "cohort"
  | "case_control"
  | "case_report"
  | "review"
  | "animal"
  | "in_vitro"
  | "preprint";
export type RunStatus = "running" | "awaiting_approval" | "completed" | "error";
export type OutcomeStatus = "answered" | "degraded" | "referred" | "failed";

export interface UserProfile {
  goal?: string | null;
  sex?: "male" | "female" | "other" | null;
  age?: number | null;
  bodyweight_kg?: number | null;
  training_age_years?: number | null;
  training_status?: "untrained" | "novice" | "intermediate" | "advanced" | null;
  days_per_week?: number | null;
  session_minutes?: number | null;
  equipment?: string | null;
  dietary_pattern?: string | null;
  notes?: string | null;
}

export interface Budget {
  max_sub_questions?: number;
  max_searches_per_sub_question?: number;
  max_total_tokens?: number;
  max_usd?: number;
  wall_clock_s?: number;
}

export interface Position {
  stance: string;
  supporting_findings: string[];
  evidence_grade: EvidenceGrade;
  best_counterargument: string;
}

export interface CoachAnswer {
  status: EvidenceStatus;
  recommendation: string;
  confidence: Confidence;
  why: string;
  positions: Position[];
  what_would_change_this: string;
  applies_to_you: string;
  practical_protocol: string;
  whats_new: string | null;
  cited_finding_ids: string[];
  disclaimer: string;
}

export interface Finding {
  id: string;
  sub_question_id?: string | null;
  claim: string;
  source_id: string;
  source_url: string;
  title?: string | null;
  study_type: StudyType;
  population: string;
  sample_size: number | null;
  effect: string | null;
  published: string; // ISO date
  funding_conflict: boolean | null;
  retracted: boolean;
  snippet: string;
  grade: EvidenceGrade;
}

export interface SubQuestion {
  id: string;
  question: string;
  rationale: string;
  search_terms: string[];
}

export interface PlanApprovalRequest {
  run_id: string;
  question: string;
  sub_questions: SubQuestion[];
  dropped: SubQuestion[];
  max_sub_questions: number;
  error: string | null;
}

export interface SubQuestionEdit {
  id?: string | null;
  question: string;
  rationale?: string | null;
  search_terms?: string[] | null;
}

export interface PlanDecision {
  action: "approve" | "edit";
  sub_questions?: SubQuestionEdit[];
  note?: string | null;
}

export interface Degradation {
  kind: string;
  node: string;
  detail: string;
  sub_question_id?: string | null;
}

export interface RunOutcome {
  status: OutcomeStatus;
  message: string;
  failure: { reason: string; node: string; message: string; budget_kind?: string | null } | null;
  degradations: Degradation[];
  referral_message: string | null;
}

export interface CitationIssue {
  field: string;
  kind: string;
  sentence: string;
  number: string | null;
  cited: string[];
  found_in: string[];
  detail: string;
}

export interface CitationReport {
  attempt: number;
  ok: boolean;
  claims_checked: number;
  numbers_checked: number;
  numbers_from_user: number;
  citations: number;
  issues: CitationIssue[];
}

export interface Spend {
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  llm_calls: number;
  searches: number;
}

export interface StepOverride {
  prompt_addendum?: string | null;
  model?: string | null;
  tool_result?: unknown[] | null;
}

export interface ForkInfo {
  parent_run_id: string;
  from_step: number;
  step_id: string;
  node: string;
  scope: { agent: string; sub_question_id?: string | null; position_no?: number | null; override: StepOverride };
  inherited_steps: number;
}

export interface FieldChange {
  field: string;
  before: unknown;
  after: unknown;
}

export interface PositionChange {
  stance_before: string | null;
  stance_after: string | null;
  grade_before: string | null;
  grade_after: string | null;
  findings_added: string[];
  findings_removed: string[];
}

export interface AnswerDiff {
  original_run_id: string;
  fork_run_id: string;
  from_step: number;
  changed: boolean;
  changes: FieldChange[];
  positions_added: Position[];
  positions_removed: Position[];
  positions_changed: PositionChange[];
  citations_added: string[];
  citations_removed: string[];
  cost_usd_original: number;
  cost_usd_fork: number;
  new_cost_usd: number;
}

export interface RunView {
  run_id: string;
  status: RunStatus;
  question: string;
  conversation_id: string | null;
  created_at: string;
  updated_at: string;
  plan: PlanApprovalRequest | null;
  plan_decision_pending: boolean;
  sub_questions: SubQuestion[];
  answer: CoachAnswer | null;
  outcome: RunOutcome | null;
  citations: CitationReport | null;
  findings: Finding[];
  spend: Spend;
  steps: number;
  error: string | null;
  parent_run_id: string | null;
  fork: ForkInfo | null;
  diff: AnswerDiff | null;
}

export interface ResearchRequest {
  question: string;
  profile?: UserProfile | null;
  conversation_id?: string | null;
  budget?: Budget | null;
  require_plan_approval?: boolean;
}

export interface ResearchCreated {
  run_id: string;
  conversation_id: string | null;
  status: RunStatus;
  parent_run_id?: string | null;
}

export interface CallRecord {
  call_id: string;
  agent: string;
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  replayed: boolean;
  ok: boolean;
  error?: string | null;
}

export interface ToolCall {
  tool: string;
  input: string;
  state: string;
  results: number;
  latency_ms: number;
  message?: string | null;
}

export interface TraceStep {
  index: number;
  step_id: string;
  run_id: string;
  node: string;
  agent: string | null;
  sub_question_id: string | null;
  position_id: string | null;
  attempt: number;
  model: string | null;
  escalated: boolean;
  route_reason: string | null;
  input_summary: string;
  output_summary: string;
  output: Record<string, unknown> | null;
  decision: string | null;
  status: "ok" | "degraded" | "failed" | "skipped";
  error: string | null;
  llm_calls: CallRecord[];
  tool_calls: ToolCall[];
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  latency_ms: number;
  started_at: string;
  finished_at: string;
}

export interface TraceView {
  run_id: string;
  status: RunStatus;
  steps: TraceStep[];
  inherited_steps: number;
  spend: Spend;
  totals: Record<string, number>;
}

export interface Conversation {
  id: string;
  user_id: string | null;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface RunSummary {
  run_id: string;
  question: string;
  status: RunStatus;
  outcome_status: string | null;
  cost_usd: number;
  parent_run_id: string | null;
  created_at: string;
}

export interface ConversationView {
  conversation: Conversation;
  runs: RunSummary[];
}

export interface ProfileView {
  user_id: string;
  profile: UserProfile;
}

/** Running totals carried by `step` and `run_finished` events. */
export interface Totals {
  tokens: number;
  cost_usd: number;
  llm_calls?: number;
  searches?: number;
}

export type ProgressEventType =
  | "run_started"
  | "run_resumed"
  | "step"
  | "plan_ready"
  | "plan_decided"
  | "run_finished"
  | "run_error";

/** One SSE event as the UI keeps it (`data` of the server event, plus its type and seq). */
export interface ProgressEvent {
  seq: number;
  type: ProgressEventType;
  at?: string;
  message?: string;
  node?: string;
  agent?: string | null;
  status?: string;
  sub_question_id?: string | null;
  tokens?: number;
  cost_usd?: number;
  totals?: Totals;
  [key: string]: unknown;
}

/** Agents whose steps can be forked (backend/graph/overrides.py LLM_AGENTS). */
export const FORKABLE_AGENTS = new Set([
  "intake",
  "planner",
  "researcher",
  "position_mapper",
  "advocate",
  "judge",
  "writer",
]);

export const MODELS = [
  { id: "claude-haiku-5-5", label: "Haiku 5.5" },
  { id: "claude-sonnet-5-5", label: "Sonnet 5.5" },
  { id: "claude-opus-5-5", label: "Opus 5.5" },
] as const;

-- Evidence Coach: run state, progress events, topic findings, LLM call ledger,
-- conversations and profiles, plus the LangGraph Postgres checkpointer tables.
--
-- The backend connects with the service role (DATABASE_URL), which bypasses RLS.
-- RLS is enabled on every table so the anon / authenticated keys see nothing by
-- default; the few policies below let a signed-in user read their own rows
-- (and edit their own profile) if the frontend ever queries Supabase directly.

-- LangGraph checkpointer (langgraph-checkpoint-postgres 3.x schema) ----------
-- Same DDL as PostgresSaver.setup(); the version rows mark its migrations done,
-- so setup() is a no-op here and only applies migrations newer than these.

create table if not exists public.checkpoint_migrations (
    v integer primary key
);

create table if not exists public.checkpoints (
    thread_id text not null,
    checkpoint_ns text not null default '',
    checkpoint_id text not null,
    parent_checkpoint_id text,
    type text,
    checkpoint jsonb not null,
    metadata jsonb not null default '{}',
    primary key (thread_id, checkpoint_ns, checkpoint_id)
);

create table if not exists public.checkpoint_blobs (
    thread_id text not null,
    checkpoint_ns text not null default '',
    channel text not null,
    version text not null,
    type text not null,
    blob bytea,
    primary key (thread_id, checkpoint_ns, channel, version)
);

create table if not exists public.checkpoint_writes (
    thread_id text not null,
    checkpoint_ns text not null default '',
    checkpoint_id text not null,
    task_id text not null,
    idx integer not null,
    channel text not null,
    type text,
    blob bytea not null,
    task_path text not null default '',
    primary key (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

create index if not exists checkpoints_thread_id_idx on public.checkpoints (thread_id);
create index if not exists checkpoint_blobs_thread_id_idx on public.checkpoint_blobs (thread_id);
create index if not exists checkpoint_writes_thread_id_idx on public.checkpoint_writes (thread_id);

insert into public.checkpoint_migrations (v)
select generate_series(0, 9)
on conflict (v) do nothing;

-- Conversations and profiles (used by /conversations and /profile) -----------

create table if not exists public.conversations (
    id text primary key,
    user_id text,
    title text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists conversations_user_idx on public.conversations (user_id, updated_at desc);

create table if not exists public.profiles (
    user_id text primary key,
    profile jsonb not null default '{}',
    updated_at timestamptz not null default now()
);

-- Runs: one row per research run (the graph thread id is the run id) ----------

create table if not exists public.runs (
    run_id text primary key,
    user_id text,
    conversation_id text references public.conversations (id) on delete set null,
    question text not null,
    profile jsonb,
    require_plan_approval boolean not null default false,
    status text not null default 'running'
        check (status in ('running', 'awaiting_approval', 'completed', 'error')),
    outcome_status text,
    outcome jsonb,
    cost_usd double precision not null default 0,
    error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists runs_user_idx on public.runs (user_id, created_at desc);
create index if not exists runs_conversation_idx on public.runs (conversation_id, created_at);
create index if not exists runs_status_idx on public.runs (status, updated_at);

-- Progress events, polled by the SSE endpoint (seq > last seen) ---------------

create table if not exists public.run_events (
    run_id text not null references public.runs (run_id) on delete cascade,
    seq integer not null,
    type text not null,
    key text,
    data jsonb not null default '{}',
    created_at timestamptz not null default now(),
    primary key (run_id, seq)
);
create unique index if not exists run_events_key_idx on public.run_events (run_id, key) where key is not null;

-- Topic findings store: findings by normalized topic, reused across runs ------

create table if not exists public.topic_findings (
    topic_key text primary key,
    tokens text[] not null,
    question text not null,
    sub_question_id text not null,
    findings jsonb not null,
    run_id text not null,
    created_at timestamptz not null default now()
);
create index if not exists topic_findings_tokens_idx on public.topic_findings using gin (tokens);
create index if not exists topic_findings_created_idx on public.topic_findings (created_at);

-- LLM call ledger: one row per completed call, keyed so a retried step is free ---

create table if not exists public.llm_calls (
    run_id text not null,
    call_key text not null,
    call_id text not null,
    agent text not null,
    model text not null,
    cost_usd double precision not null default 0,
    record jsonb not null,
    output_json text not null,
    created_at timestamptz not null default now(),
    primary key (run_id, call_key)
);

-- Row level security -----------------------------------------------------------

alter table public.checkpoint_migrations enable row level security;
alter table public.checkpoints enable row level security;
alter table public.checkpoint_blobs enable row level security;
alter table public.checkpoint_writes enable row level security;
alter table public.conversations enable row level security;
alter table public.profiles enable row level security;
alter table public.runs enable row level security;
alter table public.run_events enable row level security;
alter table public.topic_findings enable row level security;
alter table public.llm_calls enable row level security;

-- Checkpoints, topic findings and the ledger: service role only (no policies).

create policy "read own conversations" on public.conversations
    for select to authenticated using (user_id = (select auth.uid())::text);

create policy "read own profile" on public.profiles
    for select to authenticated using (user_id = (select auth.uid())::text);
create policy "update own profile" on public.profiles
    for update to authenticated
    using (user_id = (select auth.uid())::text)
    with check (user_id = (select auth.uid())::text);

create policy "read own runs" on public.runs
    for select to authenticated using (user_id = (select auth.uid())::text);

create policy "read own run events" on public.run_events
    for select to authenticated using (
        exists (
            select 1 from public.runs r
            where r.run_id = run_events.run_id and r.user_id = (select auth.uid())::text
        )
    );

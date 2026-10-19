-- Phase 6: serverless execution (worker lease + reaper), plan decisions queued for the
-- worker, the run's budget, and time-travel forks.
--
-- A worker advances a run only while it holds the lease (lease_owner, lease_expires_at).
-- Taking it is one conditional UPDATE, so of two concurrent workers at most one wins:
--   update runs set lease_owner = $me, lease_expires_at = now() + ttl
--   where run_id = $id and status = 'running'
--     and (lease_expires_at is null or lease_expires_at < now() or lease_owner = $me)
-- The reaper (a cron route) re-dispatches runs whose lease expired: the worker died or hit
-- the platform's time limit mid-slice.

alter table public.runs add column if not exists budget jsonb;
alter table public.runs add column if not exists lease_owner text;
alter table public.runs add column if not exists lease_expires_at timestamptz;
alter table public.runs add column if not exists attempts integer not null default 0;
alter table public.runs add column if not exists pending_decision jsonb;
alter table public.runs add column if not exists parent_run_id text
    references public.runs (run_id) on delete set null;
alter table public.runs add column if not exists fork jsonb;


create index if not exists runs_reap_idx on public.runs (status, lease_expires_at)
    where status in ('running', 'error');
create index if not exists runs_parent_idx on public.runs (parent_run_id) where parent_run_id is not null;

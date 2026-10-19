# Evidence Coach frontend

Next.js 16 (App Router) + TypeScript + Tailwind 4, shadcn/ui-style primitives (Radix) restyled
monochrome, TanStack Query for run status and history, `EventSource` for the progress stream,
`react-markdown` with citation-chip renderers, Geist / Geist Mono.

## Setup

```bash
cd frontend
npm install
npm run dev:mock                                   # no backend: fixture answers + fake SSE, http://localhost:3000
BACKEND_URL=http://127.0.0.1:8000 npm run dev      # against the FastAPI backend (see backend/README.md)
```

Checks:

```bash
npm run lint && npm run typecheck && npm test && npm run build
```

`npm test` runs Vitest + Testing Library (`tests/`): citation markers and chips (`[n]` ->
`cited_finding_ids[n-1]`, hover details, PubMed links), status markers and the confidence bar,
the both-sides omission rule (never for consensus), failed / referred cards, and the mock API's
run lifecycle.

## Environment

| Variable | Where | Meaning |
|---|---|---|
| `BACKEND_URL` | server, **build time** | FastAPI base URL. `/api/:path*` is rewritten to `${BACKEND_URL}/:path*` (`next.config.ts`). Default `http://127.0.0.1:8000`. |
| `NEXT_PUBLIC_MOCK` | browser, build time | `1` = demo mode: `lib/mock/` serves fixture runs, plan checkpoint, SSE-like events, trace, forks and diffs, all in memory. |
| `NEXT_PUBLIC_API_BASE` | browser, build time | Base path for API calls. Default `/api` (the rewrite); only change it to call a backend directly (then set `CORS_ORIGINS` on the backend). |

## How it talks to the backend

- The browser only calls its own origin: `/api/*` is rewritten to the backend, so no CORS, the
  (future) Supabase session cookie stays first-party, and SSE streams through the rewrite.
  `compress: false` keeps `next start` from buffering the stream.
- **Auth seam** (`lib/auth.ts`): today an anonymous id (`anon-<uuid>`) is created once, kept in
  localStorage and sent as `X-User-Id`. `EventSource` cannot set headers, so the id is mirrored in
  a first-party cookie (`ec_uid`) and `proxy.ts` turns it into `X-User-Id` on `/api/*`.
  For Supabase Auth: implement `AuthProvider.headers()` to return `Authorization: Bearer <token>`
  and read the session cookie in `proxy.ts`; set `SUPABASE_JWT_SECRET` on the backend.
- Endpoints used: `POST /research` (with `conversation_id` for follow-ups, `budget.max_usd`,
  `require_plan_approval` from Settings), `GET /research/{id}`, `GET /research/{id}/events`
  (SSE; `end` closes the stream, reconnects resume with `Last-Event-ID`), `POST /research/{id}/plan`,
  `GET /research/{id}/trace`, `POST /research/{id}/fork?from_step=N`, `GET/DELETE /conversations`,
  `GET/PUT /profile`. A slow `GET /research/{id}` poll backs up a dropped stream.

## Layout

| Path | What it is |
|---|---|
| `app/(chat)/layout.tsx` | Mounts `ChatShell` once for `/` (new chat) and `/c/[id]` (a conversation) |
| `components/ChatShell.tsx` | Sidebar + turns + composer + trace drawer; starts runs and follow-ups |
| `components/Turn.tsx` | One run: question, live progress, plan checkpoint, answer / referral / failure, fork label + diff |
| `components/ProgressSteps.tsx` | SSE step lines with running tokens and dollars; collapses to one line when done |
| `components/PlanCard.tsx` | Sub-questions with Approve / Edit (edit, add, remove; validated like the backend) |
| `components/AnswerCard.tsx`, `BothSides.tsx` | Headline + status marker + confidence bar, why, both sides (omitted for consensus), applies to you / what would change this / what's new, protocol box, disclaimer |
| `components/CitationChip.tsx`, `Markdown.tsx`, `lib/citations.ts` | `[n]` markers -> chips; hover: study type, n, year, population; click: PubMed or the source URL |
| `components/OutcomeCards.tsx` | Out-of-scope referral, failed runs (`citations_failed` shows the gate's issues and the evidence found), interrupted runs |
| `components/TraceDrawer.tsx`, `DiffView.tsx` | Step timeline (agent, model, tokens, latency, cost, tool and LLM calls, JSON), "Fork from here" with prompt addendum / model / tool result, then the diff |
| `components/Sidebar.tsx` | New chat, history by date, profile (feeds intake), settings (theme, plan review, budget); a drawer on mobile |
| `components/StatusMarker.tsx` | ● ◐ ○ (drawn as SVG) and the 3-segment confidence bar |
| `lib/api/` | Types mirroring the backend, `HttpApi`, `getApi()` (mock or real) |
| `lib/mock/` | `MockApi` + fixtures (leaning, consensus, referral, citations_failed scenarios) |
| `app/globals.css` | Color tokens (black, white, grays; dark default, `[data-theme=light]` inverts), the 13/15/18/24 type scale, 6/8px radii, motion |
| `scripts/screenshots.mjs` | Playwright walkthrough that saves screenshots to `screenshots/` |

Mock questions: anything about protein / cuts gives a consensus answer, injuries / pain / medication
a referral, "citation" / "fail" a `citations_failed` run, everything else a leaning answer with
both sides.

## Screenshots

```bash
npm run dev:mock -- -p 3100 &
BASE_URL=http://localhost:3100 npm run screenshots                       # mock-*.png
BASE_URL=http://localhost:3000 LIVE=1 npm run screenshots                # live-*.png, real backend, costs ~$0.15-0.30
```

It uses a preinstalled Chromium (`CHROMIUM_PATH`, or the newest `chromium-*` under
`PLAYWRIGHT_BROWSERS_PATH`, default `/opt/pw-browsers`); it never downloads browsers.

## Deploy on Vercel

The frontend is its own Vercel project (the backend is another one at the repo root, see
`backend/README.md`):

1. New Project -> import this repo -> **Root Directory: `frontend`** (framework: Next.js).
2. Environment variables: `BACKEND_URL=https://<backend-project>.vercel.app` (no trailing slash).
   Leave `NEXT_PUBLIC_MOCK` unset (or set it to `1` for a public demo without a backend).
3. Deploy. Rewrites are resolved at build time, so changing `BACKEND_URL` needs a redeploy.
   If the backend sits behind Deployment Protection, point `BACKEND_URL` at a domain without it.
4. Long runs: SSE responses last at most the backend's `SSE_MAX_S` (240s); `EventSource`
   reconnects with `Last-Event-ID` by itself, and the UI also polls the run.

"""Vercel Python function: the whole FastAPI app (backend/api) behind one function.

vercel.json rewrites every path to /api/index, and the app routes on the original path
(/research, /research/{id}/events, /internal/..., ...). The repo root goes on sys.path so the
`backend` package imports the same way it does locally (pyproject maps backend/ -> `backend`).
Configuration comes from Vercel's environment variables; backend/.env is loaded only if present
(local `vercel dev`), never overriding what is already set.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api import create_app  # noqa: E402

app = create_app()

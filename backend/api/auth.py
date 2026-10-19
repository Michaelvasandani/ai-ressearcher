"""Minimal auth.

- `AUTH_MODE=header` (default without SUPABASE_JWT_SECRET): the caller says who it is in
  `X-User-Id`. Fine for local dev and a single-user demo; anyone can claim any id.
- `AUTH_MODE=jwt` (default when SUPABASE_JWT_SECRET is set): `Authorization: Bearer <Supabase
  access token>`, verified with the project's legacy HS256 JWT secret; the user id is `sub`.
  `X-User-Id` is ignored.

No credentials means an anonymous caller (user id None). Anonymous runs are readable by anyone
who has the run id; a user's runs, conversations and profile only by that user.

TODO: Supabase projects on asymmetric JWT signing keys need JWKS verification
(`https://<ref>.supabase.co/auth/v1/.well-known/jwks.json`) instead of the shared secret.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from backend.api.settings import ApiSettings

MAX_USER_ID = 128


def user_from_request(request: Request, settings: ApiSettings) -> str | None:
    auth = request.headers.get("authorization") or ""
    if settings.auth == "jwt":
        if not auth.lower().startswith("bearer "):
            return None
        if not settings.supabase_jwt_secret:
            raise HTTPException(500, "AUTH_MODE=jwt needs SUPABASE_JWT_SECRET")
        import jwt

        try:
            claims = jwt.decode(
                auth[7:].strip(), settings.supabase_jwt_secret, algorithms=["HS256"], audience="authenticated"
            )
        except jwt.PyJWTError as e:
            raise HTTPException(401, f"invalid token: {e}") from e
        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub:
            raise HTTPException(401, "token has no subject")
        return sub
    user = (request.headers.get("x-user-id") or "").strip()
    if not user:
        return None
    if len(user) > MAX_USER_ID:
        raise HTTPException(400, "X-User-Id is too long")
    return user


def check_secret(request: Request, *secrets: str | None) -> None:
    """Internal routes: `Authorization: Bearer <secret>` must match one of the configured secrets."""
    configured = [s for s in secrets if s]
    if not configured:
        raise HTTPException(503, "internal routes are disabled: set WORKER_SECRET")
    auth = request.headers.get("authorization") or ""
    given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not given or not any(hmac.compare_digest(given.encode(), s.encode()) for s in configured):
        raise HTTPException(401, "bad worker secret")

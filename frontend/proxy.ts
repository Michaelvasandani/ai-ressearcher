import { NextResponse, type NextRequest } from "next/server";

/**
 * Runs before the `/api/*` rewrite to the backend. EventSource cannot send headers, so the
 * anonymous user id travels as a first-party cookie and becomes `X-User-Id` here. A request
 * that already sets the header (fetch calls do) is left alone.
 *
 * Supabase Auth later: read the session cookie here and set `Authorization: Bearer <token>`.
 */
export function proxy(request: NextRequest) {
  const uid = request.cookies.get("ec_uid")?.value;
  if (!uid || request.headers.has("x-user-id")) return NextResponse.next();
  const headers = new Headers(request.headers);
  headers.set("x-user-id", decodeURIComponent(uid).slice(0, 128));
  return NextResponse.next({ request: { headers } });
}

export const config = { matcher: "/api/:path*" };

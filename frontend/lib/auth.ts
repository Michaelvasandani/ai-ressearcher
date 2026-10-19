/**
 * Who the browser is, as far as the API is concerned.
 *
 * Today: an anonymous id, generated once and kept in localStorage, sent as `X-User-Id`.
 * It is mirrored into a first-party cookie so requests the browser makes on its own
 * (EventSource cannot set headers) carry it too: `proxy.ts` turns the cookie into the
 * header on the way to the backend.
 *
 * Later (Supabase Auth): implement `AuthProvider` with the Supabase client, return
 * `{ Authorization: "Bearer <access_token>" }` from `headers()`, and have `proxy.ts` read
 * the Supabase session cookie instead. Nothing else in the app changes.
 */

export interface AuthProvider {
  /** Stable id of the current user (anonymous or signed in). */
  userId(): string;
  /** Headers every API request carries. */
  headers(): Record<string, string>;
  /** Short label for the sidebar. */
  label(): string;
}

export const USER_ID_KEY = "ec.userId";
export const USER_ID_COOKIE = "ec_uid";

function randomId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return `anon-${crypto.randomUUID()}`;
  return `anon-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* private mode: the id lives for this page only */
  }
}

export class AnonymousAuth implements AuthProvider {
  private id: string | null = null;

  userId(): string {
    if (this.id) return this.id;
    if (typeof window === "undefined") return "anon-server";
    let id = readStorage(USER_ID_KEY);
    if (!id) {
      id = randomId();
      writeStorage(USER_ID_KEY, id);
    }
    this.id = id;
    document.cookie = `${USER_ID_COOKIE}=${encodeURIComponent(id)}; path=/; max-age=31536000; samesite=lax`;
    return id;
  }

  headers(): Record<string, string> {
    return { "X-User-Id": this.userId() };
  }

  label(): string {
    return `Guest ${this.userId().replace(/^anon-/, "").slice(0, 6)}`;
  }
}

export const auth: AuthProvider = new AnonymousAuth();

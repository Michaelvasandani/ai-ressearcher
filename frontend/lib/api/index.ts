import { HttpApi, type Api } from "./client";
import { MockApi } from "@/lib/mock/api";

export const MOCK = process.env.NEXT_PUBLIC_MOCK === "1" || process.env.NEXT_PUBLIC_MOCK === "true";

let instance: Api | null = null;

/** The API for this page: the mock backend in demo mode, else FastAPI through `/api`. */
export function getApi(): Api {
  instance ??= MOCK ? new MockApi() : new HttpApi();
  return instance;
}

export * from "./client";
export type * from "./types";

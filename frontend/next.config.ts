import type { NextConfig } from "next";

/**
 * The browser only ever talks to its own origin: `/api/*` is rewritten to the FastAPI
 * backend (a separate Vercel project, see backend/README.md), so there is no CORS, the
 * Supabase session cookie stays first-party, and EventSource streams through the rewrite.
 * Rewrites are resolved at build time: change BACKEND_URL, redeploy.
 */
const backend = (process.env.BACKEND_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  // Gzip would buffer the SSE stream in `next start`; Vercel compresses at the edge anyway.
  compress: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/:path*` }];
  },
};

export default nextConfig;

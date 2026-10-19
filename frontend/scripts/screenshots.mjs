#!/usr/bin/env node
/**
 * Drive the UI with Playwright and save screenshots: desktop dark, desktop light, mobile,
 * plus the plan checkpoint and the trace drawer.
 *
 *   BASE_URL=http://localhost:3000 node scripts/screenshots.mjs            # mock mode server
 *   BASE_URL=... LIVE=1 QUESTION="..." node scripts/screenshots.mjs        # real backend (slow, costs money)
 *
 * Uses a preinstalled Chromium (CHROMIUM_PATH, or the one under PLAYWRIGHT_BROWSERS_PATH).
 */
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import path from "node:path";
import { chromium } from "playwright-core";

const BASE = process.env.BASE_URL || "http://localhost:3000";
const LIVE = process.env.LIVE === "1";
const QUESTION =
  process.env.QUESTION || "I'm a woman, lifting for 3 years, 4 days a week. How many sets per week for hypertrophy?";
const OUT = path.resolve(process.env.OUT_DIR || "screenshots");
const PREFIX = process.env.PREFIX || (LIVE ? "live-" : "mock-");
const ANSWER_TIMEOUT = LIVE ? 420_000 : 60_000;

function chromiumPath() {
  if (process.env.CHROMIUM_PATH) return process.env.CHROMIUM_PATH;
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH || "/opt/pw-browsers";
  const dir = readdirSync(root).filter((d) => /^chromium-\d+$/.test(d)).sort().pop();
  const exe = path.join(root, dir, "chrome-linux", "chrome");
  if (!existsSync(exe)) throw new Error(`no chromium at ${exe}`);
  return exe;
}

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ executablePath: chromiumPath() });
const shot = async (page, name) => {
  const file = path.join(OUT, `${PREFIX}${name}.png`);
  await page.screenshot({ path: file, fullPage: false });
  console.log("saved", file);
};

async function setTheme(page, theme) {
  await page.evaluate((t) => {
    const s = JSON.parse(localStorage.getItem("ec.settings") || "{}");
    localStorage.setItem("ec.settings", JSON.stringify({ ...s, theme: t, maxUsd: 0.3, requirePlanApproval: true }));
  }, theme);
}

try {
  // --- Desktop, dark: ask, approve the plan, wait for the answer -----------------------
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  page.on("console", (m) => m.type() === "error" && console.log("console error:", m.text()));
  await page.goto(BASE);
  await setTheme(page, "dark");
  await page.reload();
  await page.waitForSelector("text=What does the research say?");
  await shot(page, "desktop-dark-empty");

  await page.getByLabel("Question").fill(QUESTION);
  await page.keyboard.press("Enter");
  await page.waitForSelector('[data-testid="plan-card"]', { timeout: LIVE ? 120_000 : 20_000 });
  await page.waitForTimeout(400);
  await shot(page, "desktop-dark-plan");
  await page.getByRole("button", { name: "Approve and research" }).click();

  // mid-run progress
  await page.waitForFunction(() => document.body.innerText.includes("Researching"), null, { timeout: 60_000 });
  await page.waitForTimeout(LIVE ? 4_000 : 1_500);
  await shot(page, "desktop-dark-progress");

  await page.waitForSelector('[data-testid="answer-card"], [data-testid="failed-card"], [data-testid="referral-card"]', {
    timeout: ANSWER_TIMEOUT,
  });
  await page.waitForTimeout(800);
  const card = page.locator('[data-testid="answer-card"], [data-testid="failed-card"], [data-testid="referral-card"]').first();
  await card.scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, -140));
  await shot(page, "desktop-dark-answer");

  // citation hover
  const chip = page.locator("a[data-cite]").first();
  if (await chip.count()) {
    await chip.scrollIntoViewIfNeeded();
    await chip.hover();
    await page.waitForTimeout(500);
    await shot(page, "desktop-dark-citation");
  }

  // trace drawer, one step expanded
  await page.getByRole("button", { name: "Open trace" }).first().click();
  await page.waitForSelector('[data-testid="trace-drawer"] li');
  const writer = page.locator('[data-testid="trace-drawer"] li button', { hasText: "writer" }).first();
  if (await writer.count()) {
    await writer.click();
    await page.getByRole("button", { name: "⑂ Fork from here" }).first().click();
    await page.waitForTimeout(300);
  }
  await shot(page, "desktop-dark-trace");
  const url = page.url();

  // --- Desktop, light --------------------------------------------------------------------
  await page.keyboard.press("Escape");
  // Toggle in the page (the sidebar button): mock state lives in memory, a reload would lose it.
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await page.waitForSelector('[data-testid="answer-card"], [data-testid="failed-card"], [data-testid="referral-card"]', { timeout: 30_000 });
  await page.waitForTimeout(600);
  await page.locator('[data-testid="answer-card"], [data-testid="failed-card"]').first().scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, -140));
  await shot(page, "desktop-light-answer");
  const both = page.locator('[data-testid="both-sides"]');
  if (await both.count()) {
    await both.scrollIntoViewIfNeeded();
    await page.evaluate(() => window.scrollBy(0, -80));
    await shot(page, "desktop-light-both-sides");
  }
  const state = await ctx.storageState();
  await ctx.close();

  // --- Mobile ----------------------------------------------------------------------------
  const mctx = await browser.newContext({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
    storageState: state, // same anonymous user, so the live conversation is visible
  });
  const m = await mctx.newPage();
  await m.goto(BASE);
  await setTheme(m, "dark");
  if (LIVE) {
    await m.goto(url);
  } else {
    // mock state lives in the page: ask again here
    await m.reload();
    await m.waitForSelector("text=What does the research say?");
    await shot(m, "mobile-dark-empty");
    await m.getByLabel("Question").fill(QUESTION);
    await m.getByLabel("Send").click();
    await m.waitForSelector('[data-testid="plan-card"]');
    await m.getByRole("button", { name: "Approve and research" }).click();
  }
  const mcard = m.locator('[data-testid="answer-card"], [data-testid="failed-card"], [data-testid="referral-card"]').first();
  await mcard.waitFor({ timeout: ANSWER_TIMEOUT });
  await m.waitForTimeout(600);
  await mcard.scrollIntoViewIfNeeded();
  await m.evaluate(() => window.scrollBy(0, -60));
  await shot(m, "mobile-dark-answer");
  const mboth = m.locator('[data-testid="both-sides"]');
  if (await mboth.count()) {
    await mboth.scrollIntoViewIfNeeded();
    await shot(m, "mobile-dark-both-sides");
  }
  await m.getByLabel("Open sidebar").click();
  await m.waitForTimeout(400);
  await shot(m, "mobile-dark-sidebar");
  await mctx.close();
} finally {
  await browser.close();
}

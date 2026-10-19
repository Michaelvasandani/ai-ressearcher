"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

export type Theme = "dark" | "light";

export interface Settings {
  theme: Theme;
  /** Pause after planning so the user can approve or edit the sub-questions. */
  requirePlanApproval: boolean;
  /** Dollar cap per run (Budget.max_usd). */
  maxUsd: number;
}

export const DEFAULT_SETTINGS: Settings = { theme: "dark", requirePlanApproval: true, maxUsd: 0.5 };
const KEY = "ec.settings";

interface Ctx {
  settings: Settings;
  update(patch: Partial<Settings>): void;
}

const SettingsContext = createContext<Ctx>({ settings: DEFAULT_SETTINGS, update: () => {} });

function load(): Settings {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);

  useEffect(() => {
    // Read once on mount: localStorage is not available during server rendering.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSettings(load());
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme;
  }, [settings.theme]);

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        window.localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        /* not persisted */
      }
      return next;
    });
  }, []);

  return <SettingsContext.Provider value={{ settings, update }}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  return useContext(SettingsContext);
}

/** Inline in <head>: applies the saved theme before first paint (no flash). */
export const THEME_SCRIPT = `try{var s=JSON.parse(localStorage.getItem("${KEY}")||"{}");document.documentElement.dataset.theme=s.theme==="light"?"light":"dark"}catch(e){}`;

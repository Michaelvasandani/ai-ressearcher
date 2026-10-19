"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { getApi, MOCK } from "@/lib/api";
import type { Conversation, UserProfile } from "@/lib/api/types";
import { auth } from "@/lib/auth";
import { keys } from "@/lib/hooks";
import { useSettings } from "@/lib/settings";
import { cn, dateBucket } from "@/lib/utils";
import { Dot } from "./StatusMarker";
import { Button } from "./ui/button";
import { Dialog, DialogContent } from "./ui/dialog";
import { Field, Input, Select, Textarea } from "./ui/field";

function groupByDate(convs: Conversation[]): [string, Conversation[]][] {
  const groups = new Map<string, Conversation[]>();
  for (const c of [...convs].sort((a, b) => b.updated_at.localeCompare(a.updated_at))) {
    const k = dateBucket(c.updated_at);
    groups.set(k, [...(groups.get(k) ?? []), c]);
  }
  return [...groups.entries()];
}

function num(v: string): number | null {
  const n = Number(v);
  return v.trim() === "" || Number.isNaN(n) ? null : n;
}

function ProfileForm({ initial, onSave, saving }: { initial: UserProfile; onSave: (p: UserProfile) => void; saving: boolean }) {
  const [p, setP] = useState<UserProfile>(initial);
  const set = <K extends keyof UserProfile>(k: K, v: UserProfile[K]) => setP({ ...p, [k]: v });
  return (
    <form
      className="grid grid-cols-2 gap-3"
      onSubmit={(e) => {
        e.preventDefault();
        onSave(p);
      }}
    >
      <div className="col-span-2">
        <Field label="Goal">
          <Input value={p.goal ?? ""} onChange={(e) => set("goal", e.target.value || null)} placeholder="hypertrophy, strength, fat loss…" />
        </Field>
      </div>
      <Field label="Training status">
        <Select value={p.training_status ?? ""} onChange={(e) => set("training_status", (e.target.value || null) as UserProfile["training_status"])}>
          <option value="">—</option>
          <option value="untrained">Untrained</option>
          <option value="novice">Novice</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </Select>
      </Field>
      <Field label="Training age (years)">
        <Input inputMode="decimal" value={p.training_age_years ?? ""} onChange={(e) => set("training_age_years", num(e.target.value))} />
      </Field>
      <Field label="Sex">
        <Select value={p.sex ?? ""} onChange={(e) => set("sex", (e.target.value || null) as UserProfile["sex"])}>
          <option value="">—</option>
          <option value="female">Female</option>
          <option value="male">Male</option>
          <option value="other">Other</option>
        </Select>
      </Field>
      <Field label="Age">
        <Input inputMode="numeric" value={p.age ?? ""} onChange={(e) => set("age", num(e.target.value))} />
      </Field>
      <Field label="Days per week">
        <Input inputMode="numeric" value={p.days_per_week ?? ""} onChange={(e) => set("days_per_week", num(e.target.value))} />
      </Field>
      <Field label="Session length (min)">
        <Input inputMode="numeric" value={p.session_minutes ?? ""} onChange={(e) => set("session_minutes", num(e.target.value))} />
      </Field>
      <div className="col-span-2">
        <Field label="Equipment">
          <Input value={p.equipment ?? ""} onChange={(e) => set("equipment", e.target.value || null)} placeholder="full gym, dumbbells only…" />
        </Field>
      </div>
      <div className="col-span-2">
        <Field label="Notes">
          <Textarea rows={2} value={p.notes ?? ""} onChange={(e) => set("notes", e.target.value || null)} />
        </Field>
      </div>
      <div className="col-span-2 flex justify-end">
        <Button type="submit" variant="primary" disabled={saving}>
          {saving ? "Saving…" : "Save profile"}
        </Button>
      </div>
    </form>
  );
}

function ProfileDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: keys.profile, queryFn: () => getApi().getProfile(), enabled: open });
  const save = useMutation({
    mutationFn: (p: UserProfile) => getApi().putProfile(p),
    onSuccess: (view) => {
      qc.setQueryData(keys.profile, view);
      onOpenChange(false);
    },
  });
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Profile" description="Feeds intake: findings from populations unlike you are down-weighted.">
        {isLoading || !data ? (
          <p className="text-xs text-muted">Loading…</p>
        ) : (
          <ProfileForm initial={data.profile} saving={save.isPending} onSave={(p) => save.mutate(p)} />
        )}
        {save.error ? <p className="mt-2 text-xs">✕ {(save.error as Error).message}</p> : null}
      </DialogContent>
    </Dialog>
  );
}

function SettingsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const { settings, update } = useSettings();
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Settings">
        <div className="flex flex-col gap-4">
          <Field label="Theme">
            <div className="grid grid-cols-2 gap-1 rounded-sm border border-border p-0.5">
              {(["dark", "light"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => update({ theme: t })}
                  aria-pressed={settings.theme === t}
                  className={cn(
                    "h-7 rounded-[4px] text-xs capitalize cursor-pointer",
                    settings.theme === t ? "bg-invert text-invert-fg" : "text-muted hover:text-fg",
                  )}
                >
                  {t}
                </button>
              ))}
            </div>
          </Field>
          <label className="flex items-start gap-3">
            <input
              type="checkbox"
              className="mt-1 accent-current"
              checked={settings.requirePlanApproval}
              onChange={(e) => update({ requirePlanApproval: e.target.checked })}
            />
            <span>
              <span className="block text-sm">Review the research plan</span>
              <span className="block text-xs text-subtle">Pause after planning so you can approve or edit the sub-questions.</span>
            </span>
          </label>
          <Field label="Budget per question" hint="A hard cap enforced in the graph; the run degrades or stops cleanly when it is reached.">
            <Select value={String(settings.maxUsd)} onChange={(e) => update({ maxUsd: Number(e.target.value) })}>
              {[0.2, 0.5, 1].map((v) => (
                <option key={v} value={v}>
                  ${v.toFixed(2)}
                </option>
              ))}
            </Select>
          </Field>
          <p className="font-mono text-[11px] text-subtle">
            {MOCK ? "Demo mode: fixture answers, no backend." : `Signed in anonymously as ${auth.userId()}`}
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** New chat, history grouped by date, profile and settings. Collapses to a drawer on mobile. */
export function Sidebar({ activeId, open, onClose }: { activeId: string | null; open: boolean; onClose: () => void }) {
  const router = useRouter();
  const qc = useQueryClient();
  const { settings, update } = useSettings();
  const [profileOpen, setProfileOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [label, setLabel] = useState("Guest");
  const { data: conversations, isLoading, error } = useQuery({ queryKey: keys.conversations, queryFn: () => getApi().listConversations() });
  const remove = useMutation({
    mutationFn: (id: string) => getApi().deleteConversation(id),
    onSuccess: (_, id) => {
      void qc.invalidateQueries({ queryKey: keys.conversations });
      if (id === activeId) router.push("/");
    },
  });

  useEffect(() => {
    // The anonymous id lives in localStorage, so it is only known after mount.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLabel(MOCK ? "Demo user" : auth.label());
  }, []);

  return (
    <>
      {open ? <div className="fixed inset-0 z-30 bg-overlay md:hidden animate-fade-in" onClick={onClose} aria-hidden /> : null}
      <nav
        aria-label="Conversations"
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-[264px] shrink-0 flex-col border-r border-border bg-surface transition-transform duration-200 ease-out",
          "md:sticky md:top-0 md:h-dvh md:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between px-4 pt-4 pb-3">
          <Link href="/" className="text-sm font-semibold tracking-[-0.01em]" onClick={onClose}>
            Evidence Coach
          </Link>
          <Button variant="ghost" size="icon-sm" className="md:hidden" onClick={onClose} aria-label="Close sidebar">
            ×
          </Button>
        </div>
        <div className="px-3">
          <Button asChild className="w-full justify-start" variant="outline">
            <Link href="/" onClick={onClose}>
              <span aria-hidden className="text-sm leading-none">+</span> New chat
            </Link>
          </Button>
        </div>

        <div className="mt-4 min-h-0 flex-1 overflow-y-auto px-3 pb-4">
          {isLoading ? <p className="px-2 text-xs text-subtle">Loading…</p> : null}
          {error ? <p className="px-2 text-xs text-subtle">History unavailable</p> : null}
          {conversations && !conversations.length ? <p className="px-2 text-xs text-subtle">No conversations yet.</p> : null}
          {groupByDate(conversations ?? []).map(([bucket, convs]) => (
            <div key={bucket} className="mb-4">
              <div className="px-2 pb-1 text-xs text-subtle">{bucket}</div>
              <ul>
                {convs.map((c) => (
                  <li key={c.id} className="group relative">
                    <Link
                      href={`/c/${c.id}`}
                      onClick={onClose}
                      aria-current={c.id === activeId ? "page" : undefined}
                      className={cn(
                        "block truncate rounded-sm py-1.5 pr-7 pl-2 text-xs transition-colors",
                        c.id === activeId ? "bg-surface-2 text-fg" : "text-muted hover:bg-surface-2 hover:text-fg",
                      )}
                    >
                      {c.title}
                    </Link>
                    <button
                      type="button"
                      aria-label={`Delete ${c.title}`}
                      onClick={() => remove.mutate(c.id)}
                      className="absolute top-1/2 right-1 hidden h-5 w-5 -translate-y-1/2 items-center justify-center rounded-[4px] text-subtle hover:text-fg group-hover:flex cursor-pointer"
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-border p-3">
          <button
            type="button"
            onClick={() => setProfileOpen(true)}
            className="flex w-full items-center gap-2.5 rounded-sm px-2 py-1.5 text-left text-xs text-muted hover:bg-surface-2 hover:text-fg cursor-pointer"
          >
            <span aria-hidden className="flex h-6 w-6 items-center justify-center rounded-full border border-border-strong font-mono text-[11px] text-fg">
              {label.slice(0, 1)}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-fg-2">{label}</span>
              <span className="block text-[11px] text-subtle">Profile</span>
            </span>
          </button>
          <div className="mt-1 flex items-center gap-1">
            <Button variant="ghost" size="sm" className="flex-1 justify-start" onClick={() => setSettingsOpen(true)}>
              Settings
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => update({ theme: settings.theme === "dark" ? "light" : "dark" })}
              aria-label={`Switch to ${settings.theme === "dark" ? "light" : "dark"} theme`}
            >
              <Dot fill="half" />
              {settings.theme === "dark" ? "Light" : "Dark"}
            </Button>
          </div>
          {MOCK ? <div className="mt-2 px-2 font-mono text-[11px] text-subtle">demo mode · fixture data</div> : null}
        </div>
      </nav>
      <ProfileDialog open={profileOpen} onOpenChange={setProfileOpen} />
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  );
}

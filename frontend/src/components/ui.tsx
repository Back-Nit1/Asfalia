import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError, api } from "../api/client";

export function Badge({ kind = "info", children }: { kind?: "ok" | "warn" | "bad" | "info"; children: ReactNode }) {
  return <span className={`badge ${kind}`}>{children}</span>;
}

export function NotConfigured({ what, detail }: { what: string; detail?: string }) {
  return (
    <div className="banner" role="status">
      <Badge kind="warn">not configured</Badge> <b>{what}</b>
      {detail && <div className="muted">{detail}</div>}
    </div>
  );
}

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null;
  const e = error instanceof ApiError ? `${error.status}: ${error.message}` : String(error);
  return <div className="banner bad" role="alert"><Badge kind="bad">error</Badge> {e}</div>;
}

export function Card({ title, value, sub }: { title: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="panel card">
      <h2>{title}</h2>
      <div className="v">{value}</div>
      {sub && <div className="s">{sub}</div>}
    </div>
  );
}

/** Polls a JSON endpoint; keeps the last good value and flags staleness instead of blanking. */
export function useJson<T>(path: string | null, everyMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [tries, setTries] = useState(0);
  const [error, setError] = useState<unknown>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const alive = useRef(true);
  const load = useCallback(async () => {
    if (!path) return;
    try {
      const d = await api<T>(path);
      if (alive.current) { setData(d); setError(null); setUpdatedAt(Date.now()); setTries((n) => n + 1); }
    } catch (e) {
      if (alive.current) { setError(e); setTries((n) => n + 1); }
    }
  }, [path]);
  useEffect(() => {
    alive.current = true;
    load();
    const t = everyMs ? setInterval(load, everyMs) : undefined;
    return () => { alive.current = false; if (t) clearInterval(t); };
  }, [load, everyMs]);
  return { data, error, updatedAt, reload: load, loading: tries === 0 };
}

export function Loading({ what }: { what: string }) {
  return <p className="muted" role="status" aria-live="polite">Loading {what}…</p>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="muted" role="status">{children}</p>;
}

/** Honest freshness: if the last poll failed we keep showing the last good data but say so. */
export function Freshness({ error, updatedAt }: { error: unknown; updatedAt: number | null }) {
  if (!error || updatedAt === null) return null;
  const s = Math.round((Date.now() - updatedAt) / 1000);
  return <div className="banner bad" role="alert"><Badge kind="bad">disconnected</Badge> showing last known data from {s}s ago — the backend is not responding.</div>;
}

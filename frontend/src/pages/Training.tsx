import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, mutate } from "../api/client";
import { metricsHz } from "../api/config";
import { openStream } from "../api/sse";
import type { Checkpoint, Evaluation, ExpConfig, MetricRow, Resolved, Run } from "../api/types";
import { Badge, Empty, ErrorBanner, Freshness, Loading, NotConfigured, useJson } from "../components/ui";

export const stateKind = (s: string): "ok" | "warn" | "bad" | "info" =>
  s === "completed" ? "ok" : s === "failed" || s === "cancelled" ? "bad" : s === "queued" || s === "stopped" ? "warn" : "info";
const ACTIVE = ["starting", "running", "draining", "checkpointing"];

function Chart({ rows }: { rows: { env_steps: number; mean_return: number | null }[] }) {
  const pts = rows.filter((r) => r.mean_return !== null);
  if (pts.length < 2) return <Empty>Not enough metric rows to chart yet ({pts.length}).</Empty>;
  const xs = pts.map((p) => p.env_steps), ys = pts.map((p) => p.mean_return as number);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (x: number) => 8 + ((x - x0) / (x1 - x0 || 1)) * 304, sy = (y: number) => 92 - ((y - y0) / (y1 - y0 || 1)) * 84;
  return (
    <svg viewBox="0 0 320 100" role="img" aria-label={`mean return over env steps, ${pts.length} points, ${y0} to ${y1}`} style={{ width: "100%", maxWidth: 480 }}>
      <polyline fill="none" stroke="var(--accent)" strokeWidth="2" points={pts.map((p) => `${sx(p.env_steps)},${sy(p.mean_return as number)}`).join(" ")} />
    </svg>
  );
}

function RunDetail({ run, onChange }: { run: Run; onChange: () => void }) {
  const refresh = 1000 / metricsHz;
  const metrics = useJson<{ metrics: { attempt_id: string; seq: number; env_steps: number; data: MetricRow }[] }>(`/api/runs/${run.id}/metrics`, refresh);
  const cps = useJson<{ checkpoints: Checkpoint[] }>(`/api/runs/${run.id}/checkpoints`, refresh);
  const [err, setErr] = useState<unknown>(null);
  const [live, setLive] = useState<string[]>([]);
  const [conn, setConn] = useState("reconnecting");
  useEffect(() => openStream(`/api/experiments/stream?run_id=${run.id}`, (m) => {
    if (m.event === "envelope") {
      const e = m.data as { event_type: string; payload: Record<string, unknown> };
      setLive((l) => [`${e.event_type} ${e.payload.env_steps ?? e.payload.checkpoint_id ?? ""}`, ...l].slice(0, 8));
    }
  }, setConn), [run.id]);

  const list = cps.data?.checkpoints ?? [];
  const act = async (fn: () => Promise<unknown>) => { setErr(null); try { await fn(); } catch (e) { setErr(e); } onChange(); };
  const rows = (metrics.data?.metrics ?? []).map((m) => ({ env_steps: m.env_steps, mean_return: m.data.mean_return, policy_version: m.data.policy_version }));
  const active = ACTIVE.includes(run.state);

  return (
    <section className="panel" aria-label={`Run ${run.id}`} style={{ marginTop: 12 }}>
      <h2>Run {run.id}</h2>
      <div className="row">
        <Badge kind={stateKind(run.state)}>{run.state}</Badge>
        {run.stale && <Badge kind="warn">stale heartbeat {run.heartbeat_age_s?.toFixed(0)}s</Badge>}
        <span className="muted">backend {run.backend} · owner {run.owner} · attempts {run.attempts.length}</span>
      </div>
      {run.error && <div className="banner bad" role="alert">{run.error}</div>}
      {run.backend_error && <div className="banner bad" role="alert">backend error: {run.backend_error}</div>}
      {run.artifact_problems.length > 0 && <div className="banner" role="status"><Badge kind="warn">partial artifacts</Badge> {run.artifact_problems.join("; ")}</div>}
      <div className="row" style={{ margin: "8px 0" }}>
        <button disabled={run.state !== "running"} onClick={() => act(() => mutate(`/api/runs/${run.id}/drain`))}>Drain</button>
        <button disabled={run.state !== "running"} onClick={() => act(() => mutate(`/api/runs/${run.id}/checkpoint-stop`))}>Checkpoint and stop</button>
        <button disabled={!active} onClick={() => act(() => mutate(`/api/runs/${run.id}/cancel`))}>Force cancel</button>
        <button disabled={!run.resumable} onClick={() => act(() => mutate(`/api/runs/${run.id}/resume`))}>Resume</button>
      </div>
      <ErrorBanner error={err ?? metrics.error} />
      <Freshness error={metrics.error} updatedAt={metrics.updatedAt} />
      <h2>Metrics (ingested from job files)</h2>
      {metrics.loading ? <Loading what="metrics" /> : rows.length === 0
        ? <Empty>No metrics have been written by this run yet. Nothing is simulated.</Empty> : (
          <>
            <Chart rows={rows} />
            <table aria-label="metrics"><thead><tr><th>env steps</th><th>policy v</th><th>mean return</th></tr></thead>
              <tbody>{rows.slice(-6).map((m, i) => <tr key={i}><td>{m.env_steps}</td><td>{m.policy_version}</td><td>{m.mean_return ?? "—"}</td></tr>)}</tbody></table>
          </>
        )}
      <h2 style={{ marginTop: 12 }}>Live feed (sampled, read-only) <span className="muted">{conn}</span></h2>
      {live.length === 0 ? <Empty>No live events yet.</Empty> : <ul aria-label="live feed" className="mono">{live.map((l, i) => <li key={i}>{l}</li>)}</ul>}
      <h2 style={{ marginTop: 12 }}>Checkpoints</h2>
      {cps.loading ? <Loading what="checkpoints" /> : list.length === 0 ? <Empty>No checkpoint has been written yet.</Empty> : (
        <table aria-label="checkpoints"><thead><tr><th>id</th><th>role</th><th>steps</th><th>sha256</th><th>parent</th><th>attempt</th><th /></tr></thead>
          <tbody>{list.map((c) => (
            <tr key={c.id}>
              <td className="mono">{c.id}</td><td>{c.role}</td><td>{c.env_steps}</td><td className="mono" title={c.sha256}>{c.sha256.slice(0, 12)}…</td>
              <td className="mono">{c.parent_id ?? "—"}</td><td>{c.attempt_id}</td>
              <td><button onClick={() => act(async () => { await mutate(`/api/runs/${run.id}/debug-fork?checkpoint_id=${c.id}&tick=2`); })}>Debug fork</button></td>
            </tr>))}</tbody></table>
      )}
    </section>
  );
}

function EvalPanel({ cps, onDone }: { cps: Checkpoint[]; onDone: () => void }) {
  const [picked, setPicked] = useState<string[]>([]);
  const [episodes, setEpisodes] = useState("4");
  const [err, setErr] = useState<unknown>(null);
  const [out, setOut] = useState<Evaluation | null>(null);
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-2)));
  const chosen = picked.map((id) => cps.find((c) => c.id === id)).filter(Boolean) as Checkpoint[];
  const submit = async () => {
    setErr(null);
    const atk = chosen.find((c) => c.role === "attacker") ?? chosen[0], dfn = chosen.find((c) => c.role === "defender") ?? chosen[1];
    try {
      setOut(await mutate<Evaluation>("/api/evaluations", { suite: "frozen", attacker_checkpoint_id: atk.id, defender_checkpoint_id: dfn.id, episodes: Number(episodes) }));
      onDone();
    } catch (e) { setErr(e); }
  };
  return (
    <section className="panel" style={{ marginTop: 12 }} aria-label="Frozen evaluation">
      <h2>Frozen evaluation</h2>
      {cps.length === 0 ? <Empty>No checkpoints exist yet; run training first.</Empty> : (
        <table aria-label="all checkpoints"><thead><tr><th>select</th><th>id</th><th>role</th><th>run</th><th>steps</th><th>available</th></tr></thead>
          <tbody>{cps.map((c) => (
            <tr key={c.id}>
              <td><input type="checkbox" aria-label={`select ${c.id}`} checked={picked.includes(c.id)} onChange={() => toggle(c.id)} disabled={!c.available} /></td>
              <td className="mono">{c.id}</td><td>{c.role}</td><td>{c.run_id}</td><td>{c.env_steps}</td><td>{c.available ? "yes" : "cleaned up"}</td>
            </tr>))}</tbody></table>)}
      <p className="muted">Pairs one attacker with one defender checkpoint (any runs); weights and normalizers stay frozen. Select exactly two.</p>
      <div className="row">
        <label>Episodes <input aria-label="Episodes" type="number" min={1} max={20} value={episodes} onChange={(e) => setEpisodes(e.target.value)} style={{ width: 70 }} /></label>
        <button disabled={chosen.length !== 2} onClick={submit}>Evaluate selected (frozen)</button>
        {out && <span role="status">evaluation {out.id} {out.state} — <Link to="/experiments">see Experiments</Link></span>}
      </div>
      <ErrorBanner error={err} />
    </section>
  );
}

export default function Training() {
  const cfg = useJson<ExpConfig>("/api/experiments/config");
  const runs = useJson<{ runs: Run[] }>("/api/runs", 1000 / metricsHz);
  const allCps = useJson<{ checkpoints: Checkpoint[] }>("/api/checkpoints", 1000 / metricsHz);
  const [preset, setPreset] = useState("");
  const [f, setF] = useState({ seed: "0", total_steps: "", timeout_s: "300", cpu: "1", memory_mb: "2048" });
  const [resolved, setResolved] = useState<Resolved | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [sel, setSel] = useState<string | null>(null);
  const presets = cfg.data?.presets ?? {};
  useEffect(() => { if (!preset && cfg.data) setPreset(Object.keys(cfg.data.presets)[0] ?? ""); }, [cfg.data, preset]);
  const body = useMemo(() => ({ preset, seed: Number(f.seed), total_steps: f.total_steps ? Number(f.total_steps) : null,
    timeout_s: Number(f.timeout_s), cpu: Number(f.cpu), memory_mb: Number(f.memory_mb) }), [preset, f]);
  useEffect(() => { setResolved(null); }, [body]);     // any edit invalidates the validated preview

  if (cfg.error instanceof ApiError && cfg.error.status === 404)
    return <><h1>Training</h1><NotConfigured what="experiment controller not configured" detail="Start the API with --experiments-root to enable runs." /></>;
  const list = runs.data?.runs ?? [];
  const run = list.find((r) => r.id === sel) ?? null;
  const validate = async () => { setErr(null); try { setResolved(await mutate<Resolved>("/api/runs/validate", body)); } catch (e) { setErr(e); } };
  const start = async () => {
    setErr(null);
    try { const r = await mutate<Run>("/api/runs", body); setSel(r.id); await runs.reload(); } catch (e) { setErr(e); }
  };

  return (
    <>
      <h1>Training</h1>
      {cfg.loading && <Loading what="controller" />}
      <ErrorBanner error={cfg.error} />
      <Freshness error={runs.error} updatedAt={runs.updatedAt} />
      {cfg.data?.backend === "fake" && <div className="banner" role="status"><Badge kind="warn">fake backend</Badge> deterministic test double: metrics and checkpoints are synthetic, no training happens.</div>}
      {cfg.data && (
        <section className="panel" aria-label="New run">
          <h2>New run</h2>
          <div className="row">
            <label>Preset <select aria-label="Preset" value={preset} onChange={(e) => setPreset(e.target.value)}>
              {Object.entries(presets).map(([k, v]) => <option key={k} value={k}>{k} ({v.role})</option>)}</select></label>
            {(["seed", "total_steps", "timeout_s", "cpu", "memory_mb"] as const).map((k) => (
              <label key={k}>{k} <input aria-label={k} value={f[k]} placeholder={k === "total_steps" ? "preset default" : ""} style={{ width: 90 }}
                onChange={(e) => setF({ ...f, [k]: e.target.value })} /></label>))}
          </div>
          <p className="muted">Ceilings: {Object.entries(cfg.data.ceilings).map(([k, v]) => `${k} ≤ ${v}`).join(" · ")}. Shared Hospital has no bounded dashboard preset.</p>
          <div className="row">
            <button onClick={validate}>Validate config</button>
            <button disabled={!resolved} onClick={start}>Start</button>
          </div>
          <ErrorBanner error={err} />
          {resolved && (
            <div role="status" aria-label="Resolved config">
              <p><Badge kind="ok">valid</Badge> {resolved.role} · {resolved.caps.total_steps} steps · timeout {resolved.caps.timeout_s}s · cpu {resolved.caps.cpu} · {resolved.caps.memory_mb} MB · 1 container</p>
              <p className="muted">Estimate: {resolved.estimate.basis === "measured"
                ? `~${resolved.estimate.seconds?.toFixed(0)}s at ${resolved.estimate.steps_per_sec?.toFixed(1)} steps/s (measured, ${resolved.estimate.note})`
                : `unavailable — ${resolved.estimate.note}`}</p>
              <pre>{JSON.stringify(resolved.train_config, null, 2)}</pre>
            </div>)}
        </section>)}
      <section className="panel" style={{ marginTop: 12 }} aria-label="Runs">
        <h2>Runs</h2>
        {runs.loading ? <Loading what="runs" /> : list.length === 0 ? <Empty>No runs yet. Validate a preset and start one.</Empty> : (
          <table><thead><tr><th>run</th><th>preset</th><th>state</th><th>steps</th><th>heartbeat</th><th /></tr></thead>
            <tbody>{list.map((r) => (
              <tr key={r.id}>
                <td className="mono">{r.id}</td><td>{r.preset}</td>
                <td><Badge kind={stateKind(r.state)}>{r.state}</Badge>{r.stale && <> <Badge kind="warn">stale</Badge></>}</td>
                <td>{r.latest_metric?.env_steps ?? "—"} / {r.resolved.caps.total_steps}</td>
                <td>{r.heartbeat_age_s === null ? "—" : `${r.heartbeat_age_s.toFixed(0)}s`}</td>
                <td><button onClick={() => setSel(r.id)} aria-label={`open ${r.id}`}>Open</button></td>
              </tr>))}</tbody></table>)}
      </section>
      <EvalPanel cps={allCps.data?.checkpoints ?? []} onDone={() => runs.reload()} />
      {run && <RunDetail run={run} onChange={() => { runs.reload(); allCps.reload(); }} />}
    </>
  );
}

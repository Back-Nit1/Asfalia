import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, mutate } from "../api/client";
import { visualHz } from "../api/config";
import { openStream } from "../api/sse";
import type { Envelope, Task, TasksResponse } from "../api/types";
import { Badge, ErrorBanner, NotConfigured, useJson } from "../components/ui";

type TaskData = { task: Task; selected: boolean; age: string };

function TaskNode({ data }: NodeProps<Node<TaskData>>) {
  const { task: t, selected, age } = data;
  return (
    <div className={`tasknode st-${t.status} ${selected ? "sel" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <b>{t.id}</b>
      <div className="muted">{t.role}</div>
      <div>{t.status.toUpperCase()} · {t.owner} · try {t.attempt}</div>
      {t.status === "running" && <div className="muted">heartbeat {age}</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
const nodeTypes = { task: TaskNode };

function layout(tasks: Task[]): Record<string, { x: number; y: number }> {
  const depth: Record<string, number> = {};
  const byId = Object.fromEntries(tasks.map((t) => [t.id, t]));
  const d = (id: string): number => (depth[id] ??= Math.max(-1, ...byId[id].dependencies.map(d)) + 1);
  const rows: Record<number, number> = {};
  const pos: Record<string, { x: number; y: number }> = {};
  for (const t of tasks) {
    const col = d(t.id);
    pos[t.id] = { x: col * 250, y: (rows[col] = (rows[col] ?? -1) + 1) * 110 };
  }
  return pos;
}

function ago(now: number, ts: number | null): string {
  if (ts === null) return "—";
  const s = Math.max(0, Math.round(now - ts));
  return s < 90 ? `${s}s ago` : `${Math.round(s / 60)}m ago`;
}

/** Server snapshot + SSE. Reload / reconnect re-reads server state; nothing is kept only in the browser. */
function useBoard() {
  const first = useJson<TasksResponse>("/api/tasks");
  const [resp, setResp] = useState<TasksResponse | null>(null);
  const [live, setLive] = useState<"connected" | "reconnecting">("reconnecting");
  const [log, setLog] = useState<Envelope[]>([]);
  const dirty = useRef(false);
  const reload = first.reload;
  useEffect(() => { if (first.data) setResp(first.data); }, [first.data]);
  useEffect(() => {
    const stop = openStream("/api/events/stream", (m) => {
      if (m.event === "resync") { dirty.current = true; return; }
      const env = m.data as Envelope;
      dirty.current = true;
      setLog((l) => [env, ...l].slice(0, 200));
    }, setLive);
    // coalesce bursts: refetch at most visualHz times per second
    const tick = setInterval(() => { if (dirty.current) { dirty.current = false; reload(); } }, 1000 / visualHz);
    return () => { stop(); clearInterval(tick); };
  }, [reload]);
  return { resp, live, log, error: first.error, reload };
}

export default function Build() {
  const { resp, live, log, error, reload } = useBoard();
  const [sel, setSel] = useState<string | null>(null);
  const tasks = resp?.tasks ?? [];
  const now = resp?.now ?? Date.now() / 1000;
  const selected = tasks.find((t) => t.id === sel) ?? null;

  const { nodes, edges } = useMemo(() => {
    const pos = layout(tasks);
    const nodes: Node<TaskData>[] = tasks.map((t) => ({ id: t.id, type: "task", position: pos[t.id], draggable: false,
      data: { task: t, selected: t.id === sel, age: ago(now, t.heartbeat) } }));
    const edges: Edge[] = tasks.flatMap((t) => t.dependencies.map((d) => ({ id: `${d}->${t.id}`, source: d, target: t.id })));
    return { nodes, edges };
  }, [tasks, sel, now]);

  return (
    <>
      <div className="row"><h1>Build</h1>
        <Badge kind={live === "connected" ? "ok" : "warn"}>stream {live}</Badge></div>
      <ErrorBanner error={error} />
      {resp && !resp.runtime.configured && <NotConfigured what="coding runtime not configured" detail={`${resp.runtime.detail}. Tasks marked human can still be claimed and submitted for review; no agent progress is simulated.`} />}
      {resp && tasks.length === 0 && <NotConfigured what="no task plan loaded" detail="Start the API with --plan configs/orchestrator.yaml." />}
      <div className="split">
        <div className="flow" aria-label="Task dependency graph">
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} nodesDraggable={false} nodesConnectable={false}
            onNodeClick={(_, n) => setSel(n.id)} fitView proOptions={{ hideAttribution: true }}>
            <Background /><Controls showInteractive={false} />
          </ReactFlow>
        </div>
        <TaskPanel task={selected} now={now} reload={reload} />
      </div>
      <div className="panel" style={{ marginTop: 12 }}>
        <h2>Tasks (keyboard list)</h2>
        <table><thead><tr><th>Task</th><th>Status</th><th>Owner</th><th>Attempt</th><th>Heartbeat</th><th>Integration</th><th>Outcome</th></tr></thead>
          <tbody>{tasks.map((t) => (
            <tr key={t.id}>
              <td><button onClick={() => setSel(t.id)} aria-pressed={t.id === sel}>{t.id}</button></td>
              <td>{t.status}</td><td>{t.owner}</td><td>{t.attempt}</td><td>{ago(now, t.heartbeat)}</td>
              <td>{t.status === "integrated" ? "integrated" : t.status === "passed" ? "awaiting integration" : "not yet"}</td>
              <td className="muted">{t.outcome}</td>
            </tr>))}</tbody></table>
      </div>
      <div className="panel" style={{ marginTop: 12 }}>
        <h2>Live development events</h2>
        {log.length === 0 ? <span className="muted">none received on this connection</span> :
          <pre>{log.slice(0, 30).map((e) => `${e.timestamp.slice(11, 19)} ${e.event_type} ${e.task_id ?? ""} ${JSON.stringify(e.payload).slice(0, 140)}`).join("\n")}</pre>}
      </div>
    </>
  );
}

function TaskPanel({ task: t, now, reload }: { task: Task | null; now: number; reload: () => void }) {
  const [diff, setDiff] = useState<string | null>(null);
  const [events, setEvents] = useState<Envelope[]>([]);
  const [msg, setMsg] = useState<string>("");
  const id = t?.id;
  useEffect(() => { setDiff(null); setMsg(""); setEvents([]); }, [id]);
  const act = useCallback(async (verb: string) => {
    if (!id) return;
    try { await mutate(`/api/tasks/${id}/${verb}`); setMsg(`${verb}: ok`); } catch (e) { setMsg(`${verb}: ${e instanceof ApiError ? e.message : String(e)}`); }
    reload();
  }, [id, reload]);
  if (!t) return <div className="panel"><h2>Task</h2><span className="muted">Select a task.</span></div>;
  const loadDiff = async () => setDiff((await api<{ available: boolean; reason?: string; diff: string }>(`/api/tasks/${t.id}/diff`)).diff || "(empty diff)");
  const loadEvents = async () => setEvents((await api<{ events: Envelope[] }>(`/api/tasks/${t.id}/events`)).events);
  const human = t.owner === "human";
  return (
    <div className="panel" aria-live="polite">
      <h2>{t.id}</h2>
      <div className="row"><Badge kind={t.status === "passed" || t.status === "integrated" ? "ok" : t.status === "failed" || t.status === "blocked" ? "bad" : "info"}>{t.status}</Badge><span className="muted">{t.role} · {t.owner}</span></div>
      <p>{t.description}</p>
      <div className="muted">attempt {t.attempt} · heartbeat {ago(now, t.heartbeat)} · lease {t.lease_expiry ? `expires in ${Math.round(t.lease_expiry - now)}s` : "—"}</div>
      {t.outcome && <p>Outcome: {t.outcome}</p>}
      {!!t.path_conflicts?.length && <div className="banner">Waiting on path ownership: {t.path_conflicts.join(", ")}</div>}
      <div className="muted">Owns: <span className="mono">{t.assigned_paths.join(", ")}</span></div>
      <h2 style={{ marginTop: 10 }}>Evidence</h2>
      <div>Changed files: {t.evidence.changed_files.length ? <span className="mono">{t.evidence.changed_files.join(", ")}</span> : <span className="muted">none recorded</span>}</div>
      {!!t.evidence.out_of_scope_files.length && <div className="banner bad">Outside assigned paths: {t.evidence.out_of_scope_files.join(", ")}</div>}
      {t.evidence.test_results.length ? <table><tbody>{t.evidence.test_results.map((r) => (
        <tr key={r.command}><td><Badge kind={r.exit_code ? "bad" : "ok"}>{r.exit_code ? `exit ${r.exit_code}` : "pass"}</Badge></td><td className="mono">{r.command}</td></tr>))}</tbody></table>
        : <div className="muted">No acceptance results yet (the orchestrator runs them; an agent's claim is not evidence).</div>}
      {t.evidence.agent_summary && <div className="muted">Agent summary: {t.evidence.agent_summary}</div>}
      <div className="row" style={{ marginTop: 10 }}>
        <button onClick={loadDiff}>View diff</button><button onClick={loadEvents}>Load events</button>
        <button onClick={() => act("cancel")} disabled={["integrated", "cancelled"].includes(t.status)}>Cancel</button>
        <button onClick={() => act("retry")} disabled={!["failed", "blocked"].includes(t.status)}>Retry</button>
        <button onClick={() => act("integrate")} disabled={t.status !== "passed"}>Integrate</button>
        {human && <><button onClick={() => act("claim")} disabled={t.status !== "ready"}>Claim</button>
          <button onClick={() => act("heartbeat")} disabled={t.status !== "running"}>Heartbeat</button>
          <button onClick={() => act("submit")} disabled={t.status !== "running"}>Submit for review</button></>}
      </div>
      {msg && <div role="status" className="muted">{msg}</div>}
      {diff !== null && <pre aria-label="diff">{diff}</pre>}
      {events.length > 0 && <pre aria-label="task events">{events.map((e) => `${e.event_type} ${JSON.stringify(e.payload).slice(0, 160)}`).join("\n")}</pre>}
    </div>
  );
}

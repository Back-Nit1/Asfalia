import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { api, getToken, setToken, worldPath } from "../api/client";
import { metricsHz, visualHz } from "../api/config";
import type { EpisodeInfo, Frame, NodeView, Perspective, Replay } from "../api/types";
import { Badge, Card, ErrorBanner, NotConfigured, useJson } from "../components/ui";

const ZONES = ["dmz", "clinical", "core", "business", "device"];
const GLYPH: Record<string, string> = { vpn: "VPN", workstation: "WS", idp: "IDP", ehr: "EHR", pacs: "IMG", billing: "BIL", k8s_service: "K8S", device: "DEV" };
type AssetData = { n: NodeView; selected: boolean };

function AssetNode({ data }: NodeProps<Node<AssetData>>) {
  const { n, selected } = data;
  return (
    <div className={`tasknode st-${n.status} ${selected ? "sel" : ""}`} style={{ width: 150 }}>
      <Handle type="target" position={Position.Left} />
      <b>{GLYPH[n.asset_type ?? ""] ?? "?"} · {n.id}</b>
      <div>{n.status.toUpperCase()}</div>
      {n.tags.length > 0 && <div className="muted">{n.tags.join(", ").replaceAll("_", " ")}</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
const nodeTypes = { asset: AssetNode };

/** Stable layout: zones are fixed columns; nodes keep their order inside a zone, so state changes never move them. */
function buildGraph(nodes: NodeView[], edges: Replay["edges"], selected: string | null) {
  const zones = [...new Set(nodes.map((n) => n.zone ?? "other"))].sort((a, b) => (ZONES.indexOf(a) + 99) % 99 - (ZONES.indexOf(b) + 99) % 99 || a.localeCompare(b));
  const out: Node[] = [];
  zones.forEach((z, zi) => {
    const members = nodes.filter((n) => (n.zone ?? "other") === z).sort((a, b) => a.id.localeCompare(b.id));
    out.push({ id: `zone:${z}`, type: "group", position: { x: zi * 200, y: 0 }, data: { label: z }, draggable: false, selectable: false,
      style: { width: 180, height: 40 + members.length * 86, background: "transparent", border: "1px dashed var(--line)", color: "var(--muted)", fontSize: 11 } });
    members.forEach((n, i) => out.push({ id: n.id, type: "asset", parentId: `zone:${z}`, extent: "parent", position: { x: 15, y: 34 + i * 86 },
      draggable: false, data: { n, selected: n.id === selected } as AssetData }));
  });
  const ids = new Set(nodes.map((n) => n.id));
  const es: Edge[] = (edges ?? []).filter((e) => ids.has(e.src) && ids.has(e.dst)).map((e, i) => ({
    id: `${e.src}-${e.dst}-${e.kind}-${i}`, source: e.src, target: e.dst, animated: false,
    style: e.kind === "trust" ? { strokeDasharray: "4 3", stroke: "#9aa6b6" } : { stroke: "#5b6a7e" }, label: e.kind === "trust" ? "trust" : undefined }));
  return { nodes: out, edges: es };
}

export default function World() {
  const { episodeId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const persp = (["attacker", "defender", "evaluator"].includes(params.get("p") ?? "") ? params.get("p") : "defender") as Perspective;
  const [token, setTok] = useState(getToken("evaluator"));
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [sel, setSel] = useState<string | null>(null);
  const eps = useJson<{ episodes: EpisodeInfo[]; live: { configured: boolean; message?: string } }>("/api/world/episodes");

  // Each perspective is a different endpoint and DTO; nothing is hidden client-side.
  useEffect(() => {
    let dead = false;
    setReplay(null); setError(null); setI(0); setPlaying(false);   // reset before max shrinks, else the range DOM clamps behind React's back
    api<Replay>(worldPath(persp, episodeId, "replay")).then((r) => { if (!dead) setReplay(r); }, (e) => { if (!dead) setError(e); });
    return () => { dead = true; };
  }, [persp, episodeId, token]);

  const frames = replay?.frames ?? [];
  // Playback steps at most visualHz times a second; speed multiplies ticks per step.
  useEffect(() => {
    if (!playing || !frames.length) return;
    const t = setInterval(() => setI((x) => { if (x >= frames.length - 1) { setPlaying(false); return x; } return Math.min(frames.length - 1, x + speed); }), 1000 / visualHz);
    return () => clearInterval(t);
  }, [playing, speed, frames.length]);

  const frame: Frame | undefined = frames[Math.min(i, frames.length - 1)];
  // metric cards refresh at ~metricsHz while playing
  const [cards, setCards] = useState<Frame | undefined>(undefined);
  const latest = useRef(frame); latest.current = frame;
  useEffect(() => { const t = setInterval(() => setCards(latest.current), 1000 / metricsHz); setCards(latest.current); return () => clearInterval(t); }, [replay]);

  const feed = useMemo(() => frames.slice(0, i + 1).flatMap((f) => f.feed), [frames, i]);
  const graph = useMemo(() => buildGraph(frame?.nodes ?? [], replay?.edges, sel), [frame, replay, sel]);
  const selected = frame?.nodes.find((n) => n.id === sel) ?? null;
  const info = eps.data?.episodes.find((e) => e.id === episodeId);
  const switchTo = (p: Perspective) => setParams({ p });
  const lastTick = frames.length ? frames[frames.length - 1].tick : 0;

  return (
    <>
      <h1>World · <span className="mono">{episodeId}</span></h1>
      <div className="panel row" role="group" aria-label="Episode status">
        <Badge kind="info">REPLAY</Badge>
        {replay && <Badge kind="warn">{replay.label}</Badge>}
        <span className="muted" title="No live producer connected">LIVE: not configured</span>
        <span>tick <b>{frame?.tick ?? "—"}</b> / {lastTick}</span>
        <span className="muted">scenario {replay?.scenario ?? "—"}</span>
        <span className="muted">checkpoints: none (scripted policies)</span>
        <span className="muted">last event age: n/a (recorded)</span>
        <label>perspective{" "}
          <select aria-label="Perspective" value={persp} onChange={(e) => switchTo(e.target.value as Perspective)}>
            <option value="defender">defender</option><option value="attacker">attacker</option><option value="evaluator">evaluator (privileged)</option>
          </select></label>
      </div>
      {persp === "evaluator" && (
        <div className="banner priv" role="alert"><Badge kind="bad">PRIVILEGED</Badge> Ground-truth evaluator view — never shown to a policy.
          <div className="row" style={{ marginTop: 6 }}><label>evaluator token <input type="password" value={token} onChange={(e) => { setTok(e.target.value); setToken("evaluator", e.target.value); }} /></label></div></div>
      )}
      <ErrorBanner error={error} />
      {replay && !replay.topology_exported && <NotConfigured what="topology not exported" detail="This episode file has no topology block, so the graph cannot be drawn." />}
      {replay?.label === "FIXTURE" && <div className="banner"><Badge kind="warn">FIXTURE</Badge> Recorded scripted tiny episode shipped with the dashboard — not live data. {info ? `${info.ticks} ticks.` : ""}</div>}
      <div className="grid cards" style={{ marginBottom: 12 }} aria-label="Metrics">
        <Metrics persp={persp} replay={replay} f={cards} />
      </div>
      <div className="split">
        <div className="flow" aria-label="Hospital topology grouped by zone">
          {persp === "attacker" && (frame?.nodes.length ?? 0) === 0 ? <p className="muted" style={{ padding: 12 }}>Nothing discovered yet at this tick.</p> :
            <ReactFlow key={`${persp}-${graph.nodes.length}`} nodes={graph.nodes} edges={graph.edges} nodeTypes={nodeTypes} nodesDraggable={false} nodesConnectable={false}
              onNodeClick={(_, n) => { if (!n.id.startsWith("zone:")) setSel(n.id); }} fitView proOptions={{ hideAttribution: true }}>
              <Background /><Controls showInteractive={false} />
            </ReactFlow>}
        </div>
        <div className="panel"><h2>Selected asset</h2>
          {selected ? <>
            <b>{selected.id}</b> <span className="muted">{selected.asset_type} · {selected.zone}</span>
            <div>State: <Badge>{selected.status}</Badge></div>
            <div>Tags: {selected.tags.length ? selected.tags.join(", ") : "none"}</div>
            <h2 style={{ marginTop: 10 }}>Actions on this asset</h2>
            <ul>{feed.filter((e) => e.target === selected.id).slice(-6).map((e) => <li key={e.idx}>t{e.tick} {e.actor} {e.action ?? e.type}: {e.outcome}</li>)}</ul>
          </> : <span className="muted">Select a node. Workflow dependencies are not exported yet.</span>}
        </div>
      </div>
      <div className="panel row" style={{ marginTop: 12 }} role="group" aria-label="Replay controls">
        <button onClick={() => setPlaying((p) => !p)} disabled={!frames.length}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => { setPlaying(false); setI((x) => Math.max(0, x - 1)); }} disabled={!frames.length}>Step back</button>
        <button onClick={() => { setPlaying(false); setI((x) => Math.min(frames.length - 1, x + 1)); }} disabled={!frames.length}>Step</button>
        <label>speed <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>{[1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}</select></label>
        <label style={{ flex: 1 }}>tick <input key={`${persp}-${frames.length}`} type="range" min={0} max={Math.max(0, frames.length - 1)} value={i} aria-label="Scrub" style={{ width: "70%" }}
          onChange={(e) => { setPlaying(false); setI(Number(e.target.value)); }} /></label>
      </div>
      <div className="panel feed" style={{ marginTop: 12 }}>
        <h2>Action feed ({persp} visible)</h2>
        <table><thead><tr><th>tick</th><th>actor</th><th>action</th><th>target</th><th>observed outcome</th><th>duration</th></tr></thead>
          <tbody>{[...feed].reverse().slice(0, 60).map((e) => (
            <tr key={e.idx}><td>{e.tick}</td><td>{e.actor}</td><td>{e.action ?? e.type}</td><td>{e.target ?? "—"}</td><td>{e.outcome}</td><td>{e.duration ?? "—"}</td></tr>))}</tbody></table>
        {feed.length === 0 && <span className="muted">No events visible to this perspective yet.</span>}
      </div>
    </>
  );
}

const f2 = (x: number | undefined) => (x === undefined ? "—" : x.toFixed(2));

function Metrics({ persp, replay, f }: { persp: Perspective; replay: Replay | null; f: Frame | undefined }) {
  if (!replay || !f) return <Card title="Metrics" value="—" sub="loading" />;
  const own = persp === "attacker" ? "attacker" : "defender";
  const c = f.cumulative.components[own] ?? {};
  const fm = replay.final_metrics;
  return (
    <>
      <Card title="Sensitive records exfiltrated" value={fm ? String(fm.records_exfiltrated) : "withheld"}
        sub={fm ? "episode total (evaluator only; not per tick)" : persp === "defender" ? "not observable to defender; see exfil penalty" : "not visible to this perspective"} />
      <Card title={fm || persp !== "defender" ? "Critical workflow success" : "Critical workflow penalty"}
        value={fm ? f2(Number(fm.critical_success)) : persp === "defender" ? f2(c.critical) : "—"}
        sub={fm ? "episode total (evaluator only)" : persp === "defender" ? "cumulative reward term; 0 = no critical workflow failed" : "not visible to this perspective"} />
      {fm && <Card title="Containment" value={fm.contained ? `tick ${fm.containment_tick}` : "not contained"} sub="shown beside workflow success: over-securing must stay visibly costly" />}
      <Card title="Simulated operational spend" value={persp === "defender" ? f2(c.spend) : fm ? String(fm.spend) : "—"}
        sub={persp === "defender" ? "cumulative spend reward term (simulated, not cloud spend)" : fm ? "episode total (simulated)" : "not visible to this perspective"} />
      <Card title="Remaining action budget" value="not exported" sub="per-tick budget is not in the episode export" />
      <Card title={`${own} cumulative reward`} value={f2(f.cumulative.reward[own])} sub={Object.entries(c).map(([k, v]) => `${k} ${f2(v)}`).join(" · ")} />
    </>
  );
}

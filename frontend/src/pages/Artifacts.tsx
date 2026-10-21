import { metricsHz } from "../api/config";
import type { Checkpoint, Run } from "../api/types";
import { Badge, Empty, ErrorBanner, Freshness, Loading, useJson } from "../components/ui";

interface Art { id: string; kind: string; task_id: string | null; path: string; created_at: string }
interface RunArtifacts { attempts: { attempt_id: string; cleaned: boolean; partial: boolean; problems: string[]; files: { name: string; size: number }[] }[] }

function RunFiles({ run }: { run: Run }) {
  const { data, error } = useJson<RunArtifacts>(`/api/runs/${run.id}/artifacts`, 3000);
  return (
    <li>
      <b>{run.id}</b> <span className="muted">({run.state})</span>
      <ErrorBanner error={error} />
      {data?.attempts.map((a) => (
        <div key={a.attempt_id} className="mono">
          {a.attempt_id}: {a.cleaned ? <Badge kind="info">cleaned up</Badge> : a.partial ? <Badge kind="warn">partial</Badge> : <Badge kind="ok">manifest verified</Badge>}{" "}
          {a.files.length} files {a.problems.length > 0 && <span>— {a.problems.join("; ")}</span>}
        </div>))}
    </li>
  );
}

export default function Artifacts() {
  const cps = useJson<{ checkpoints: Checkpoint[] }>("/api/checkpoints", 1000 / metricsHz);
  const runs = useJson<{ runs: Run[] }>("/api/runs", 3000);
  const dev = useJson<{ artifacts: Art[] }>("/api/artifacts");
  const list = cps.data?.checkpoints ?? [];
  return (
    <>
      <h1>Artifacts</h1>
      <ErrorBanner error={cps.error ?? dev.error} />
      <Freshness error={cps.error} updatedAt={cps.updatedAt} />
      <section className="panel" aria-label="Checkpoint lineage">
        <h2>Checkpoints (provenance and lineage)</h2>
        {cps.loading ? <Loading what="checkpoints" /> : list.length === 0 ? <Empty>No checkpoints recorded.</Empty> : (
          <table><thead><tr><th>id</th><th>parent</th><th>role</th><th>run/attempt</th><th>steps</th><th>policy v</th><th>sha256</th><th>size</th><th>state</th></tr></thead>
            <tbody>{list.map((c) => (
              <tr key={c.id}>
                <td className="mono">{c.id}</td><td className="mono">{c.parent_id ?? "root"}</td><td>{c.role}</td><td>{c.run_id}/{c.attempt_id}</td>
                <td>{c.env_steps}</td><td>{c.policy_version}</td><td className="mono" title={c.sha256}>{c.sha256.slice(0, 16)}…</td><td>{c.size} B</td>
                <td>{c.available ? <Badge kind="ok">available</Badge> : <Badge kind="info">cleaned up</Badge>}</td>
              </tr>))}</tbody></table>)}
      </section>
      <section className="panel" style={{ marginTop: 12 }} aria-label="Run artifacts">
        <h2>Run artifacts</h2>
        {(runs.data?.runs ?? []).length === 0 ? <Empty>No runs.</Empty> : <ul>{runs.data!.runs.map((r) => <RunFiles key={r.id} run={r} />)}</ul>}
      </section>
      <section className="panel" style={{ marginTop: 12 }} aria-label="Task artifacts">
        <h2>Development task artifacts</h2>
        {dev.data && (dev.data.artifacts.length === 0 ? <Empty>No task artifacts recorded.</Empty> : (
          <table><thead><tr><th>id</th><th>kind</th><th>task</th><th>path</th><th>created</th></tr></thead>
            <tbody>{dev.data.artifacts.map((a) => <tr key={a.id}><td>{a.id}</td><td>{a.kind}</td><td>{a.task_id}</td><td className="mono">{a.path}</td><td>{a.created_at}</td></tr>)}</tbody></table>))}
      </section>
    </>
  );
}

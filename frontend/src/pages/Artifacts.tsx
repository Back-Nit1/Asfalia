import { ErrorBanner, NotConfigured, useJson } from "../components/ui";

interface Art { id: string; kind: string; task_id: string | null; path: string; created_at: string }

export default function Artifacts() {
  const { data, error } = useJson<{ artifacts: Art[] }>("/api/artifacts");
  return (
    <>
      <h1>Artifacts</h1>
      <ErrorBanner error={error} />
      <NotConfigured what="no checkpoints or trajectories" detail="Only task diffs recorded by the orchestrator are listed below." />
      {data && (data.artifacts.length === 0 ? <p className="muted">No artifacts recorded.</p> : (
        <table><thead><tr><th>id</th><th>kind</th><th>task</th><th>path</th><th>created</th></tr></thead>
          <tbody>{data.artifacts.map((a) => <tr key={a.id}><td>{a.id}</td><td>{a.kind}</td><td>{a.task_id}</td><td className="mono">{a.path}</td><td>{a.created_at}</td></tr>)}</tbody></table>
      ))}
    </>
  );
}

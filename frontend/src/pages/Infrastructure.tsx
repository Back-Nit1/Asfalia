import { useState } from "react";
import { mutate } from "../api/client";
import { metricsHz } from "../api/config";
import { Badge, Empty, ErrorBanner, Freshness, Loading, NotConfigured, useJson } from "../components/ui";

interface Job { run_id: string; attempt_id: string; job_id: string | null; job_state: string | null; run_state: string; heartbeat_age_s: number | null;
  stale: boolean; backend_error: string | null; error: string | null; owner: string; cleaned: boolean }
interface Infra {
  configured: boolean; backend?: string; jobs: Job[]; failures: Job[]; orphans: { run_id: string; attempt_id: string }[]; evaluator: string | null;
  policy_versions: { checkpoint_id: string; run_id: string; policy_version: number | null; role: string }[];
  estimate: { basis: string; steps_per_sec: number | null; age_s: number | null }; message?: string;
}
interface Plan { dry_run: boolean; plan: { run_id: string; attempt_id: string; action: string; reason: string | null }[]; orphans_reported_not_deleted: unknown[] }

export default function Infrastructure() {
  const { data, error, updatedAt, loading, reload } = useJson<Infra>("/api/infrastructure", 1000 / metricsHz);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const clean = async (dry: boolean) => {
    setErr(null);
    try { const p = await mutate<Plan>(`/api/infrastructure/cleanup?dry_run=${dry}`); setPlan(p); if (!dry) reload(); } catch (e) { setErr(e); }
  };
  return (
    <>
      <h1>Infrastructure</h1>
      <ErrorBanner error={err ?? error} />
      <Freshness error={error} updatedAt={updatedAt} />
      {loading && <Loading what="infrastructure" />}
      {data && !data.configured && <NotConfigured what="no infrastructure backend" detail={data.message} />}
      {data?.configured && (
        <>
          <p>Backend <Badge kind={data.backend === "fake" ? "warn" : "info"}>{data.backend}</Badge> · evaluator {data.evaluator ?? "not configured"} · step-rate estimate{" "}
            {data.estimate.basis === "measured" ? `${data.estimate.steps_per_sec?.toFixed(1)} steps/s, measured ${data.estimate.age_s}s ago` : "unavailable (no measurement yet)"}</p>
          <section className="panel" aria-label="Jobs">
            <h2>Jobs</h2>
            {data.jobs.length === 0 ? <Empty>No jobs have been submitted.</Empty> : (
              <table><thead><tr><th>run</th><th>attempt</th><th>job id</th><th>backend state</th><th>run state</th><th>heartbeat age</th><th>problems</th></tr></thead>
                <tbody>{data.jobs.map((j) => (
                  <tr key={`${j.run_id}/${j.attempt_id}`}>
                    <td>{j.run_id}</td><td>{j.attempt_id}</td><td className="mono">{j.job_id ?? "—"}</td><td>{j.job_state ?? "—"}</td><td>{j.run_state}</td>
                    <td>{j.heartbeat_age_s === null ? "—" : `${j.heartbeat_age_s.toFixed(0)}s`} {j.stale && <Badge kind="warn">stale</Badge>}</td>
                    <td>{j.cleaned ? <Badge kind="info">cleaned up</Badge> : j.error || j.backend_error || "—"}</td>
                  </tr>))}</tbody></table>)}
          </section>
          <section className="panel" style={{ marginTop: 12 }} aria-label="Policy versions">
            <h2>Policy versions</h2>
            {data.policy_versions.length === 0 ? <Empty>No checkpoints yet.</Empty> : (
              <p className="mono">{data.policy_versions.map((p) => `${p.role} v${p.policy_version} (${p.run_id})`).join(" · ")}</p>)}
          </section>
          <section className="panel" style={{ marginTop: 12 }} aria-label="Cleanup">
            <h2>Owned-resource cleanup</h2>
            <p className="muted">Only attempts this controller created, whose run is finished and whose manifest verifies, can be deleted. Orphans are reported, never deleted.</p>
            <div className="row">
              <button onClick={() => clean(true)}>Dry run</button>
              <button disabled={!plan?.plan.some((p) => p.action === "delete")} onClick={() => clean(false)}>Delete planned</button>
            </div>
            {plan && (
              <div role="status" aria-label="Cleanup plan">
                <p>{plan.dry_run ? "Dry run — nothing deleted." : "Deleted."}</p>
                {plan.plan.length === 0 ? <Empty>Nothing eligible.</Empty> : (
                  <ul>{plan.plan.map((p) => <li key={`${p.run_id}/${p.attempt_id}`}>{p.run_id}/{p.attempt_id}: <b>{p.action}</b> {p.reason && `(${p.reason})`}</li>)}</ul>)}
              </div>)}
            <p>Orphans: {data.orphans.length === 0 ? "none" : data.orphans.map((o) => `${o.run_id}/${o.attempt_id}`).join(", ")}</p>
          </section>
        </>)}
    </>
  );
}

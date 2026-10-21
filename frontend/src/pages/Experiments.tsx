import { Link } from "react-router-dom";
import { metricsHz } from "../api/config";
import type { Evaluation, Pair } from "../api/types";
import { Badge, Empty, ErrorBanner, Freshness, Loading, NotConfigured, useJson } from "../components/ui";

const fmt = (p: Pair) => `${p.mean.toFixed(3)} [${p.ci_low.toFixed(3)}, ${p.ci_high.toFixed(3)}]`;

/** Everything here is read from stored evaluation results; a cell with no stored pair stays "—". */
export default function Experiments() {
  const res = useJson<{ results: Evaluation[]; evaluator: string | null }>("/api/experiments/results", 1000 / metricsHz);
  const all = useJson<{ evaluations: Evaluation[] }>("/api/evaluations", 1000 / metricsHz);
  if (res.error && (res.error as { status?: number }).status === 404)
    return <><h1>Experiments</h1><NotConfigured what="no experiment results" detail="The experiment controller is not configured." /></>;
  const done = res.data?.results ?? [];
  const bad = (all.data?.evaluations ?? []).filter((e) => e.state === "failed");
  const pending = (all.data?.evaluations ?? []).filter((e) => e.state === "queued" || e.state === "running");
  const pairs = done.flatMap((e) => (e.result?.pairs ?? []).map((p) => ({ e, p })));
  const atks = [...new Set(pairs.map(({ p }) => p.attacker))], dfns = [...new Set(pairs.map(({ p }) => p.defender))];
  const suites = [...new Set(done.map((e) => e.suite))];
  return (
    <>
      <h1>Experiments</h1>
      <ErrorBanner error={res.error ?? all.error} />
      <Freshness error={res.error} updatedAt={res.updatedAt} />
      {res.data?.evaluator === "fake" && <div className="banner" role="status"><Badge kind="warn">fake evaluator</Badge> scores below come from a deterministic test double, not from policies.</div>}
      {res.loading && <Loading what="results" />}
      {pending.length > 0 && <p role="status">{pending.length} evaluation(s) pending: {pending.map((e) => e.id).join(", ")}</p>}
      {res.data && done.length === 0 && <Empty>No stored results yet. Start an evaluation from Training; nothing is shown until it finishes.</Empty>}
      {bad.length > 0 && (
        <section className="panel" aria-label="Compatibility and evaluation errors" style={{ marginBottom: 12 }}>
          <h2>Failed evaluations</h2>
          <ul>{bad.map((e) => <li key={e.id}><Badge kind="bad">{e.id}</Badge> {e.error}</li>)}</ul>
        </section>)}
      {suites.map((s) => (
        <section key={s} className="panel" style={{ marginBottom: 12 }} aria-label={`Suite ${s}`}>
          <h2>{s} (mean [95% CI], frozen policies)</h2>
          <table><thead><tr><th>eval</th><th>attacker</th><th>defender</th><th>metric</th><th>n</th><th>result</th><th>replay</th></tr></thead>
            <tbody>{done.filter((e) => e.suite === s).flatMap((e) => (e.result?.pairs ?? []).map((p, i) => (
              <tr key={`${e.id}-${i}`}>
                <td>{e.id}</td><td className="mono">{p.attacker}</td><td className="mono">{p.defender}</td><td>{p.metric}</td><td>{p.n}</td><td>{fmt(p)}</td>
                <td>{e.result?.episode_ids.map((id) => <Link key={id} to={`/world/${id}`}>replay</Link>)}</td>
              </tr>)))}</tbody></table>
        </section>))}
      {pairs.length > 0 && (
        <section className="panel" aria-label="Cross-play matrix">
          <h2>Cross-play matrix (attacker rows × defender columns)</h2>
          <table><thead><tr><th /> {dfns.map((d) => <th key={d} className="mono">{d}</th>)}</tr></thead>
            <tbody>{atks.map((a) => (
              <tr key={a}><th className="mono">{a}</th>{dfns.map((d) => {
                const hit = pairs.find(({ p }) => p.attacker === a && p.defender === d);
                return <td key={d}>{hit ? fmt(hit.p) : "—"}</td>;
              })}</tr>))}</tbody></table>
        </section>)}
    </>
  );
}

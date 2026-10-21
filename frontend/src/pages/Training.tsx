import { ErrorBanner, NotConfigured, useJson } from "../components/ui";

export default function Training() {
  const runs = useJson<{ configured: boolean; message?: string }>("/api/runs");
  const cps = useJson<{ configured: boolean; message?: string }>("/api/checkpoints");
  return (
    <>
      <h1>Training</h1>
      <ErrorBanner error={runs.error ?? cps.error} />
      {["Attacker School", "Defender School", "Shared Hospital"].map((t) => (
        <section key={t} className="panel" style={{ marginBottom: 12 }} aria-label={t}>
          <h2>{t}</h2>
          {runs.data && !runs.data.configured && <NotConfigured what="no training runs" detail={runs.data.message} />}
          {cps.data && !cps.data.configured && <NotConfigured what="no checkpoints" detail={cps.data.message} />}
          <div className="row"><button disabled>Validate config</button><button disabled>Start</button><button disabled>Checkpoint and stop</button></div>
        </section>
      ))}
    </>
  );
}

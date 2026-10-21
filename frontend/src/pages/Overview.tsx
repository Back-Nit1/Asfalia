import { Link } from "react-router-dom";
import { metricsHz } from "../api/config";
import type { Runtime } from "../api/types";
import { Card, ErrorBanner, NotConfigured, useJson } from "../components/ui";

interface Overview {
  tasks: Record<string, number>; task_total: number; runtime: Runtime; stale_running: string[];
  runs: { configured: boolean; count: number; by_state?: Record<string, number>; stale?: string[] }; infrastructure: { configured: boolean; backend?: string };
  latest_evaluation: { id: string; suite: string } | null; step_rate: number | null; spend_estimate: null;
}

export default function Overview() {
  const { data, error } = useJson<Overview>("/api/overview", 1000 / metricsHz);
  return (
    <>
      <h1>Overview</h1>
      <ErrorBanner error={error} />
      {data && (
        <>
          {!data.runtime.configured && <NotConfigured what="coding runtime not configured" detail={data.runtime.detail} />}
          <div className="grid cards">
            <Card title="Tasks" value={`${data.tasks.integrated ?? 0} / ${data.task_total}`} sub="integrated / total" />
            <Card title="Running" value={data.tasks.running ?? 0} sub={data.stale_running.length ? `stale lease: ${data.stale_running.join(", ")}` : "no stale workers"} />
            <Card title="Blocked / failed" value={`${data.tasks.blocked ?? 0} / ${data.tasks.failed ?? 0}`} />
            <Card title="Runs" value={data.runs.configured ? data.runs.count : "—"} sub={data.runs.configured ? (data.runs.stale?.length ? `stale: ${data.runs.stale.join(", ")}` : Object.entries(data.runs.by_state ?? {}).map(([k, v]) => `${v} ${k}`).join(", ")) : "experiment controller not configured"} />
            <Card title="Step rate" value={data.step_rate ? data.step_rate.toFixed(1) : "—"} sub={data.step_rate ? "steps/s, measured" : "no measurement"} />
            <Card title="Spend estimate" value="—" sub="no cost data" />
            <Card title="Latest evaluation" value={data.latest_evaluation?.id ?? "—"} sub={data.latest_evaluation?.suite ?? "none recorded"} />
          </div>
          <p><Link to="/build">Open build board</Link> · <Link to="/world/fixture_tiny_episode">Open fixture episode</Link></p>
        </>
      )}
    </>
  );
}

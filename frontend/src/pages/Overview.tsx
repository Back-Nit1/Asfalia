import { Link } from "react-router-dom";
import { metricsHz } from "../api/config";
import type { Runtime } from "../api/types";
import { Card, ErrorBanner, NotConfigured, useJson } from "../components/ui";

interface Overview {
  tasks: Record<string, number>; task_total: number; runtime: Runtime; stale_running: string[];
  runs: { configured: boolean; count: number }; infrastructure: { configured: boolean };
  latest_evaluation: null; step_rate: null; spend_estimate: null;
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
            <Card title="Runs" value={data.runs.configured ? data.runs.count : "—"} sub={data.runs.configured ? "" : "experiment controller not configured"} />
            <Card title="Step rate" value="—" sub="no training producer" />
            <Card title="Spend estimate" value="—" sub="no cost data" />
            <Card title="Latest evaluation" value="—" sub="none recorded" />
          </div>
          <p><Link to="/build">Open build board</Link> · <Link to="/world/fixture_tiny_episode">Open fixture episode</Link></p>
        </>
      )}
    </>
  );
}

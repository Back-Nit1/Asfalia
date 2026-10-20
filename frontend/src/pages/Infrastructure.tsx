import { ErrorBanner, NotConfigured, useJson } from "../components/ui";

export default function Infrastructure() {
  const { data, error } = useJson<{ configured: boolean; message?: string; workers: unknown[]; sandboxes: unknown[] }>("/api/infrastructure");
  return (
    <>
      <h1>Infrastructure</h1>
      <ErrorBanner error={error} />
      {data && !data.configured && <NotConfigured what="no infrastructure backend" detail={data.message} />}
      {data?.configured && <p>{data.workers.length} workers · {data.sandboxes.length} sandboxes</p>}
    </>
  );
}

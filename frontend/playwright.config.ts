import { defineConfig } from "@playwright/test";

// Serves the built dashboard from the real FastAPI process (fixture episode, no coding runtime, throwaway DB) with the
// experiment controller on the deterministic FAKE backend (no training, no cloud) polled every 0.4 s.
export default defineConfig({
  testDir: "e2e",
  workers: 1,
  use: { baseURL: "http://127.0.0.1:8765" },
  webServer: {
    command: "rm -rf /tmp/hospital-rl-e2e.sqlite* /tmp/hospital-rl-e2e-exp && cd .. && PYTHONPATH=src HOSPITAL_RL_EVALUATOR_TOKEN=e2e-evaluator-token .venv/bin/python -m hospital_rl.api --port 8765 --db /tmp/hospital-rl-e2e.sqlite --plan configs/orchestrator.yaml --no-agent --run-scheduler --experiments-root /tmp/hospital-rl-e2e-exp --fake-backend --tick-seconds 0.4 --static-dir frontend/dist",
    url: "http://127.0.0.1:8765/api/health",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});

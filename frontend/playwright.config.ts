import { defineConfig } from "@playwright/test";

// Serves the built dashboard from the real FastAPI process (fixture episode, no coding runtime, throwaway DB).
export default defineConfig({
  testDir: "e2e",
  use: { baseURL: "http://127.0.0.1:8765" },
  webServer: {
    command: "rm -f /tmp/hospital-rl-e2e.sqlite* && cd .. && PYTHONPATH=src HOSPITAL_RL_EVALUATOR_TOKEN=e2e-evaluator-token .venv/bin/python -m hospital_rl.api --port 8765 --db /tmp/hospital-rl-e2e.sqlite --plan configs/orchestrator.yaml --no-agent --run-scheduler --static-dir frontend/dist",
    url: "http://127.0.0.1:8765/api/health",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});

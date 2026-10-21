import { expect, test, type Page } from "@playwright/test";

async function startRun(page: Page, preset: string, steps: string) {
  await page.goto("/training");
  await expect(page.getByText("fake backend")).toBeVisible();
  await page.getByLabel("Preset").selectOption(preset);
  await page.getByLabel("total_steps").fill(steps);
  await expect(page.getByRole("button", { name: "Start" })).toBeDisabled();      // start requires a validation of exactly this config
  await page.getByRole("button", { name: "Validate config" }).click();
  await expect(page.getByLabel("Resolved config")).toContainText(`${steps} steps`);
  await expect(page.getByLabel("Resolved config")).toContainText("Estimate:");
  await page.getByRole("button", { name: "Start" }).click();
}

test("validate -> bounded run -> metrics -> checkpoint-and-stop -> resume -> frozen eval -> replay/comparison", async ({ page }) => {
  // honest empty states before anything exists
  await page.goto("/experiments");
  await expect(page.getByText("No stored results yet")).toBeVisible();
  await page.goto("/artifacts");
  await expect(page.getByText("No checkpoints recorded.")).toBeVisible();

  // invalid config is refused by the server with every problem, and nothing starts
  await page.goto("/training");
  await page.getByLabel("cpu").fill("99");
  await page.getByRole("button", { name: "Validate config" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "cpu=99.0" })).toBeVisible();
  await expect(page.getByText("No runs yet")).toBeVisible();

  // attacker run: real (file-ingested) metrics appear
  await startRun(page, "attacker-school-smoke", "2048");
  await page.getByRole("button", { name: "open run-0001" }).click();
  await expect(page.getByRole("img", { name: /mean return over env steps/ })).toBeVisible();
  await expect(page.getByLabel("live feed")).toContainText("training_metric");
  await expect(page.getByLabel("Runs").getByText("running", { exact: true })).toBeVisible();

  // checkpoint-and-stop -> stopped, with a checkpoint to resume from
  await page.getByRole("button", { name: "Checkpoint and stop" }).click();
  await expect(page.getByLabel("Runs").getByText("stopped", { exact: true })).toBeVisible();
  await expect(page.getByLabel("checkpoints", { exact: true })).toBeVisible();
  const stoppedAt = await page.request.get("/api/runs/run-0001").then((r) => r.json());
  const before = stoppedAt.latest_metric.env_steps as number;

  // resume -> new attempt, steps continue from the checkpoint (not restarted)
  await page.getByRole("button", { name: "Resume" }).click();
  await expect.poll(async () => (await page.request.get("/api/runs/run-0001").then((r) => r.json())).latest_metric.env_steps, { timeout: 15_000 }).toBeGreaterThan(before);
  const resumed = await page.request.get("/api/runs/run-0001").then((r) => r.json());
  expect(resumed.attempts).toHaveLength(2);
  expect(resumed.attempts[1].resume_from).toBe("a1");
  await page.getByRole("button", { name: "Force cancel" }).click();
  await expect(page.getByLabel("Runs").getByText("cancelled", { exact: true })).toBeVisible();

  // debug fork is allowed from a checkpoint and is labelled DEBUG, never a benchmark result
  await page.getByRole("button", { name: "Debug fork" }).first().click();
  await expect.poll(async () => (await page.request.get("/api/debug-forks").then((r) => r.json())).forks[0]?.state, { timeout: 10_000 }).toBe("done");

  // defender run to completion
  await startRun(page, "defender-school-smoke", "256");
  await expect(page.getByLabel("Runs").getByText("completed", { exact: true })).toBeVisible({ timeout: 15_000 });

  // select two checkpoints (one per role) -> frozen evaluation
  const table = page.getByLabel("all checkpoints");
  await table.locator("tr", { hasText: "attacker" }).last().getByRole("checkbox").check();
  await table.locator("tr", { hasText: "defender" }).last().getByRole("checkbox").check();
  await page.getByLabel("Episodes").fill("4");
  await page.getByRole("button", { name: "Evaluate selected (frozen)" }).click();
  await expect(page.getByText(/evaluation eval-\d+/)).toBeVisible();

  // comparison view shows the stored result (with CI) and DEBUG forks stay out of it
  await page.getByRole("link", { name: "Experiments", exact: true }).click();
  await expect(page.getByText("fake evaluator")).toBeVisible();
  const suite = page.getByLabel("Suite frozen");
  await expect(suite).toBeVisible({ timeout: 10_000 });
  await expect(suite.locator("tbody tr")).toHaveCount(1);
  await expect(suite).toContainText(/\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]/);
  await expect(page.getByLabel("Cross-play matrix")).toBeVisible();
  await expect(page.getByText("debug_fork")).toHaveCount(0);

  // replay from the stored export
  await suite.getByRole("link", { name: "replay" }).click();
  await expect(page).toHaveURL(/\/world\/eval_/);
  await expect(page.getByText("REPLAY", { exact: true })).toBeVisible();
  await expect(page.getByText("FIXTURE", { exact: true }).first()).toBeVisible();

  // artifacts + infrastructure reflect the same stored facts
  await page.getByRole("link", { name: "Artifacts", exact: true }).click();
  await expect(page.getByLabel("Checkpoint lineage").locator("tbody tr").first()).toContainText("root");
  await expect(page.getByLabel("Run artifacts")).toContainText("manifest verified");
  await page.getByRole("link", { name: "Infrastructure", exact: true }).click();
  await expect(page.getByLabel("Jobs")).toContainText("fake:run-0001:a2");
  await page.getByRole("button", { name: "Dry run" }).click();
  await expect(page.getByLabel("Cleanup plan")).toContainText("Dry run — nothing deleted.");
  await expect(page.getByLabel("Cleanup plan")).toContainText("run-0001/a1");
  expect(await page.request.get("/api/checkpoints").then((r) => r.json()).then((j) => j.checkpoints.every((c: { available: number }) => c.available === 1))).toBe(true);

  // reload keeps server truth
  await page.goto("/training");
  await expect(page.getByLabel("Runs").getByText("cancelled", { exact: true })).toBeVisible();
});

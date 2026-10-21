import { expect, test } from "@playwright/test";

test("build board -> world -> perspective switch -> reload", async ({ page }) => {
  // /build: DAG from the real orchestrator plan; honest runtime state
  await page.goto("/build");
  await expect(page.getByRole("heading", { name: "Build" })).toBeVisible();
  await expect(page.getByText("coding runtime not configured").first()).toBeVisible();
  await page.getByRole("button", { name: "experiment-pipeline" }).click();
  await expect(page.getByRole("heading", { name: "experiment-pipeline" })).toBeVisible();
  await expect(page.getByText("No acceptance results yet")).toBeVisible();
  await expect(page.getByRole("button", { name: "Integrate" })).toBeDisabled();
  await expect(page.getByText(/stream connected/)).toBeVisible();

  // /world/:id — defender by default, labelled FIXTURE + REPLAY, LIVE not configured
  await page.getByRole("link", { name: "World" }).click();
  await expect(page).toHaveURL(/\/world\/fixture_tiny_episode/);
  await expect(page.getByText("REPLAY", { exact: true })).toBeVisible();
  await expect(page.getByText("FIXTURE", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("LIVE: not configured")).toBeVisible();
  await page.getByRole("button", { name: "Step", exact: true }).click();
  await expect(page.getByText("increase_logging").first()).toBeVisible();

  // switching perspective hits a different endpoint and swaps the visible data
  const req = page.waitForRequest((r) => r.url().includes("/api/world/attacker/episodes/fixture_tiny_episode/replay"));
  await page.getByLabel("Perspective").selectOption("attacker");
  await req;
  await expect(page).toHaveURL(/p=attacker/);
  await expect(page.getByRole("cell", { name: "discover" })).toBeVisible();   // frame 0 loaded
  await page.getByLabel("Scrub").press("ArrowRight");   // keyboard scrub
  await expect(page.getByRole("cell", { name: "attempt_initial_access" }).first()).toBeVisible();
  await expect(page.getByRole("cell", { name: "increase_logging" })).toHaveCount(0);

  // evaluator is privileged: labelled, and refused without its token
  await page.getByLabel("Perspective").selectOption("evaluator");
  await expect(page.getByRole("alert").filter({ hasText: "PRIVILEGED" })).toBeVisible();
  await expect(page.getByRole("alert").filter({ hasText: "401" })).toBeVisible();
  await page.getByLabel("evaluator token").fill("e2e-evaluator-token");
  await expect(page.getByText("episode total (evaluator only; not per tick)")).toBeVisible();   // ground truth only after auth

  // reload keeps the perspective (URL) and re-reads server state
  await page.getByLabel("Perspective").selectOption("attacker");
  await page.reload();
  await expect(page.getByLabel("Perspective")).toHaveValue("attacker");
  await expect(page.getByText("FIXTURE", { exact: true }).first()).toBeVisible();

  // build state survives a reload too (server is the source of truth)
  await page.goto("/build");
  await page.reload();
  await expect(page.getByRole("button", { name: "experiment-pipeline" })).toBeVisible();
});

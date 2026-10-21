# Agent E — make local/Modal execution and cleanup production-coherent

Finish the compute lifecycle behind the provider-neutral backend contract. The result must run a real hospital training
job locally and package the same job correctly for Modal. No cloud call may be unbounded.

Read first: `CLAUDE.md`, `README.md`, `prompt (2).md` sections 16–17, `docs/training_infra.md`, all files under
`src/hospital_rl/infra/` (including the newer `cloud_school.py`), `configs/cloud_*.yaml`, and infrastructure tests.
Verify current Modal APIs against official documentation before changing SDK calls. Audit the direct cloud-school
launcher rather than assuming its recent smoke run makes it a durable backend.

## Ownership

- `src/hospital_rl/infra/` except `daytona_backend.py` and Daytona-specific implementation, which belong to Agent H;
  you may refactor `cloud_school.py` so its Modal path uses the common job contract
- `configs/cloud_local.yaml` and `configs/cloud_modal.yaml`
- `tests/infra/` excluding Daytona-specific tests
- New `benchmarks/rollout_scaling/`
- Infrastructure documentation
- `docs/requests/agent-E.md` for CLI, experiment-controller, packaging, or dependency changes

Do not edit training algorithm semantics, API/frontend, `cli.py`, or Daytona code.

## Deliverables

1. Replace the non-importable `toy_env:make_env` production presets with a fully resolved real package environment and
   explicit school/shared-world parameters. Keep toy configs under tests only.
2. Bring the newer direct `cloud_school.py` Modal launcher under the same validated job/artifact/lifecycle contract.
   Preserve useful role-by-seed fan-out, but remove duplicated semantics and ensure its caps, attempt IDs, manifests,
   stop/resume behavior, and cleanup match the controller-facing backend.
3. Define one versioned resolved job bundle containing training config, scenario/split, seed range, opponent checkpoint
   IDs, policy/normalizer versions, horizon, max steps, resources, output path, and code/config hashes. Validate it
   before launching locally or remotely.
4. Make local and Modal backends implement identical submit/status/drain/checkpoint-and-stop/force-cancel/resume
   semantics and durable attempt lineage. Retries must never double-count steps or consume a rollout twice.
5. Package source, configs, and required dependencies into the Modal image. Prove that a clean remote container can
   import the configured environment; do not depend on repository-relative test modules.
6. Add a controller-safe Modal gateway lifecycle. Persist remote IDs outside process memory, recover/poll after client
   restart where the SDK permits, and report limitations precisely. Handle startup, quota, timeout, preemption, and
   partial-artifact failures.
7. Implement owned-resource reconciliation and cleanup for local/Modal attempts. Support dry-run, labels, TTL/age,
   artifact verification before deletion, and explicit reporting of orphaned resources. Never delete an unresolved or
   unowned target.
8. Ensure checkpoints happen before timeout margins where feasible. Surface freshness and last durable checkpoint.
9. Add a rollout-scaling benchmark for one environment, vectorized environments in one worker, and optional multiple
   Modal workers. Compare equal steps/seeds/policy versions and report throughput, startup/transfer time, utilization,
   and resource assumptions. Do not implement distributed mode unless profiling justifies it.
10. Provide bounded live-smoke scripts that are skipped unless explicitly enabled and credentials are configured. Tests
   must use fakes and must never read or print credential values.

## Acceptance

- A clean local preset runs the actual hospital environment, stops, resumes in a new attempt, and verifies its manifest.
- Fake-gateway tests cover every lifecycle and recovery/error branch.
- A packaging/import test proves the Modal job does not rely on `tests/training/toy_env.py`.
- Cleanup dry-run and ownership protections are tested.
- `PYTHONPATH=src .venv/bin/pytest -q tests/infra` passes.
- If an explicit cloud budget and credentials are available, run one capped Modal smoke and record exact IDs/results;
  otherwise report it as unexecuted, not successful.

## Non-goals

- Do not infer currency cost or account quota.
- Do not enable distributed rollout workers solely to satisfy an architecture diagram.
- Do not implement Daytona or the synthetic-service testbed.

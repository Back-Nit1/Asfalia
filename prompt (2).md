# Hospital Security RL Simulator — Implementation Prompt

You are the lead implementation orchestrator, coordinating specialist coding agents and integrating their verified changes. Build a reproducible research platform for training attacker and defender policies separately, transferring their checkpoints into a shared simulated hospital, and evaluating security against disruption of clinical work. Implement working software incrementally; do not stop at architecture or placeholder functions. Read existing repository instructions before changing files. Preserve existing work. Treat the milestones below as execution gates, not claims of completed results.

## 1. Thesis and boundaries

The simulator, task suite, trajectories, and benchmark are the primary product. Learned policies demonstrate its usefulness. The hypothesis is that separate curricula followed by shared-world training improve generalization or sample efficiency while respecting operational constraints. Test this against training from scratch; do not assume it.

Use synthetic assets, synthetic patient records, and abstract attack transitions. The first release is a graph simulator, not a replica of a real hospital or a HIPAA compliance assessment. No real hospital connections, patient data, exploit payloads, arbitrary shell tools, or external attack targets. Agents interact only through typed simulator actions. A later optional database testbed remains isolated and uses synthetic records.

Do not repeat claims that real security data is unavailable, that partial observability requires RL, that self-play is necessary, or that this simulator is novel without comparison. Those are hypotheses or overstatements. Explain the limits of synthetic evidence.

## 2. Three environments, one engine

Build one transition engine and three task configurations, rather than three incompatible simulators:

1. **Attacker curriculum sandbox:** small randomized worlds, scripted/frozen defenders, gradually harder discovery, access, collection, and simulated exfiltration tasks.
2. **Defender curriculum sandbox:** small randomized worlds, scripted incidents and benign workload, tasks covering access restriction, remediation, containment, recovery, and budget allocation.
3. **Shared hospital world:** both trained policies act against each other under hospital access rules, resource budgets, workflow dependencies, and partial observability.

“Moving an agent” means loading its model weights, observation normalizer, recurrent-state specification, and versioned action/observation vocabulary into a compatible adapter. It does not mean copying a process or granting tools. Reset recurrent state at episode boundaries. Freeze normalization statistics during evaluation. Reject incompatible checkpoints with a readable error.

Distinguish hospital access policies (RBAC, maintenance windows, protected services) from learned RL policies (action selectors). Permissions are enforced by the world engine, never by prompt instructions alone.

Keep development agents distinct from the learned policies. Initially use exactly two learned roles: one attacker and one defender. The defender handles both posture and incident response. Multiple independent attackers or specialist fixers are a later extension because coordination and credit assignment add complexity. Model multiple incidents with the initial two-role API before adding more policies.

## 3. Stack and repository

Use Python 3.11+, NumPy, NetworkX, PyTorch, Gymnasium spaces, PettingZoo ParallelEnv, Pydantic, PyYAML, pytest, and a small CLI. Verify compatible current versions, lock dependencies, and record versions in artifacts. Use a minimal masked PPO trainer with a shared implementation for either role and a frozen-opponent single-agent adapter. Prefer a maintained compatible trainer if verified to support these spaces and masks; document the choice and test it. Do not create a custom RL algorithm as a project goal.

Proposed layout:

```text
pyproject.toml
src/hospital_rl/
  schemas.py
  engine/{state,transitions,workflows,telemetry,permissions,rewards}.py
  envs/{parallel_env,single_role,curricula,encoding}.py
  policies/{base,random,scripted,heuristic,ppo,checkpoint}.py
  training/{train,league,curriculum}.py
  evaluation/{benchmark,counterfactual,metrics,splits}.py
  data/{trajectory,replay}.py
  cli.py
configs/{tiny,attacker_school,defender_school,hospital40}.yaml
scenarios/{train,validation,test}/
tests/
docs/{architecture,model_card,limitations}.md
```

Build an operational dashboard early, immediately after the tiny simulator and event contracts work. It must show real development-agent events and real episode events before long training starts. Sections 14–18 specify orchestration, UI, Modal/Daytona execution, and C++ inference. Ship a thin working vertical slice before expanding the interface; a static report alone is not the requested frontend.

## 4. World state

Begin with an 8-node fixture and scale to approximately 40 nodes through configuration. Represent EHR, PACS, workstations, billing integration, identity provider, VPN, application/K8s services, and medical devices. Include a service-dependency graph distinct from network reachability and identity trust.

Node fields: stable internal ID, asset type, zone, service availability, synthetic record inventory, sensitivity weight, patchability, abstract vulnerability flags, logging coverage, encryption configuration, credential references, and active remediation jobs. Edges encode reachability, trust, authorization, and workflow dependencies explicitly. Version schemas.

Maintain identities, roles, sessions, scoped credentials, MFA, and record access separately from machine compromise. Network access is not authorization. Encryption at rest does not prevent an authorized compromised session from reading records. Credential rotation revokes relevant credentials/sessions according to explicit semantics. Patching prevents specified future transitions but does not automatically remove existing compromise. Isolation blocks specified edges but does not erase compromise or restore services.

Some devices cannot be patched during an episode. Compensating controls include segmentation, access restriction, and monitoring. Unpatchability is a scenario assumption, not a claim about every medical device.

Model benign workflows: clinician chart lookup, image retrieval, device telemetry, and billing sync. Each has a request rate, necessary service/identity path, deadline, and criticality. Calculate successful and failed requests from actual simulated permissions and service state. Include emergency-access requests with bounded, audited simulated overrides. Costs must emerge from these workloads rather than arbitrary labels such as “MFA annoys clinicians.”

## 5. Actions and timing

Attacker action types: wait, discover, probe, attempt_initial_access, attempt_privilege_change, move, collect, exfiltrate. These are probabilistic state transitions with modeled prerequisites and costs, not executable attacks. Add verified MITRE tactic/technique metadata where an abstraction actually corresponds; document incomplete coverage.

Defender action types: wait, enable_mfa, restrict_role, revoke_session, rotate_credential, patch, isolate, restore_connectivity, increase_logging, quarantine, reimage, restore_service. Every action specifies target, prerequisites, duration, spend, recurring costs, and workflow consequences. Respect maintenance windows and action capacity. Include reversible controls where applicable. Recovery must have explicit preconditions and residual-risk semantics.

Use a bounded Discrete action catalog: action type × padded target slot, plus wait. Exclude unused combinations using masks. Start with single-target actions; defer arbitrary source-target-action combinations. Use a configurable maximum of 64 nodes initially, with smaller worlds padded. Use deterministic per-episode slot permutation to discourage memorizing asset positions. Mask padding throughout encoding and pooling.

Masks must depend only on the acting role’s observable information and explicit known permissions. A mask must not reveal an unknown vulnerability, hidden compromise, or hidden edge. An action that looks possible but fails on hidden prerequisites is an observable, costed attempt. Invalid catalog indices raise errors; legitimate failed attempts remain transitions. Always keep wait legal.

Parallel actions are selected from the same pre-step observations. Resolve both through a documented event scheduler with durations and explicit tie rules. Do not inspect the opponent’s current action when selecting a move. Test simultaneous containment/exfiltration and patch/access cases. Record tie policy in environment metadata and evaluate sensitivity to it later. One tick is an abstract interval; do not claim wall-clock fidelity.

## 6. Observation and information separation

Attacker receives discovered topology, own access/credentials, probe outcomes, and visible defenses. Defender receives asset inventory and authorized posture information plus noisy, delayed, or missing auth, flow, and process events. Defender does not receive true attacker phase, hidden footholds, or attack success flags.

Use padded node features, visible adjacency, bounded event history, role-local inventory, action mask, time, and budget. Implement a masked node encoder with pooled context; add a GRU after a feed-forward baseline if memory improves held-out performance. Training can begin with a history stack. Do not claim the environment is Markov from the agent’s observation.

Keep true state and labels in a separate evaluator-only recorder. Training observations, deployment inputs, masks, logs exposed to agents, and exported model inputs must pass leakage tests. Privileged critics, if later introduced, must be explicitly distinguished from actors and evaluated without privileged inputs.

## 7. Objective and constraints

Track security and care availability separately. Use an illustrative initial defender reward, configurable and clearly labeled uncalibrated:

```text
r_defender = -5 * incremental_unique_sensitive_exfiltration_fraction
             -2 * critical_workflow_failure_fraction_this_tick
             -1 * noncritical_workflow_failure_fraction_this_tick
             -0.2 * normalized_action_spend_this_tick
             -0.1 * normalized_added_workflow_latency_this_tick
```

Normalize against documented scenario constants, not statistics fitted on the test set. Charge recurring controls per tick and one-time actions once. Count each synthetic record at most once for total exfiltration reward; sensitive records accessed and sensitive records exfiltrated are distinct metrics. Report records as well as sensitivity-weighted exposure.

Attacker reward uses incremental unique exposure minus time and explicit defender-observable detection events, not arbitrary omniscient detection probability. Tune weights on validation only. The game is general-sum because defender care costs need not benefit the attacker.

Enforce hard spend/capacity constraints in the engine. Evaluate availability feasibility separately from scalar return: make a configurable illustrative critical-workflow success threshold, initially 95%, and report violations. The number is a benchmark parameter, not a clinical standard. Rank feasible policies on exposure and cost; show infeasible policies explicitly rather than hiding them in aggregate reward. Later consider a constrained RL/Lagrangian method only if needed.

Include an isolate-everything baseline. It must incur severe care failure and fail the availability constraint. Do not end the episode early in a way that lets a policy avoid future downtime penalties. Use fixed horizons, or account explicitly for remaining-horizon costs. Prevent positive rewards for cycling controls, duplicate exfiltration, or leaving remediation jobs unfinished at termination.

## 8. Training curriculum and transfer

Stage A: validate the engine with scripted agents, random policies, no-op, static hardening, and a budget-aware greedy defender. Include benign-only episodes.

Stage B: train attacker on randomized small worlds against a mixture of scripted/frozen defenses. Progress by validation success, not only training return. Include partial task progress for learning but report unshaped objective metrics. Add difficulty in topology, credentials, visibility, and restrictions.

Stage C: independently train defender against mixed scripted incidents and benign traffic. Include posture prevention, incident containment, and recovery. Use whole episodes to measure delayed care costs. Do not train against a single predictable schedule.

Stage D: load both checkpoints into the shared hospital world and evaluate with both frozen before further training. This is the direct sandbox-to-world transfer measurement. Report performance degradation by topology, task, and visibility shift.

Stage E: adapt one side while freezing the other; alternate updates only after stable baselines. Sample opponents from a bounded league containing scripted agents and historical checkpoints. Retain a separate evaluation league that is never used for updates or checkpoint selection. Use cross-play matrices to reveal cyclic strengths and regressions.

Primary experiment: compare separate pretraining plus adaptation against random initialization plus the same shared-world budget. Also compare equal TOTAL environment-step budgets including pretraining, so extra compute cannot masquerade as better transfer. Track training time and inference latency.

Initial smoke run: CPU, tiny worlds, bounded steps. Pilot: 100k steps per role as a configurable starting budget, not a success guarantee. Scale only after profiling. Default to a local smoke run. Implement Modal as the primary cloud backend and Daytona as an optional sandbox backend, with explicit configured resource caps and a dry-run estimate before cloud submission. Run bounded configured cloud smoke tests when credentials and an explicit run budget are available; missing credentials must not block local work. Do not start unbounded cloud jobs or model-API loops. Support resume and interrupt-safe checkpointing. Record seeds, RNG states, optimizer, normalization, schema versions, action catalog, config hash, code revision when available, and training steps.

## 9. Benchmark and evaluation

Split topology families, attack schedules, vulnerability combinations, workload profiles, and seeds before training. Seeds alone are not an adequate split. Keep test scenarios out of curriculum advancement, normalization fitting, reward tuning, and opponent selection. Version and hash the benchmark.

Report mean, spread, and confidence intervals across at least five training seeds when running substantive experiments; support a one-seed smoke test clearly labeled insufficient for conclusions. Evaluate all baselines on identical held-out scenario sets.

Metrics: unique sensitive records exfiltrated, critical/noncritical workflow success, downtime, latency, spend, containment time, recovery time, detection false positives on benign episodes, attacker success, and constraint violations. Mark containment time undefined/censored when containment does not occur; do not substitute zero. Report individual metrics and a security/availability/cost tradeoff plot, not only reward.

Ablations: no pretraining, no adaptation, no workflow cost, no domain randomization, no opponent league, full observability diagnostic, and different telemetry-loss levels. Run expensive ablations after the first complete pipeline. Include randomized legal actions to look for state-invariant violations and degenerate strategies.

## 10. Trajectories and counterfactual data

Export versioned JSONL first; optional Parquet later. One row per decision records episode ID, role, step, role-visible observation, mask, chosen action, reward components, termination/truncation, checkpoint IDs, scenario/config hashes, and seed. Store evaluator labels and true states in a separate keyed artifact with an explicit privileged-data designation.

Provide two counterfactual modes:
1. Replay fixed attacker intent/actions against different defender configurations, allowing blocked actions to fail. This estimates the effect for that fixed attack sequence.
2. Rerun a frozen adaptive attacker under different postures. The realized attack path may change; report this as a different estimand.

Use keyed random streams for workload, telemetry, transition outcomes, and attacker stochastic choices so changes in one branch do not accidentally shift all randomness. Use shared exogenous draws where meaningful, not blind reuse of one global seed. Record interventions and uncertainty. These are simulator counterfactuals, not established causal effects in hospitals.

## 11. Optional synthetic database testbed and real-data evaluation

Only after the graph benchmark works, add an adapter for an isolated synthetic testbed, locally first and then optionally in Daytona: PostgreSQL with invented patient/chart metadata, a mock identity service, an imaging metadata service, and benign workload generators. Use disposable isolated runtimes, an internal synthetic-service network, no host credentials, and no outbound attacker-policy access. Export observability through a separate trusted worker channel; do not give the simulated actors provider credentials. Keep storage/audit collection separate from records the attacker can access. Agents still invoke allowlisted typed actions via an adapter; no shell or SQL execution generated by agents. Add explicit resource limits and reset each episode.

This testbed evaluates graph-to-testbed transfer and transition-model mismatch. It is not a live hospital or proof of real-world effectiveness. Synthetic FHIR-shaped records may improve interoperability testing but do not make network behavior realistic.

Public traces such as LANL, OpTC, or CICIDS can support narrowly scoped telemetry/detection evaluation after inspecting their licenses, labels, schemas, and task fit. Passive traces do not provide the outcome of unobserved defensive interventions, so do not report offline replay as validation of remediation or posture-control effectiveness. Separate simulator generalization, testbed transfer, and real-trace detection results.

Fine-tuning a small open language model is optional and last. Export only available observations as inputs and bounded structured decisions as targets. Do not fabricate reasoning chains or expose privileged labels as actor inputs. Benchmark a simple detector/policy before adding an LLM. Natural-language serialization does not solve domain shift. Report model license, compute, task, schema adapter, and held-out results.

## 12. Tests and acceptance gates

Gate 1 — engine: 8-node fixture, synthetic workflows, permissions, duration/cost accounting, deterministic seeded replay, and complete scripted episode. Test unpatchable devices, session revocation, persistence after patching, workflow disruption/restoration, record accounting, and budget enforcement.

Gate 2 — environment: PettingZoo API test passes; spaces and masks remain valid through randomized episodes; hidden-state perturbations do not change observations or masks unless emitted evidence changes. Both actions use pre-step observations. No externally reachable execution path exists.

Gate 3 — separate training: each role has a working training command, periodic checkpoint, resume, and held-out evaluation against its sandbox opponents. Verify weights change and metrics are real. Do not promise superiority after a tiny run.

Gate 4 — transfer: both checkpoints load into hospital40; frozen evaluation produces a report; an incompatible checkpoint fails explicitly; equal-budget scratch and pretrained experiment configurations exist.

Gate 5 — league and data: alternating training, bounded opponent pool, cross-play report, validated trajectory export, both counterfactual modes, and replay. Privileged data is separate from actor inputs.

Gate 6 — reporting, in addition to the required dashboard and infrastructure gates in section 18: README with exact runnable commands, architecture, metric definitions, sample tiny-world report, failure cases, and limitations. Clearly distinguish implemented features, smoke-tested paths, experiments actually run, and deferred work. No invented results.

Suggested CLI contract, implemented as real commands:

```bash
hospital-rl demo --config configs/tiny.yaml --seed 7
hospital-rl train --role attacker --config configs/attacker_school.yaml --steps 100000 --seed 7
hospital-rl train --role defender --config configs/defender_school.yaml --steps 100000 --seed 7
hospital-rl evaluate --config configs/hospital40.yaml --attacker-checkpoint runs/attacker/best.pt --defender-checkpoint runs/defender/best.pt --split validation
hospital-rl adapt --config configs/hospital40.yaml --league --steps 100000 --seed 7
hospital-rl benchmark --config configs/hospital40.yaml --split test
hospital-rl counterfactual --episode runs/demo/episode.json --mode fixed-intent
hospital-rl replay --episode runs/demo/episode.json
pytest
```

Specify concrete checkpoint arguments for adapt/benchmark in CLI help; do not silently choose unrelated runs. Keep outputs under explicit run directories. Test commands against the actual implementation and correct documentation if interfaces change.

## 13. Updated execution order

Implement sections 14–18 as part of the product, not a future wishlist. The orchestrator should begin with schema/interface contracts and a tiny scripted episode, then deliver a real dashboard showing that episode and coding-task events. Continue to separate training, checkpoint transfer, Modal execution, and league/data evaluation. Add Daytona execution after local/Modal parity. Profile and implement the C++ inference adapter after the first learned checkpoint exists. Keep each milestone runnable and distinguish executed paths from planned work. Do not wait for all components to finish before integrating them.

## Reference starting points

- PettingZoo Parallel API: https://pettingzoo.farama.org/api/parallel/
- CyberBattleSim, related abstract network simulation: https://github.com/microsoft/CyberBattleSim

Review related systems before making novelty claims. The proposed contribution is the experimentally tested combination of separate curricula, healthcare workflow constraints, and security/availability tradeoffs—not the mere existence of attacker/defender RL.


## 14. Development orchestrator and specialist coding agents

### Distinct control planes

There are three systems, with separate identities, permissions, event types, and dashboard labels:

- **Development agents:** coding assistants that implement/test this repository. Their tool calls can edit their assigned checkout and run build commands. They are not the attackers or defenders being trained.
- **Experiment controller:** a deterministic scheduler for training runs, checkpoints, rollout batches, evaluation, and cloud resources. An LLM may propose a configuration but cannot silently alter registered experiments or test splits.
- **RL policies:** numerical attacker/defender models that select typed world actions. They do not receive coding tools, cloud credentials, or access to the development control plane.

The development orchestrator decomposes implementation into tasks, assigns owners, resolves dependencies, integrates changes, and runs acceptance checks. It must be able to explain which files, commits, tests, and artifacts support a task status. No nested agent spawning by default. Use a bounded specialist pool; start with at most three simultaneous coding agents, configurable to six after integration is stable. Roles below are work responsibilities, not a requirement to keep every role running continuously.

### Responsibilities and ownership

| Specialist | Owned work | Integration dependency |
| --- | --- | --- |
| World/clinical workflows | Transition engine, reachability, identity rules, workload dependencies, world fixtures | Shared schema contract |
| Attacker curriculum | Attacker observations, scripted opponents, attacker tasks and training config | Engine and encoding contracts |
| Defender curriculum | Defender observations, remediation, cost accounting, defender tasks/config | Engine and workflow contracts |
| RL/experiment infrastructure | PPO integration, rollout protocol, checkpoint transfer, league, Modal/Daytona adapters | Environment and policy contracts |
| Dashboard/API | React interface, FastAPI, event projection, run controls, episode replay | Versioned events and command schemas |
| Evaluation/performance | Leakage/invariant tests, benchmark splits, cross-play, profiling, C++ inference adapter | Frozen schemas and working reference policy |

The lead owns shared schemas, dependency lockfiles, task graph, integration branch, and architectural decisions. Specialists propose schema changes to the lead rather than independently modifying shared contracts. Reviewer/evaluation work must include independent checks, not only assertions written by the component author.

### Task lifecycle and execution

Persist a Task record with ID, role, description, dependencies, acceptance commands, assigned paths, base commit, branch/worktree, status, heartbeat, lease expiry, attempt count, artifact references, and concise outcome. State machine: queued → ready → running → review → passed → integrated; failed/blocked/cancelled are explicit states. Store completion evidence; “the agent said done” is insufficient.

Use isolated git worktrees locally or a separate Daytona sandbox/checkout per active coding task. Workers start from a pinned commit. Prevent multiple owners writing the same paths; serialize lockfile and shared-schema changes. Upload/commit changes before terminating workspaces. The lead reviews diffs, integrates in dependency order, resolves conflicts, and runs the combined suite. Do not merge solely because an individual branch passed.

Implement a `CodingAgentAdapter` with task submission, process/session ID, streamed observable events, cancellation, result collection, and capability checks. Integrate one real configured coding-agent runtime using its documented CLI/SDK. When a parent coding tool supports subagents, it may use that capability during the initial build, but that does not automatically create a reusable in-product orchestrator. The repository needs its own adapter/event contract. If no runtime/model credentials are configured, show “coding runtime not configured” and support human-owned tasks; do not animate fake agent progress. A test adapter is allowed only in tests and visibly labeled demos.

Record public action summaries, tool outcomes, changed files, stdout/stderr with redaction, test results, and task decisions. Do not request or display hidden chain-of-thought. Enforce task timeouts, cancellation, bounded repair retries (initially two), maximum concurrent tasks, and configurable model-token limits when supported. A failed specialist becomes blocked with evidence after retries; it must not recursively spawn more agents. Keep provider keys outside prompts, source code, and browser responses.

The orchestrator is tooling that runs on demand while developing this repository, not a permanent self-modifying service operating alongside RL training. Production experiment workers execute pinned code; coding agents cannot modify a live registered run.

## 15. Developer dashboard and actual user experience

### Product shape

Build a functional React + TypeScript + Vite frontend, a FastAPI backend, typed REST commands, and server-sent events (SSE) for updates. Use a restrained dark research-console style: clear typography, sidebar navigation, compact metric cards, readable tables, and restrained amber/red for actual warnings. Support accessible contrast, keyboard controls, and labels beyond color. No decorative terminal wallpaper or fabricated live counters. Integrate a graph component such as React Flow after checking the current documented API. No browser-to-cloud-provider credentials.

The developer opens the dashboard and sees an overview of the build, the current experiments, and infrastructure capacity. They can inspect what a coding agent changed, launch bounded experiments from validated configurations, watch a selected simulated hospital episode, compare checkpoints, and replay a failure. They are observing tasks and state transitions, not agents with humanoid avatars walking through a hospital.

### Main screens

| Route | Main content | Useful controls |
| --- | --- | --- |
| `/overview` | Build milestone, run counts, healthy/stale workers, latest evaluation, current step rate and spend estimate | Open active task/run; launch tiny demo |
| `/build` | Task dependency graph and status board, role, current task, heartbeat, changed files, test evidence, integration status | Inspect diff/logs; retry bounded task; cancel; integrate only after checks |
| `/training` | Separate Attacker School, Defender School, Shared Hospital tabs; curves, curriculum level, opponent IDs, checkpoint lineage | Validate config; start; checkpoint-and-stop; resume from checkpoint; bounded evaluation |
| `/world/:episode_id` | Hospital topology, role-visible events, selected node properties, current actions and action outcomes, clinical workflow health | Attacker/defender perspective; select episode; replay; fork into a separate debug run |
| `/experiments` | Baselines, pretraining vs scratch, cross-play matrix, security/availability/cost comparisons, confidence intervals | Filter seed/split; compare compatible runs; export report |
| `/infrastructure` | Modal jobs and Daytona sandboxes, resources, age, lifecycle state, heartbeat, failures, policy version, cost-estimate freshness | Drain workers; stop/delete owned sandbox; set future concurrency caps; inspect cleanup status |
| `/artifacts` | Checkpoints, scenario/config hashes, trajectories, benchmark versions and provenance | Download; inspect model metadata; promote a compatible checkpoint |

### World screen layout

Top strip: run/episode ID, simulated tick, policy checkpoint IDs, LIVE / REPLAY / DEBUG marker, last event age, and perspective. Metric cards: unique sensitive records exfiltrated, critical workflow success, cumulative simulated operational spend, and remaining action budget. Keep actual cloud/model spend elsewhere and label it separately from simulated hospital spend.

Center: a stable-layout topology grouped by network zone; node glyphs show asset type. Visible state overlays indicate reachable, isolated, degraded, or unknown as permitted by the selected perspective. Right panel: selected asset details, active jobs, known controls, and relevant workflow dependencies. Bottom: time-ordered action feed with actor, target, observed outcome, duration, and reward components. Example content must be sourced from actual engine events: attacker probe outcome; defender credential rotation; imaging request delayed by segmentation. Include critical-workflow success next to containment, so over-securing is visibly costly.

Attacker and defender perspective endpoints serialize only their permitted observations. Evaluator mode is separately authorized, prominently labeled privileged, and served from a distinct endpoint/DTO. Hiding truth using CSS or a frontend toggle is insufficient. Coding-task streams must also be inaccessible to RL policies.

Live mode shows a sampled episode and aggregate worker metrics; do not attempt to stream every step from every environment. Throttle visual updates to roughly 2–5 Hz and metric aggregation to around 1 Hz, configurable. Training continues at full speed independently of rendering. Replay supports play/pause, step, speed, and scrub from stored snapshots plus events. Pausing replay does not pause training. A sampled live training episode is read-only; interactive control experiments use a separate forked debug episode excluded from benchmark results.

### Backend contracts and durability

Use a versioned EventEnvelope: event_id, source_kind (development/experiment/world/infrastructure), event_type, run/task/episode IDs as applicable, worker_id, per-source sequence, timestamp, simulated tick if relevant, visibility scope, schema version, and typed payload. No global ordering assumptions across workers. Deduplicate by event ID, order within a source, and use bounded queues/backpressure. Downsample UI telemetry without dropping full required trajectory records.

Persist task/run state and events. Use SQLite WAL with one local service process for development; use PostgreSQL for a multi-container cloud control plane. Do not put concurrent SQLite writers on a shared Modal Volume. SSE reconnect uses event cursors and can resync from snapshots after event retention expires. Browser refresh or backend restart must not manufacture task completion or lose run ownership. Distinguish queued, starting, running, draining, checkpointing, stopped, failed, cancelled, and completed runs.

Minimum APIs: tasks/status/events/artifacts; runs/create/status/cancel/resume; checkpoints/list/metadata; episodes/snapshot/events/replay; metrics/query; infrastructure/list/drain/cleanup. Mutation requests use idempotency keys and validated typed parameters. Show backend errors, stale data, disconnected providers, absent credentials, empty runs, and partial artifacts honestly. Start cloud jobs asynchronously; do not hold a request open for training. Use authenticated access, server-side authorization, and command auditing for cloud deployment. Keep the local dashboard bound to localhost by default.

UI run sequence: choose a preset → see resolved configuration, resource caps, and estimated resource-hours → submit → get a durable run ID → watch startup and metrics → checkpoint-and-stop → select both school checkpoints → frozen shared-world evaluation → compare against baselines → optionally adapt with the league. Validation errors should explain missing or incompatible artifacts.

## 16. Modal primary compute backend

Implement `infra/modal_app.py` plus provider-neutral JobBackend and ArtifactStore interfaces. Verify current SDK versions and documented APIs before coding; do not copy unverified decorators or assume account quotas. Modal hosts CPU environment simulation and learner jobs, with optional GPU resources only when measured useful. The dashboard/API may be served through a Modal ASGI endpoint after local verification; cloud deployment includes authentication and durable external control-plane storage.

### Start simple, then distribute

First cloud mode: one bounded Modal job contains vectorized environments, inference, and learning together. This removes cross-provider networking from the first working training loop. Multiple small graph environments share one worker process or process pool. A logical RL environment is not a dedicated cloud sandbox.

Second mode, only after profiling: CPU rollout workers generate batches, and a learner performs updates. Each rollout request pins the scenario family/split, seed range, opponent checkpoints, trainable policy version, observation normalization version, horizon, max steps, and output path. Load a frozen local copy of policies into each worker and batch inference there. Avoid a network request to a central GPU for every action in a 40-node world.

PPO data collection is versioned and on-policy: collect a bounded rollout round with policy version v, record behavior log probabilities/value estimates and recurrent-state metadata, then update to v+1. Reject or explicitly handle stale/duplicate batches. Do not silently mix arbitrary historical trajectories into PPO. Report worker utilization and policy lag. Do not turn on distributed actors until a single-worker reference run is correct.

Persist checkpoints and immutable trajectory shards on Modal Volumes or a configured object store. Use unique paths by run/worker/rollout ID, atomic publication of completed artifacts, manifest hashes, and a single owner for mutable run indexes. Respect Volume commit/reload semantics from current docs; a shared mount is not automatically an immediately consistent event bus. A control-plane database or verified queue coordinates status and work. Keep the learner’s writes separate from worker output directories.

Set explicit max containers, per-job timeout, CPU/memory, rollout size, max training steps, and retry limits in configuration. Start with a small worker cap and scale 1 → 2 → 4 only if throughput warrants it. Expose dry-run resource-hours, estimate provenance/time, and hard resource caps. Do not infer available credits or real-time billing from configuration. Cost estimates are estimates; distinguish them from provider-reported usage. Limits on steps/runtime/resources are enforceable even when exact currency billing is delayed.

Handle startup failures, quota errors, worker preemption, and expired jobs. Checkpoint before timeout margins; resume with a new attempt ID and original run lineage. Rollout retries must not double-count data. Drain stops new assignments; checkpoint-and-stop lets the learner finish its safe boundary; force cancel is distinct and can lose recent work. Show all three accurately in the dashboard.

Use separate secret scopes for Modal, Daytona, the database, artifact access, and optional coding-model APIs. Provider secrets belong only to trusted controllers or worker bootstrap, never observations or RL actor tools. Do not expose unauthenticated run-launch/cancel endpoints.

## 17. Daytona sandbox backend and scaling hypothesis

Use Daytona for two concrete workloads:

1. Isolated coding-agent workspaces with pinned repository commits, controlled dependencies, test execution, artifact upload, and cleanup.
2. The optional synthetic-service hospital testbed, where each sandbox runs a bounded complete scenario bundle (mock identity, synthetic database, imaging service, workloads) rather than placing each network node in its own paid sandbox.

Implement an optional Daytona rollout backend to test whether its CPU sandboxes improve cost/throughput for the graph engine. It must implement the same versioned rollout contract as Modal. Do not make cross-cloud RL a prerequisite for the first working system. Platform sandbox creation is not curriculum generation: topology, controls, noise, attack schedules, and workloads come from the scenario generator inside each worker.

Create from a pinned image/snapshot when available. Pool a bounded number of warm sandboxes only if startup overhead is material. Reset episode state and synthetic data between tasks; provider snapshots and policy checkpoints are different artifacts. Do not assume a filesystem snapshot captures process memory or a database-consistent snapshot. Use explicit quiescing/export/reset protocols supported by the chosen runtime. Validate supported nested containers/services before designing around Docker-in-Docker.

Use labels for project/run/task/attempt ownership, lifecycle TTLs, idle-stop/cleanup rules, and a reconciler for orphaned resources. Track provider IDs in the backend, not in browser credentials. Always upload needed results before deletion. Sandbox failures must trigger bounded retries, not endless replacement. Concurrency and account limits are configurable and surfaced in the infrastructure screen. Package installation may need controlled outbound access during image build; experimental attacker policies still have no arbitrary egress or credentials.

For optional Daytona rollouts feeding a Modal learner, transfer compressed immutable rollout batches and checkpoint artifacts through the configured trusted transport. Include version/hash verification, bounded transfer timeouts, and per-worker scoped access. Avoid chatty per-step RPC. Benchmark transfer latency and egress as part of end-to-end cost; do not assume a shared filesystem or zero-cost connectivity between providers.

Run a scaling experiment: one environment process vs vectorized environments in one worker vs multiple Modal workers vs optional Daytona workers, using equal policy versions, scenario distribution, total environment steps, and documented resource allocations. Report steps/sec, time to target held-out performance, learning variance, cost per million transitions, startup/transfer time, and learner utilization. More parallel rollouts may reduce wall-clock collection time but do not automatically improve sample efficiency, exploration quality, or convergence. Prevent duplicated seeds and excessive correlation.

## 18. C++ inference path, build plan, and acceptance

### C++ has a measured role

Keep model training in PyTorch/Python. Build an optional **C++17 policy-inference runner** after the first reference policy works. Prefer a verified exported runtime such as ONNX Runtime C++ for the small feed-forward/GRU policy; alternatively use a compatible documented PyTorch AOTInductor C++ path if supported by the pinned model/operators/platform. Investigate/export once, document the decision, and avoid maintaining two native runtimes initially. C++ is not an LLM orchestrator rewrite and does not make already-native PyTorch tensor kernels inherently faster.

Implement `InferenceBackend` with Python-reference and C++ implementations. The adapter accepts the same encoded batch, masks, normalization, and optional recurrent state, and returns logits/value/new state. Define ownership, dtype, shape, contiguity, device, and error semantics. Keep world transitions and legal-action semantics outside the model. Use a pybind11 extension or a long-lived local runner; do not launch a subprocess per action. No network hop per decision. Ship an optional CMake build and a Python fallback for unsupported platforms.

Use a portable exported feed-forward policy as the initial native milestone; include GRU state only when actually supported and parity-tested. Export metadata with schema/catalog hashes, normalization constants, max node shape, model version, and training checkpoint identity. Test multiple batch sizes, padded graphs, legal masks, extreme finite inputs, and recurrent reset boundaries if applicable. Apply masks consistently and handle all-nonwait-invalid states safely. Use deterministic action selection for parity tests; stochastic sampling requires explicitly compatible RNG or distribution-level comparison.

Check numerical parity of logits/values within documented tolerances and action parity except near documented ties. Reject incompatible artifacts; never silently run a different model. Use a golden observation corpus spanning both roles and varied scenario sizes. C++ must not access evaluator state.

Benchmark Python eager, Python batched/compiled when compatible, and native C++ on identical hardware, model, observations, batch sizes, and warmup. Measure preprocessing + inference + postprocessing p50/p95 latency, throughput, memory, cold load/compile time, and full environment steps/sec with dashboard disabled/enabled. Report device and thread counts. Do not invent a speedup target or keep a slower native backend as the default just to say the project uses C++.

Profile the simulator separately. If Python graph traversal/observation construction dominates, a later compact-array C++ transition or encoding kernel behind pybind11 may offer more benefit than inference. Preserve the Python engine as a semantic oracle; require seeded state/reward/event parity and property tests before enabling any native simulation kernel. Do not port NetworkX or the whole engine without bottleneck evidence.

### Additional repository paths

```text
orchestration/{tasks,contracts,scheduler,adapters,events,integration}.py
infra/{backend,local_backend,modal_app,daytona_backend,artifacts,reconciler}.py
api/{app,auth,routes,events,storage}.py
frontend/src/{pages,components,api,events}/
native/{CMakeLists.txt,policy_runner.cpp,bindings.cpp}/
configs/{orchestrator,cloud_local,cloud_modal,cloud_daytona}.yaml
benchmarks/{inference,rollout_scaling}/
```

These paths may be placed beneath the existing package as appropriate; avoid duplicate competing service implementations. Shared DTOs and schema versioning are owned by the lead.

### Vertical slices and acceptance

1. **Contracts + tiny world:** lead owns schemas/event DTOs. World specialist builds Gate 1; API/UI specialist builds event ingestion and basic build/world screens against explicit fixtures labeled as fixtures. Replace fixtures with the real tiny episode immediately. Bring up the local task runner and one real coding-runtime adapter if credentials exist.
2. **Observable development and demo:** dashboard shows real task state, file/test evidence, and a real scripted episode. Browser reload and SSE reconnect work. The task scheduler respects dependencies, leases, ownership, retries, and concurrency. The lead integrates checked changes; no UI-only fake orchestration.
3. **Independent training and transfer:** attacker/defender specialists configure their curricula against shared engine contracts; infrastructure specialist implements shared training/checkpoints. Dashboard displays actual learning metrics and loaded checkpoint provenance. Both frozen policies run together in the shared world.
4. **Modal cloud:** bounded single-job smoke run has a real job ID, durable checkpoint, live status, recoverable failure, and verified cleanup. Local/cloud environment semantics match. Later distributed rollout rounds enforce policy versions and idempotency. Missing credentials produce setup instructions and a working local fallback, never a fabricated successful deployment.
5. **Daytona integration:** a bounded coding-task workspace can start, run a real test, return its artifact/commit, and clean up. Optional graph rollout backend uses the same protocol. Add the synthetic database testbed only after the graph benchmark gates; verify service isolation and repeatable reset. Mark each path actually tested or deferred.
6. **Native inference:** export a real trained policy, compile the runner, run parity tests, benchmark against Python, and display measured backend/latency in run metadata. If the toolchain is unavailable, report the precise missing prerequisite and leave tested Python inference usable; do not claim native execution occurred.
7. **End-to-end acceptance:** from a clean documented setup, open dashboard → inspect a coding task → run tiny demo → train both roles locally → transfer checkpoints → launch bounded Modal smoke run when configured → inspect metrics/replay → checkpoint-and-stop → resume → export comparison. Test dashboard authorization, hidden-state separation, cancellation, SSE reconnect, provider cleanup, checkpoint compatibility, and native parity. Include frontend typecheck/build and meaningful browser workflow checks.

Suggested new CLI surfaces, implemented and documented with exact arguments:

```bash
hospital-rl orchestrate --plan configs/orchestrator.yaml --dry-run
hospital-rl orchestrate --plan configs/orchestrator.yaml --max-agents 3
hospital-rl dashboard --host 127.0.0.1 --port 8000
hospital-rl cloud-plan --backend modal --config configs/cloud_modal.yaml
hospital-rl train --backend modal --role defender --config configs/defender_school.yaml --steps 10000 --seed 7
hospital-rl workers list --backend daytona
hospital-rl cleanup --run-id RUN_ID --dry-run
hospital-rl export-policy --checkpoint runs/defender/best.pt --format onnx
hospital-rl benchmark-inference --checkpoint runs/defender/best.pt --backends python,cpp
```

The implementer must make config resolution explicit: runtime flags select the provider config, which overlays resource settings onto the task config without changing registered task semantics. Document required secret names without values. Include real stop/resume, artifact paths, and compile commands. Orchestration can begin with a fixed task DAG and one coding adapter; no need for a complex autonomous planning framework. Keep the experience coherent: development agents build the platform, the experiment controller schedules compute, and the RL policies learn inside the worlds.

### Additional official implementation references

- Modal scale controls: https://modal.com/docs/guide/scale
- Modal artifact storage and consistency: https://modal.com/docs/guide/volumes
- Modal ASGI/web hosting: https://modal.com/docs/guide/webhooks
- Daytona sandbox lifecycle and configuration: https://www.daytona.io/docs/en/sandboxes/
- ONNX Runtime C++ API: https://onnxruntime.ai/docs/api/c/
- PyTorch AOTInductor: https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html

Verify these docs against pinned SDK/runtime versions at implementation time. Provider quotas, prices, availability, and runtime support must not be hardcoded from assumptions.

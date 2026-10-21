export type Perspective = "attacker" | "defender" | "evaluator";

export interface NotConfigured {
  configured: false;
  state: "not_configured";
  message?: string;
}

export interface TestResult { command: string; exit_code: number; duration_s: number; stdout_tail: string; stderr_tail: string }
export interface Evidence {
  changed_files: string[]; out_of_scope_files: string[]; test_results: TestResult[]; commits: string[]; agent_summary: string;
}
export type TaskStatus = "queued" | "ready" | "running" | "review" | "passed" | "integrated" | "failed" | "blocked" | "cancelled";
export interface Task {
  id: string; role: string; description: string; owner: "agent" | "human"; dependencies: string[];
  acceptance_commands: string[]; assigned_paths: string[]; base_commit: string; branch: string; worktree: string;
  status: TaskStatus; heartbeat: number | null; lease_expiry: number | null; started_at: number | null;
  attempt: number; session_id: string | null; artifact_refs: string[]; outcome: string; evidence: Evidence;
  path_conflicts?: string[];
}
export interface Runtime { configured: boolean; name: string | null; detail: string }
export interface TasksResponse { runtime: Runtime; now: number; tasks: Task[] }
export interface Snapshot { cursor: number; tasks: Task[]; runtime: Runtime }

export interface Envelope {
  event_id: string; source_kind: string; event_type: string; task_id: string | null; worker_id: string;
  sequence: number; timestamp: string; payload: Record<string, unknown>;
}

export interface EpisodeInfo { id: string; label: string; scenario: string; seed: number | null; ticks: number; topology_exported: boolean }
export interface NodeView { id: string; asset_type: string | null; zone: string | null; status: string; tags: string[] }
export interface Edge { src: string; dst: string; kind: string }
export interface FeedEntry {
  idx: number; tick: number; type: string; actor: string; action: string | null; target: string | null;
  outcome: string; duration: number | null; scope: string | null;
}
export interface Cumulative { reward: Record<string, number>; components: Record<string, Record<string, number>> }
export interface Frame { tick: number; nodes: NodeView[]; feed: FeedEntry[]; reward: Record<string, number>; cumulative: Cumulative }
export interface Replay {
  episode_id: string; label: string; scenario: string; topology_exported: boolean; perspective: Perspective;
  privileged: boolean; frames: Frame[]; edges?: Edge[]; final_metrics?: Record<string, number | boolean | null>; notice?: string;
}

export interface Caps { total_steps: number; timeout_s: number; grace_s: number; cpu: number; memory_mb: number; max_containers: number }
export interface Estimate { steps_per_sec: number | null; seconds: number | null; basis: "measured" | "unavailable"; measured_at: number | null; note: string }
export interface Resolved { preset: string; role: string; seed: number; train_config: Record<string, unknown>; caps: Caps; estimate: Estimate; warnings: string[] }
export interface Attempt {
  attempt_id: string; n: number; job_id: string | null; job_state: string | null; resume_from: string | null; resume_checkpoint: string | null;
  started_at: number | null; ended_at: number | null; error: string | null; cleaned_at: number | null;
}
export interface MetricRow { env_steps: number; mean_return: number | null; policy_version: number; [k: string]: unknown }
export interface Run {
  id: string; owner: string; preset: string; state: string; backend: string; resolved: Resolved; created_at: number; updated_at: number;
  error: string | null; heartbeat_age_s: number | null; stale: boolean; resumable: boolean; backend_error: string | null;
  artifact_problems: string[]; attempts: Attempt[]; latest_metric: MetricRow | null;
}
export interface Checkpoint {
  id: string; run_id: string; attempt_id: string; name: string; path: string; sha256: string; size: number; env_steps: number | null;
  policy_version: number | null; parent_id: string | null; role: string; meta: Record<string, unknown>; available: number;
}
export interface Pair { attacker: string; defender: string; metric: string; mean: number; ci_low: number; ci_high: number; n: number }
export interface Evaluation {
  id: string; run_id: string | null; suite: string; debug: boolean; label: string | null; state: string; error: string | null; actor: string;
  spec: { attacker_checkpoint_id: string; defender_checkpoint_id: string; episodes: number; frozen: boolean };
  result: { pairs: Pair[]; episode_ids: string[]; evaluator: string; note?: string } | null;
}
export interface ExpConfig {
  configured: boolean; backend: string; evaluator: string | null; presets: Record<string, { role: string; total_steps: number }>;
  ceilings: Record<string, number>; suites: string[]; max_eval_episodes: number; stale_after_s: number;
}

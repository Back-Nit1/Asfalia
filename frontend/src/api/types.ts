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

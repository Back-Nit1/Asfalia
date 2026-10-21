"""Data contracts for the development orchestrator: plan config, evidence, and the coding-agent adapter interface."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanConfig(BaseModel):
    max_concurrency: int = 3
    retry_cap: int = 2                # retries after the first attempt
    lease_seconds: float = 300
    task_timeout_seconds: float = 3600
    acceptance_timeout_seconds: float = 900
    integration_branch: str = "integration"
    integration_repo: str | None = None   # dedicated checkout of the integration branch; never a developer working copy
    combined_suite: list[str] = Field(default_factory=list)   # run after each merge, before it is kept
    repo: str = "."
    worktree_root: str = "../worktrees"
    base_commit: str = "base-v0"
    validate_paths: bool = False       # opt-in because some plans intentionally assign prospective paths


class TaskSpec(BaseModel):
    """One node of the fixed task DAG, as written in configs/orchestrator.yaml."""
    id: str
    role: str
    description: str
    owner: Literal["agent", "human"] = "agent"
    dependencies: list[str] = Field(default_factory=list)
    acceptance_commands: list[str] = Field(default_factory=list)
    assigned_paths: list[str] = Field(default_factory=list)
    prompt: str = ""


class TestResult(BaseModel):
    __test__ = False   # not a pytest class
    command: str
    exit_code: int
    duration_s: float
    stdout_tail: str = ""   # redacted
    stderr_tail: str = ""   # redacted


class Evidence(BaseModel):
    changed_files: list[str] = Field(default_factory=list)
    out_of_scope_files: list[str] = Field(default_factory=list)
    test_results: list[TestResult] = Field(default_factory=list)
    commits: list[str] = Field(default_factory=list)
    agent_summary: str = ""   # public final message of the agent; never reasoning


class AdapterCapabilities(BaseModel):
    name: str
    streaming: bool = True
    cancel: bool = True
    configured: bool
    detail: str = ""


class AgentEvent(BaseModel):
    """Public, observable agent activity (action summaries, tool outcomes). No chain-of-thought."""
    kind: Literal["message", "tool_use", "tool_result", "error", "stderr"]
    summary: str


class AgentResult(BaseModel):
    ok: bool
    summary: str = ""
    exit_code: int | None = None
    stderr_tail: str = ""   # redacted

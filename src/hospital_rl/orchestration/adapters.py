"""Coding-agent runtime adapters. The only real one drives the documented `claude -p` stream-json CLI."""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from .contracts import AdapterCapabilities, AgentEvent, AgentResult
from .events import clean_env, tail
from .tasks import Task

NOT_CONFIGURED = "coding runtime not configured"


class CodingAgentAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    @abstractmethod
    def submit(self, task: Task, cwd: Path) -> str:
        """Start work for `task` inside `cwd`; returns a process/session id."""

    @abstractmethod
    def drain_events(self, session_id: str) -> list[AgentEvent]:
        """Observable events produced since the last call."""

    @abstractmethod
    def result(self, session_id: str) -> AgentResult | None:
        """None while still running."""

    @abstractmethod
    def cancel(self, session_id: str) -> None: ...


class ClaudeCodeCliAdapter(CodingAgentAdapter):
    """`claude -p <prompt> --output-format stream-json --verbose`, one subprocess per task, run in the task worktree.
    Only public actions are surfaced: assistant text, tool names/inputs, tool results. Thinking blocks are dropped."""

    def __init__(self, binary: str | None = None, extra_args: list[str] | None = None):
        self.binary = binary or shutil.which("claude")
        self.extra_args = extra_args or []
        self._procs: dict[str, subprocess.Popen] = {}
        self._events: dict[str, queue.Queue[AgentEvent]] = {}
        self._result: dict[str, AgentResult] = {}
        self._stderr: dict[str, list[str]] = {}

    def capabilities(self) -> AdapterCapabilities:
        ok = self.binary is not None
        return AdapterCapabilities(name="claude-code-cli", configured=ok,
                                   detail="" if ok else f"{NOT_CONFIGURED}: `claude` binary not found on PATH")

    def submit(self, task: Task, cwd: Path) -> str:
        if not self.binary:
            raise RuntimeError(NOT_CONFIGURED)
        sid = str(uuid.uuid4())
        prompt = task.prompt or task.description
        cmd = [self.binary, "-p", prompt, "--output-format", "stream-json", "--verbose", *self.extra_args]
        proc = subprocess.Popen(cmd, cwd=cwd, env=clean_env() | _passthrough_auth(), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._procs[sid], self._events[sid], self._stderr[sid] = proc, queue.Queue(), []
        threading.Thread(target=self._pump_out, args=(sid, proc), daemon=True).start()
        threading.Thread(target=self._pump_err, args=(sid, proc), daemon=True).start()
        return sid

    def _pump_out(self, sid: str, proc: subprocess.Popen) -> None:
        final = ""
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            for ev in parse_stream_message(msg):
                self._events[sid].put(ev)
            if msg.get("type") == "result":
                final = msg.get("result") or ""
        rc = proc.wait()
        self._result[sid] = AgentResult(ok=rc == 0, summary=tail(final, 2000), exit_code=rc,
                                        stderr_tail=tail("".join(self._stderr[sid])))

    def _pump_err(self, sid: str, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            self._stderr[sid].append(line)

    def drain_events(self, session_id: str) -> list[AgentEvent]:
        q, out = self._events[session_id], []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    def result(self, session_id: str) -> AgentResult | None:
        return self._result.get(session_id)

    def cancel(self, session_id: str) -> None:
        proc = self._procs.get(session_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _passthrough_auth() -> dict[str, str]:
    """The CLI needs its own credentials; pass only those, never to acceptance commands."""
    import os
    return {k: os.environ[k] for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN") if k in os.environ}


def parse_stream_message(msg: dict) -> list[AgentEvent]:
    out: list[AgentEvent] = []
    content = (msg.get("message") or {}).get("content")
    if not isinstance(content, list):
        return out
    for block in content:
        t = block.get("type")
        if t == "text" and msg.get("type") == "assistant":
            out.append(AgentEvent(kind="message", summary=tail(block.get("text", ""), 500)))
        elif t == "tool_use":
            out.append(AgentEvent(kind="tool_use", summary=tail(f"{block.get('name')} {json.dumps(block.get('input', {}))}", 500)))
        elif t == "tool_result":
            body = block.get("content")
            body = body if isinstance(body, str) else json.dumps(body)
            out.append(AgentEvent(kind="error" if block.get("is_error") else "tool_result", summary=tail(body or "", 500)))
        # "thinking" / "redacted_thinking" blocks are intentionally never recorded
    return out


def default_adapter() -> CodingAgentAdapter | None:
    a = ClaudeCodeCliAdapter()
    return a if a.capabilities().configured else None

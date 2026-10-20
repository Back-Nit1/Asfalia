import stat
import time

from hospital_rl.orchestration.adapters import NOT_CONFIGURED, ClaudeCodeCliAdapter, parse_stream_message
from hospital_rl.orchestration.events import redact
from hospital_rl.orchestration.tasks import load_plan

from .conftest import mk


def test_repo_plan_loads_and_is_acyclic():
    cfg, tasks = load_plan("configs/orchestrator.yaml")
    assert cfg.retry_cap == 2 and cfg.max_concurrency == 3 and len(tasks) >= 5


def test_thinking_blocks_never_recorded():
    msg = {"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "secret reasoning"}, {"type": "text", "text": "editing file"},
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.py"}}]}}
    out = parse_stream_message(msg)
    assert [e.kind for e in out] == ["message", "tool_use"] and "secret reasoning" not in str(out)


def test_redaction(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "supersecretvalue123")
    text = "key=supersecretvalue123 Authorization: Bearer abcdefghijklmnopqrstuvwxyz sk-abcdefghijklmnopqrstu"
    r = redact(text)
    assert "supersecretvalue123" not in r and "abcdefghijklmnopqrstuvwxyz" not in r and "sk-abc" not in r


def test_missing_binary_is_reported_not_faked(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    a = ClaudeCodeCliAdapter()
    caps = a.capabilities()
    assert caps.configured is False and NOT_CONFIGURED in caps.detail


def test_cli_adapter_streams_a_real_subprocess(tmp_path):
    """TEST ONLY: a stand-in executable that speaks the stream-json shape; exercises the real process plumbing."""
    exe = tmp_path / "claude"
    exe.write_text("""#!/bin/sh
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"working"},{"type":"thinking","thinking":"hidden"}]}}'
echo '{"type":"result","subtype":"success","result":"all done"}'
echo "token=abcdefghijklmnop" >&2
""")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    a = ClaudeCodeCliAdapter(binary=str(exe))
    sid = a.submit(mk("t"), tmp_path)
    for _ in range(100):
        if a.result(sid):
            break
        time.sleep(0.05)
    res = a.result(sid)
    assert res and res.ok and res.summary == "all done" and "abcdefghijklmnop" not in res.stderr_tail
    assert [e.summary for e in a.drain_events(sid)] == ["working"]


def test_cancel_terminates_process(tmp_path):
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\nsleep 30\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    a = ClaudeCodeCliAdapter(binary=str(exe))
    sid = a.submit(mk("t"), tmp_path)
    a.cancel(sid)
    for _ in range(100):
        if a.result(sid):
            break
        time.sleep(0.05)
    assert a.result(sid) is not None and a.result(sid).ok is False

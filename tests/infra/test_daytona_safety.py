"""Static + dry-run guarantees: no .env reading, no shell interpolation of user input, dry run never touches Daytona."""
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fake_daytona import FakeSandboxClient

from hospital_rl.infra import cloud_school, daytona_backend, daytona_workspace
from hospital_rl.infra.plan import cloud_plan

ROOT = Path(__file__).resolve().parents[2]
CODE = [ROOT / "src/hospital_rl/infra" / f for f in ("daytona_backend.py", "daytona_workspace.py", "cloud_school.py")] + [ROOT / "scripts/daytona_smoke.py"]


def test_no_dotenv_reads_or_shell_true():
    for f in CODE:
        t = f.read_text()
        assert not re.search(r"dotenv|load_dotenv|open\([^)]*\.env|\.env[\"']", t.replace("no .env", "").replace("never reads .env", "")), f
        assert "shell=True" not in t and "os.system" not in t, f


def test_dry_run_resolves_without_contacting_daytona():
    code = ("import sys; from hospital_rl.infra.plan import cloud_plan; p = cloud_plan('daytona', 'configs/cloud_daytona.yaml', env={}); "
            "print(p.effective_backend, p.hard_caps['max_steps'], 'daytona' in sys.modules, 'torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.stdout.split()[:2] == ["local", "2000"] and r.stdout.split()[2] == "False", r.stdout + r.stderr


def test_smoke_script_without_opt_in_executes_nothing():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/daytona_smoke.py")], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "DAYTONA_API_KEY": "SECRETVALUE"})
    assert '"executed": "none"' in r.stdout and "NOT EXECUTED" in r.stdout and "SECRETVALUE" not in r.stdout + r.stderr


def test_setup_command_contains_only_constants():
    s = daytona_backend.DaytonaGateway._setup()
    assert "run" not in s.split("&&")[0] and "$" not in s and ";" not in s


def test_unsafe_ids_rejected_before_sandbox_creation(tmp_path):
    from helpers import spec
    cl = FakeSandboxClient(tmp_path / "b")
    cfg = daytona_backend.DaytonaConfig(mirror_dir=str(tmp_path / "m"))
    gw = daytona_backend.DaytonaGateway(cfg, cl)
    with pytest.raises(ValueError, match="unsafe id"):
        gw.spawn(spec(attempt="a1; rm -rf /"))
    assert cl.n == 0


def test_workspace_smoke_pinned_commit_with_evidence_and_cleanup(tmp_path):
    cl = FakeSandboxClient(tmp_path / "b")
    def run(sid, argv, t):
        if m := re.search(r"> (\S+) 2>&1", argv[-1]):
            cl.upload(sid, m.group(1), b"1 passed")
        return 0

    cl.run = run
    out = daytona_workspace.workspace_smoke(cl, ROOT, "HEAD", "tests/test_events.py", tmp_path / "ev")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert out["commit"] == head and out["exit_code"] == 0 and out["upload_verified"] and cl.deleted == ["sb-1"]
    assert Path(out["report"]).read_bytes() == b"1 passed"
    for bad in ("../x.py", "tests/a b.py", "src/x.py", "tests/x.py; id"):
        with pytest.raises(ValueError):
            daytona_workspace.workspace_smoke(cl, ROOT, "HEAD", bad, tmp_path / "ev")


def test_workspace_setup_failure_still_cleans_up(tmp_path):
    cl = FakeSandboxClient(tmp_path / "b")
    cl.run = lambda *a: 5
    with pytest.raises(RuntimeError, match="setup failed"):
        daytona_workspace.workspace_smoke(cl, ROOT, "HEAD", "tests/test_events.py", tmp_path / "ev")
    assert cl.deleted == ["sb-1"]

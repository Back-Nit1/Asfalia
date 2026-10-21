"""cloud_school's Modal path is just job specs on the common lifecycle (no second backend); Daytona code untouched."""
import argparse
from pathlib import Path

from helpers import ScriptedGateway

from hospital_rl.infra import cloud_school
from hospital_rl.infra.backend import JobState
from hospital_rl.infra.bundle import resolve, validate
from hospital_rl.infra.lifecycle import ResourceCaps
from hospital_rl.infra.modal_app import ModalBackend, ModalConfig


def args(**kw):
    return argparse.Namespace(**{"tag": "t1", "roles": "attacker,defender", "seeds": "7,8", "steps": 4096, "eval_every": 1024,
                                 "lr": 1e-3, "ent_coef": 0.01, "timeout": 600, **kw})


def test_school_fan_out_specs_validate_and_obey_caps(tmp_path):
    cfg = ModalConfig(caps=ResourceCaps(cpu=2, memory_mb=4096, timeout_s=600, grace_s=120, max_containers=4, max_steps=4096))
    be = ModalBackend(cfg, ScriptedGateway(tmp_path / "s"), tmp_path / "reg")
    specs = cloud_school.school_specs(args(), be)
    assert [s.run_id for s in specs] == ["t1-attacker-s7", "t1-attacker-s8", "t1-defender-s7", "t1-defender-s8"]
    for s in specs:
        validate(resolve(s))
        assert be.submit(s).startswith("modal:") and s.kind == "school"
    assert be.gw.spawned == 4


def test_rerunning_a_tag_resumes_from_the_previous_attempt(tmp_path):
    cfg = ModalConfig(caps=ResourceCaps(cpu=2, memory_mb=4096, timeout_s=600, grace_s=120, max_containers=4, max_steps=4096))
    gw = ScriptedGateway(tmp_path / "s", ("done", {"state": "drained", "env_steps": 1024}))
    be = ModalBackend(cfg, gw, tmp_path / "reg")
    first = cloud_school.school_specs(args(roles="defender", seeds="7"), be)[0]
    assert first.attempt_id == "a1" and first.resume_from_attempt is None
    assert be.status(be.submit(first)).state == JobState.DRAINED
    second = cloud_school.school_specs(args(roles="defender", seeds="7"), be)[0]
    assert (second.attempt_id, second.resume_from_attempt) == ("a2", "a1")


def test_daytona_only_import_stays_light():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", "import sys, hospital_rl.infra.cloud_school; print('torch' in sys.modules or 'modal' in sys.modules)"],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parents[2] / "src")})
    assert r.stdout.strip() == "False", r.stderr[-400:]

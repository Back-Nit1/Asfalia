import json

import pytest

from hospital_rl.evaluation import transfer
from hospital_rl.policies.checkpoint import IncompatibleCheckpointError
from hospital_rl.training.world import WorldConfigError, load_world, world_penv


def test_world_config_resolves_every_field(tmp_path):
    w = load_world("configs/experiments/smoke_world.yaml")
    penv = world_penv(w, "validation")
    penv.reset(seed=10_000)
    assert penv.spec.horizon == w.horizon == 20
    assert abs(len(penv.spec.nodes) - w.size) <= 3
    other = w.model_copy(update={"base_logging": 0})
    assert other.hash() != w.hash()
    p2 = world_penv(other, "validation"); p2.reset(seed=10_000)
    assert p2.spec.hash() != penv.spec.hash()          # base_logging reaches the generated scenario


def test_world_config_errors(tmp_path):
    with pytest.raises(WorldConfigError, match="not found"):
        load_world(tmp_path / "nope.yaml")
    bad = tmp_path / "w.yaml"
    bad.write_text("role: shared\nsize: 12\nmystery: 1\n")
    with pytest.raises(WorldConfigError, match="mystery"):
        load_world(bad)


def test_report_cross_play_real_actors(school_ckpts, world_path):
    r = transfer.report(school_ckpts["attacker"], school_ckpts["defender"], world_path, "validation", 2)
    cp = r["cross_play"]
    assert len(cp["attackers"]) == 2 and len(cp["defenders"]) == 4
    for metric in ("weighted_exposure_frac", "availability_violation_rate"):
        m = cp["matrices"][metric]
        assert (len(m["values"]), len(m["values"][0])) == (2, 4) and m["rows"] == "attacker"
    assert len(r["matchups"]) == 8 and r["frozen"] is True
    prov = r["provenance"]
    for k in ("world_config_hash", "split_hash", "catalog_hash", "code_revision", "eval_seeds", "actors"):
        assert k in prov
    learned = [a for a in prov["actors"].values() if a["kind"] == "checkpoint"]
    assert {a["checkpoint_id"] for a in learned} and all(a["config_hash"] for a in learned)
    json.dumps(r)                                      # machine-readable
    assert r == transfer.report(school_ckpts["attacker"], school_ckpts["defender"], world_path, "validation", 2)


def test_report_errors_and_exit_codes(school_ckpts, world_path, tmp_path, capsys):
    a, d = school_ckpts["attacker"], school_ckpts["defender"]
    with pytest.raises(transfer.CheckpointError, match="not found"):
        transfer.report(str(tmp_path / "no.pt"), d, world_path, "validation", 1)
    with pytest.raises(IncompatibleCheckpointError, match="role: checkpoint has 'defender', expected 'attacker'"):
        transfer.report(d, a, world_path, "validation", 1)       # roles swapped
    junk = tmp_path / "junk.pt"; junk.write_bytes(b"not a checkpoint")
    with pytest.raises(transfer.CheckpointError, match="unreadable"):
        transfer.report(str(junk), d, world_path, "validation", 1)
    base = ["--attacker-checkpoint", a, "--defender-checkpoint", d]
    assert transfer.main(base + ["--config", str(tmp_path / "missing.yaml")]) == 2
    assert transfer.main(["--attacker-checkpoint", d, "--defender-checkpoint", a, "--config", world_path]) == 2
    out = tmp_path / "r.json"
    assert transfer.main(base + ["--config", world_path, "--episodes", "1", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["stage"] == "D"

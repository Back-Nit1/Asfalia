import copy
import json
from pathlib import Path

import pytest

from hospital_rl.training import experiments as X
from hospital_rl.training.lineage import LineageError
from hospital_rl.training.schools import run_school

ROOT = Path(__file__).resolve().parents[2]


def fake(init="pretrained", adapt=100, pre=0, **cmp):
    return {"kind": "adapt", "identity": {"init": init}, "best_validation_score": -0.1, "validation": {},
            "checkpoints": {"best_path": f"/x/{init}{adapt}", "best": init},
            "budget": {"requested_adapt_steps": adapt, "adapt_env_steps": adapt, "pretrain_env_steps": pre,
                       "total_env_steps": adapt + pre},
            "opponent_frozen": {"unchanged": True},
            "comparability": {"trainable_role": "defender", "world_hash": "w", "split_hash": "s", "catalog_hash": "c",
                              "opponent_checkpoint_id": "o", "league": False, "val_episodes": 2, "seed": 1, **cmp}}


def test_compare_accepts_matched_budgets():
    assert X.compare([fake(pre=300), fake("scratch", 100)], "equal_adapt")["budget"] == 100
    assert X.compare([fake(pre=300), fake("scratch", 400)], "equal_total")["budget"] == 400


def test_compare_rejects_incomparable():
    with pytest.raises(X.ComparisonError, match="equal_adapt"):
        X.compare([fake(pre=300), fake("scratch", 400)], "equal_adapt")
    with pytest.raises(X.ComparisonError, match="equal_total"):
        X.compare([fake(pre=300), fake("scratch", 100)], "equal_total")
    with pytest.raises(X.ComparisonError, match="opponent_checkpoint_id"):
        X.compare([fake(), fake("scratch", opponent_checkpoint_id="other")], "equal_adapt")
    with pytest.raises(X.ComparisonError, match="world_hash"):
        X.compare([fake(), fake("scratch", world_hash="w2")], "equal_adapt")
    unfinished = fake("scratch"); unfinished["budget"]["adapt_env_steps"] = 50
    with pytest.raises(X.ComparisonError, match="unfinished"):
        X.compare([fake(), unfinished], "equal_adapt")
    moved = copy.deepcopy(fake("scratch")); moved["opponent_frozen"]["unchanged"] = False
    with pytest.raises(X.ComparisonError, match="frozen"):
        X.compare([fake(), moved], "equal_adapt")
    with pytest.raises(X.ComparisonError, match="two runs"):
        X.compare([fake()], "equal_adapt")


def test_experiment_config_validation(tmp_path):
    for name in ("smoke", "pretrained_adapt", "scratch_adapt", "equal_total"):
        X.load_experiment(str(ROOT / f"configs/experiments/{name}.yaml"))
    bad = tmp_path / "b.yaml"
    bad.write_text("name: x\nworld_config: w\nseed: 1\ntrainable_role: defender\ntransfer: {split: test}\n")
    with pytest.raises(ValueError, match="invalid experiment config"):
        X.load_experiment(str(bad))                       # test split is not allowed; neither school nor checkpoints
    assert X.main(["run", "--config", str(tmp_path / "none.yaml"), "--out", str(tmp_path / "o")]) == 2


def test_school_run_dir_lineage(tmp_path):
    kw = dict(eval_every=64, n_val=1, n_envs=2, rollout_len=16, log=lambda s: None)
    cfg = str(ROOT / "configs/defender_school.yaml")
    run_school("defender", cfg, 64, 1, tmp_path / "s", **kw)
    with pytest.raises(LineageError, match="seed"):
        run_school("defender", cfg, 64, 2, tmp_path / "s", **kw)
    with pytest.raises(LineageError):
        run_school("attacker", str(ROOT / "configs/attacker_school.yaml"), 64, 1, tmp_path / "s", **kw)


@pytest.mark.slow
def test_real_hospital_smoke_pipeline(tmp_path):
    """Both roles trained, frozen Stage D transfer, adaptation arms, reports + comparison manifest, real envs."""
    cfg = X.load_experiment(str(ROOT / "configs/experiments/smoke.yaml"))
    res = X.run_experiment(cfg, tmp_path / "exp", log=lambda s: None)
    out = tmp_path / "exp"
    assert (out / "school/attacker/best.pt").exists() and (out / "school/defender/best.pt").exists()
    before = json.loads((out / "stage_d/transfer_before_adaptation.json").read_text())
    assert before["frozen"] and "weighted_exposure_frac" in before["cross_play"]["matrices"]
    assert (out / "stage_d/transfer_after_adaptation.json").exists()
    arms = {a: json.loads(Path(p).read_text()) for a, p in res["arms"].items()}
    assert set(arms) == set(X.ARMS)
    sch = res["school_env_steps"]["defender"]
    assert arms["pretrained"]["budget"]["total_env_steps"] == sch + 128
    assert arms["scratch_same_adapt"]["budget"]["adapt_env_steps"] == 128
    assert arms["scratch_equal_total"]["budget"]["total_env_steps"] == sch + 128      # equal total, no pretraining
    assert [c["rule"] for c in res["comparisons"]] == ["equal_adapt", "equal_total"]
    assert all("rejected" not in c for c in res["comparisons"])
    assert (out / "comparison_manifest.json").exists()
    assert X.main(["compare", "--rule", "equal_total", str(out / "arms/pretrained"), str(out / "arms/scratch_same_adapt")]) == 2
    assert X.main(["compare", "--rule", "equal_total", str(out / "arms/pretrained"), str(out / "arms/scratch_equal_total")]) == 0
    # rerun is a resume of the same lineage, not an overwrite
    assert X.run_experiment(cfg, out, log=lambda s: None)["arms"] == res["arms"]
    with pytest.raises(LineageError):
        X.run_experiment(cfg.model_copy(update={"seed": 99}), out, log=lambda s: None)

from pathlib import Path

import pytest

from hospital_rl.evaluation import ablations as A
from hospital_rl.training.experiments import load_experiment

ROOT = Path(__file__).resolve().parents[2]
BASE = load_experiment(str(ROOT / "configs/experiments/ablation_full.yaml"))
RUNNABLE = [n for n, a in A.ABLATIONS.items() if not a.request]
BLOCKED = [n for n, a in A.ABLATIONS.items() if a.request]


def test_required_ablations_all_defined():
    need = {"no_pretraining", "no_adaptation", "no_workflow_cost", "no_domain_randomization", "no_opponent_league",
            "full_observability_diagnostic"}
    assert need <= set(A.ABLATIONS) and {"telemetry_loss_30", "telemetry_loss_50"} <= set(A.ABLATIONS)


@pytest.mark.parametrize("name", RUNNABLE)
def test_runnable_variants_are_valid_configs_and_checked_in(name):
    v = A.make_config(BASE, name)
    A.check_pair(BASE, v, name)
    disk = load_experiment(str(ROOT / f"configs/experiments/ablations/{name}.yaml"))   # checked-in == generated
    assert disk == v


@pytest.mark.parametrize("name", BLOCKED)
def test_engine_dependent_ablations_refuse_instead_of_faking(name):
    with pytest.raises(A.Unavailable, match="agent-F.md"):
        A.make_config(BASE, name)
    with pytest.raises(A.Unavailable):
        A.check_pair(BASE, BASE, name)


def test_invalid_comparisons_rejected():
    v = A.make_config(BASE, "no_opponent_league")
    with pytest.raises(A.AblationError, match="only"):
        A.check_pair(BASE, v.model_copy(update={"seed": 8}), "no_opponent_league")          # second factor changed
    with pytest.raises(A.AblationError, match="must set"):
        A.check_pair(BASE, BASE.model_copy(update={"name": "x"}), "no_opponent_league")     # nothing ablated
    with pytest.raises(A.AblationError, match="only"):
        A.check_pair(BASE, v.model_copy(update={"world_config": "configs/tiny.yaml"}), "no_opponent_league")
    with pytest.raises(A.AblationError):
        A.check_pair(BASE, v, "no_pretraining")                                              # wrong ablation label
    with pytest.raises(A.AblationError, match="unknown"):
        A.check_pair(BASE, v, "bogus")


def test_test_stratum_rejected_in_ablation_pair():
    full = BASE.model_copy(update={"transfer": BASE.transfer.model_copy(update={"split": "test"})})
    v = A.make_config(BASE, "no_opponent_league").model_copy(update={"transfer": full.transfer})
    with pytest.raises(A.AblationError, match="test"):
        A.check_pair(full, v, "no_opponent_league")


def test_stored_run_comparability():
    comp = {"trainable_role": "defender", "world_hash": "w", "split_hash": "s", "catalog_hash": "c", "league": True,
            "val_episodes": 2, "seed": 1, "opponent_checkpoint_id": "o"}
    a = {"comparability": comp, "code_revision": "r"}
    b = {"comparability": comp | {"league": False}, "code_revision": "r"}
    A.check_runs(a, b, "no_opponent_league")
    with pytest.raises(A.AblationError, match="nothing was ablated"):
        A.check_runs(a, a, "no_opponent_league")
    with pytest.raises(A.AblationError, match="world_hash"):
        A.check_runs(a, {**b, "comparability": b["comparability"] | {"world_hash": "z"}}, "no_opponent_league")
    with pytest.raises(A.AblationError, match="code_revision"):
        A.check_runs(a, {**b, "code_revision": "q"}, "no_opponent_league")


def test_cli(tmp_path):
    assert A.main(["list"]) == 0
    assert A.main(["write", "--base", str(ROOT / "configs/experiments/ablation_full.yaml"), "--out", str(tmp_path)]) == 0
    assert A.main(["check", str(ROOT / "configs/experiments/ablation_full.yaml"), str(tmp_path / "no_opponent_league.yaml"),
                   "--ablation", "no_opponent_league"]) == 0
    assert A.main(["check", str(ROOT / "configs/experiments/ablation_full.yaml"), str(tmp_path / "no_adaptation.yaml"),
                   "--ablation", "no_opponent_league"]) == 2
    assert A.main(["check", "a", "b", "--ablation", "telemetry_loss_30"]) == 2

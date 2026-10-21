import pathlib

import pytest
import yaml

from hospital_rl.evaluation import strata as S

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHANGES = {"topology_families": ("zzz",), "workload_profiles": ("zzz",), "attack_schedule": "other",
           "vulnerability_difficulty": (0.1,), "telemetry_loss": 0.5, "visibility_mode": "full_diagnostic",
           "seed_start": 5, "seed_stop": 6}


def test_default_definition_valid_and_disjoint():
    S.DEFINITION.validate()
    assert S.DEFINITION.get("test").role == "report_only"


@pytest.mark.parametrize("name", ["train", "validation", "test"])
@pytest.mark.parametrize("field", sorted(CHANGES))
def test_every_stratum_change_changes_hash(name, field):
    d = S.DEFINITION
    s = d.get(name).model_copy(update={field: CHANGES[field]})
    d2 = d.model_copy(update={"strata": tuple(s if x.name == name else x for x in d.strata)})
    assert d2.hash() != d.hash() and d2.stratum_hash(name) != d.stratum_hash(name)
    assert all(d2.stratum_hash(o) == d.stratum_hash(o) for o in ("train", "validation", "test") if o != name)


def test_overlap_with_test_rejected():
    d = S.DEFINITION
    tr = d.get("train").model_copy(update={"seed_stop": 20_001})
    with pytest.raises(S.StratumError, match="seeds"):
        d.model_copy(update={"strata": (tr, d.get("validation"), d.get("test"))}).validate()


def test_test_stratum_cannot_enter_training():
    assert S.require_trainable("train") == "train" and S.require_trainable("validation")
    for use in ("training", "curriculum", "opponent selection"):
        with pytest.raises(S.StratumError, match="report-only"):
            S.require_trainable("test", use)


def test_checked_in_configs_never_name_test():
    for f in list((ROOT / "configs").rglob("*.yaml")):
        S.check_training_config(yaml.safe_load(f.read_text()), f.name)
    with pytest.raises(S.StratumError):
        S.check_training_config({"school": {"levels": [{"eval_split": "test"}]}})


def test_unsupported_dimensions_flagged_not_runnable():
    s = S.DEFINITION.get("train").model_copy(update={"telemetry_loss": 0.3})
    assert s.runnable() == ["telemetry_loss=0.3"] and not S.DEFINITION.get("train").runnable()

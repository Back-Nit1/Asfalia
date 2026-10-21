import argparse

import pytest

from hospital_rl import cli


@pytest.mark.parametrize("command, expected", [
    ("adapt", "--trainable-role"),
    ("benchmark-inference", "--checkpoint"),
    ("export-trajectories", "--attacker-checkpoint"),
    ("cloud-reconcile", "--backend"),
])
def test_delegated_help_reaches_real_parser(command, expected, capsys):
    with pytest.raises(SystemExit) as ex:
        cli.main([command, "--help"])
    assert ex.value.code == 0
    assert expected in capsys.readouterr().out


def test_benchmark_config_fully_controls_world_settings(tmp_path, monkeypatch):
    config = tmp_path / "world.yaml"
    config.write_text("role: shared\nsize: 17\nhorizon: 23\ndifficulty: 0.25\nbase_logging: 2\n")
    seen = {}

    def fake_run(split, episodes, **settings):
        seen.update(split=split, episodes=episodes, **settings)
        return {"split": split, "split_hash": "hash", "n_scenarios": episodes, "defenders": {}}

    from hospital_rl.evaluation import benchmark
    monkeypatch.setattr(benchmark, "run", fake_run)
    args = argparse.Namespace(config=str(config), split="validation", episodes=3, size=8, out=None,
                              tie_sensitivity=False)
    assert cli._bench(args) == 0
    assert seen == {"split": "validation", "episodes": 3, "size": 17, "horizon": 23,
                    "difficulty": 0.25, "base_logging": 2}

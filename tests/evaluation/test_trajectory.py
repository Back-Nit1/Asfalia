import json
import shutil

import numpy as np
import pytest

from hospital_rl.data import trajectory as T
from hospital_rl.evaluation.transfer import evaluate_pair, scripted_actor
from hospital_rl.training.world import WorldConfig, world_penv

WORLD = WorldConfig(size=12, horizon=10, difficulty=0.5)


def record(out, att="ScriptedAttacker", dfn="GreedyDefender", n=2, split="validation"):
    w = T.TrajectoryWriter(out, WORLD.hash(), split)
    r = evaluate_pair(scripted_actor(att), scripted_actor(dfn), WORLD, split, n, sink=w)
    return r, w


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    d = tmp_path_factory.mktemp("traj")
    record(d)
    return d


def test_export_validates_roundtrips_and_is_deterministic(good, tmp_path):
    out = T.validate_dir(good)
    assert [o["steps"] for o in out] == [WORLD.horizon] * 2
    man = json.loads(next(good.glob("*.manifest.json")).read_text())
    rows = T.load_rows(good / man["files"]["actor"]["path"])
    assert rows[0]["observation"]["node_features"].dtype == np.float32 and rows[0]["action_mask"].dtype == np.int8
    assert rows[0]["action_mask"][rows[0]["action"]] == 1
    record(tmp_path)                                               # same inputs -> byte-identical artifacts
    assert all((good / p.name).read_bytes() == p.read_bytes() for p in tmp_path.iterdir())


def test_actor_file_has_no_privileged_content(good):
    txt = next(good.glob("*.jsonl")).read_text()
    assert "truth" not in txt and "implant" not in txt and "PRIVILEGED" not in txt
    pv = json.loads(next(good.glob("*.privileged.json")).read_text())
    assert pv["PRIVILEGED"] and "observation" not in pv


def _tamper(good, tmp_path, fn, target="actor", retag=True):
    d = tmp_path / "t"
    shutil.copytree(good, d)
    mp = sorted(d.glob("*.manifest.json"))[0]
    m = json.loads(mp.read_text())
    p = d / m["files"][target]["path"]
    if target == "actor":
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        fn(rows)
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    else:
        pv = json.loads(p.read_text()); fn(pv); p.write_text(json.dumps(pv, sort_keys=True))
    if retag:                                                      # attacker who also fixes the manifest hash
        m["files"][target]["sha256"] = T._sha(p)
        mp.write_text(json.dumps(m))
    return mp


def _inject_top(rows): rows[0]["implant"] = [1]
def _inject_nested(rows): rows[1]["observation"]["implant_map"] = 1; rows[1]["observation"]["nodes"] = {"truth": 1}
def _illegal(rows): rows[0]["action_mask"][rows[0]["action"]] = 0
def _shape(rows): rows[0]["observation"]["globals"] = rows[0]["observation"]["globals"][:-1]
def _dtype(rows): rows[0]["observation"]["node_mask"][0] = 0.5
def _nan(rows): rows[0]["observation"]["globals"][0] = 1e40
def _order(rows): rows[0], rows[1] = rows[1], rows[0]
def _scen(rows): rows[3]["scenario_hash"] = "x"
def _ckpt(rows): rows[4]["checkpoint_id"] = "other"
def _schema(rows): rows[0]["observation_schema_hash"] = "bad"
def _trunc(rows): rows[0]["truncated"] = True
def _pad(rows): rows[0]["observation"]["node_features"][-1][0] = 1.0


@pytest.mark.parametrize("fn,msg", [
    (_inject_top, "keys differ"), (_inject_nested, "evaluator-only"), (_illegal, "not legal"),
    (_shape, "shape"), (_dtype, "bad JSON dtype"), (_nan, "float32"), (_order, "ordering"), (_scen, "mid-episode"),
    (_ckpt, "identity changed"), (_schema, "schema hash"), (_trunc, "termination"), (_pad, "padding")])
def test_actor_tampering_rejected(good, tmp_path, fn, msg):
    with pytest.raises(T.TrajectoryError, match=msg):
        T.validate(_tamper(good, tmp_path, fn))


def test_hash_catches_unretagged_edit(good, tmp_path):
    with pytest.raises(T.TrajectoryError, match="hash mismatch"):
        T.validate(_tamper(good, tmp_path, _inject_top, retag=False))


def test_privileged_cross_file_keys(good, tmp_path):
    with pytest.raises(T.TrajectoryError, match="keys do not match"):
        T.validate(_tamper(good, tmp_path, lambda pv: pv["keys"].reverse(), target="privileged"))
    with pytest.raises(T.TrajectoryError, match="actor input"):
        T.validate(_tamper(good, tmp_path / "b", lambda pv: pv.update(observation={}), target="privileged"))


def test_orphan_file_rejected(good, tmp_path):
    shutil.copytree(good, tmp_path / "o")
    (tmp_path / "o" / "stray.jsonl").write_text("{}")
    with pytest.raises(T.TrajectoryError, match="without a manifest"):
        T.validate_dir(tmp_path / "o")


class _Wait:
    id, provenance = "wait", {}

    def reset(self): pass
    def bind(self, penv, role): pass
    def act(self, obs): return 0


def _waiting_rows(tmp_path, perturb):
    penv = world_penv(WORLD, "validation")
    orig_reset = penv.reset

    def reset(seed=None, options=None):
        out = orig_reset(seed=seed, options=options)
        if perturb:                                    # hidden truth only: no observable evidence is emitted
            for ns in penv.world.st.nodes.values():
                ns.implant = True
                ns.taken |= {"hidden-record"}
        return out
    penv.reset = reset
    T.TrajectoryWriter(tmp_path, WORLD.hash(), "validation")(penv, _Wait(), _Wait(), 10_001)
    return T.load_rows(next(tmp_path.glob("*.jsonl")))


def test_hidden_state_perturbation_does_not_change_actor_rows(tmp_path):
    a, b = _waiting_rows(tmp_path / "a", False), _waiting_rows(tmp_path / "b", True)
    assert len(a) == len(b) == 2 * WORLD.horizon
    for x, y in zip(a, b):
        assert np.array_equal(x["action_mask"], y["action_mask"])
        assert all(np.array_equal(x["observation"][k], y["observation"][k]) for k in x["observation"])


def test_cli_export_and_validate_exit_codes(tmp_path):
    assert T.main(["validate", str(tmp_path)]) == 2                     # nothing to validate
    assert T.main(["export", "--attacker-checkpoint", "nope.pt", "--defender-checkpoint", "nope.pt",
                   "--out", str(tmp_path / "x")]) == 2

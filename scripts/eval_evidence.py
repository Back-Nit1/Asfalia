"""Deterministic evidence for the evaluation/data/reporting contract. Run: PYTHONPATH=src python scripts/eval_evidence.py
Writes docs/img/report_fixture/* (report from checked fixtures) and docs/img/trajectory_validation.json."""
import hashlib
import json
import pathlib
import shutil
import tempfile

from hospital_rl.data import trajectory as T
from hospital_rl.evaluation.strata import DEFINITION, benchmark_hash
from hospital_rl.evaluation.transfer import evaluate_pair, scripted_actor
from hospital_rl.reporting import report
from hospital_rl.training.world import WorldConfig

OUT = pathlib.Path("docs/img")
FIX = pathlib.Path("tests/fixtures/experiments")


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def main():
    rep = report.build([FIX / "smoke_seed1", FIX / "smoke_seed2"])
    shutil.rmtree(OUT / "report_fixture", ignore_errors=True)
    files = report.write(rep, OUT / "report_fixture")
    world = WorldConfig(size=12, horizon=10, difficulty=0.5)
    with tempfile.TemporaryDirectory() as d:
        w = T.TrajectoryWriter(d, world.hash(), "validation")
        evaluate_pair(scripted_actor("ScriptedAttacker"), scripted_actor("GreedyDefender"), world, "validation", 3, sink=w)
        ok = T.validate_dir(d)
        man = sorted(pathlib.Path(d).glob("*.manifest.json"))[0]
        rows = [json.loads(x) for x in (pathlib.Path(d) / json.loads(man.read_text())["files"]["actor"]["path"]).read_text().splitlines()]
        rows[0]["implant"] = [1]                                         # deliberate privileged-field injection
        actor = pathlib.Path(d) / json.loads(man.read_text())["files"]["actor"]["path"]
        actor.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        m = json.loads(man.read_text()); m["files"]["actor"]["sha256"] = T._sha(actor); man.write_text(json.dumps(m))
        try:
            T.validate(man); injected = "ACCEPTED (BUG)"
        except T.TrajectoryError as e:
            injected = f"rejected: {e}"
        sha_ep = {p.name.split(".")[0] + "." + p.name.split(".", 1)[1]: sha(p) for p in sorted(pathlib.Path(d).iterdir())
                  if not p.name.startswith(man.name.split(".")[0])}
    (OUT / "trajectory_validation.json").write_text(json.dumps({
        "trajectory_version": T.TRAJECTORY_VERSION, "benchmark_hash": benchmark_hash(),
        "strata": {s.name: {"hash": DEFINITION.stratum_hash(s.name), "role": s.role} for s in DEFINITION.strata},
        "episodes_validated": ok, "privileged_injection": injected,
        "scripted_pair": "ScriptedAttacker x GreedyDefender", "world": world.model_dump(),
        "artifact_sha256_untouched_episodes": sha_ep}, indent=1, sort_keys=True) + "\n")
    print("report files:", files, "\ntrajectory validation ->", OUT / "trajectory_validation.json")


if __name__ == "__main__":
    main()

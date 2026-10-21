import json
import shutil
from pathlib import Path

import pytest

from hospital_rl.reporting import report as R

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "experiments"
RUNS = [FIX / "smoke_seed1", FIX / "smoke_seed2"]


def _files(d):
    return {p.name: p.read_bytes() for p in sorted(Path(d).iterdir())}


def test_report_is_deterministic_and_labels_insufficient(tmp_path):
    for n in ("a", "b"):
        assert R.main(["--runs", *map(str, RUNS), "--out", str(tmp_path / n)]) == 0
    assert _files(tmp_path / "a") == _files(tmp_path / "b")
    rep = json.loads((tmp_path / "a" / "report.json").read_text())
    assert rep["report_version"] == R.REPORT_VERSION and rep["seeds"] == [1, 2]
    assert rep["evidence"].startswith("INSUFFICIENT: 2 seeds")
    assert {"security.svg", "availability.svg", "cost.svg", "arms_cost.svg", "report.md"} <= set(_files(tmp_path / "a"))
    assert "INSUFFICIENT" in (tmp_path / "a" / "report.md").read_text()
    assert "/Users" not in (tmp_path / "a" / "report.json").read_text()      # no absolute paths -> portable


def test_per_seed_values_match_stored_outputs():
    rep = R.build(RUNS)
    arm = rep["adaptation_arms"]["pretrained"]["metrics"]["spend"]
    stored = [json.loads((r / "arms/pretrained/adapt_manifest.json").read_text())["validation"]["spend"] for r in RUNS]
    assert list(arm["per_seed"].values()) == stored and arm["n"] == 2 and arm["ci95"] is not None
    cp = rep["stage_d_before_adaptation"]["cross_play"]["weighted_exposure_frac"]
    assert "learned_attacker" in cp["rows"] and "learned_defender" in cp["cols"]


def test_single_seed_has_no_spread_or_ci_and_says_so():
    rep = R.build(RUNS[:1])
    assert rep["evidence"].startswith("INSUFFICIENT: one-seed")
    s = rep["adaptation_arms"]["pretrained"]["metrics"]["spend"]
    assert s["std"] is None and s["ci95"] is None


def test_stats_ci_matches_hand_computation():
    s = R.stats([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["std"] == 1.0 and s["ci95"][1] == pytest.approx(2 + 4.303 / 3 ** 0.5)
    assert R.stats([])["mean"] is None


def _copy(tmp_path, edit):
    d = tmp_path / "bad"
    shutil.copytree(RUNS[1], d)
    edit(d)
    return d


def _edit_json(path, fn):
    j = json.loads(path.read_text()); fn(j); path.write_text(json.dumps(j))


def test_incompatible_runs_rejected(tmp_path):
    bad = _copy(tmp_path, lambda d: _edit_json(d / "arms/pretrained/adapt_manifest.json",
                                              lambda j: j["comparability"].update(world_hash="zzz")))
    with pytest.raises(R.ReportError, match="world_hash"):
        R.build([RUNS[0], bad])
    assert R.main(["--runs", str(RUNS[0]), str(bad), "--out", str(tmp_path / "o")]) == 2
    bad2 = _copy(tmp_path / "x", lambda d: _edit_json(d / "stage_d/transfer_before_adaptation.json",
                                                     lambda j: j["provenance"].update(split_hash="zzz")))
    with pytest.raises(R.ReportError, match="split_hash"):
        R.build([RUNS[0], bad2])
    with pytest.raises(R.ReportError, match="duplicate"):
        R.build([RUNS[0], RUNS[0]])


def test_missing_arm_is_reported_missing_never_imputed(tmp_path):
    bad = _copy(tmp_path, lambda d: _edit_json(d / "comparison_manifest.json",
                                              lambda j: j["arms"].pop("scratch_equal_total")))
    rep = R.build([RUNS[0], bad])
    row = rep["adaptation_arms"]["scratch_equal_total"]
    assert row["seeds"] == [1] and row["missing_seeds"] == [2] and row["metrics"]["spend"]["n"] == 1
    assert "MISSING for seeds [2]" in R.render_md(rep)


def test_missing_input_exits_2(tmp_path):
    assert R.main(["--runs", str(tmp_path), "--out", str(tmp_path / "o")]) == 2

"""Versioned, deterministic report from *stored* experiment outputs (``training.experiments run`` directories).

One run directory = one seed of one experiment. Nothing is simulated or filled in here: a missing arm or stage stays
missing and is listed as such; runs whose comparability hashes differ are rejected, never averaged. The output has no
timestamps or absolute paths, so the same inputs always give byte-identical files.

    python -m hospital_rl.reporting.report --runs RUN_DIR [RUN_DIR ...] --out OUT_DIR

Exit codes: 0 report written, 2 unusable/incompatible input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPORT_VERSION = 1
MIN_SEEDS = 5                         # below this the report says INSUFFICIENT; it still prints per-seed values
ARMS = ("pretrained", "scratch_same_adapt", "scratch_equal_total")
METRICS = {"security": "weighted_exposure_frac", "availability": "critical_success", "cost": "spend"}
LOWER_IS_BETTER = {"weighted_exposure_frac": True, "critical_success": False, "spend": True}
COMPAT_KEYS = ("trainable_role", "world_hash", "split_hash", "catalog_hash", "league", "val_episodes")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


class ReportError(ValueError):
    pass


def stats(vals: list[float]) -> dict:
    """Mean, sample std, range and a t-based 95% CI. n < 2 has no spread and no CI (null, not 0)."""
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "ci95": None}
    m = sum(vals) / n
    if n == 1:
        return {"n": 1, "mean": m, "std": None, "min": m, "max": m, "ci95": None}
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    t = T975[max(k for k in T975 if k <= n - 1)] if n - 1 < 30 else 1.96
    h = t * sd / math.sqrt(n)
    return {"n": n, "mean": m, "std": sd, "min": min(vals), "max": max(vals), "ci95": [m - h, m + h]}


def _load(p: Path) -> dict:
    if not p.is_file():
        raise ReportError(f"missing file: {p}")
    return json.loads(p.read_text())


def load_run(d: str | Path) -> dict:
    d = Path(d)
    man = _load(d / "comparison_manifest.json")
    run = {"dir": d, "name": man["name"], "code_revision": man["code_revision"], "manifest": man,
           "before": _load(d / "stage_d" / "transfer_before_adaptation.json"), "after": None, "arms": {}}
    if (d / "stage_d" / "transfer_after_adaptation.json").is_file():
        run["after"] = _load(d / "stage_d" / "transfer_after_adaptation.json")
    for a in ARMS:
        if a in man["arms"]:
            run["arms"][a] = _load(d / "arms" / a / "adapt_manifest.json")
    seeds = {m["identity"]["seed"] for m in run["arms"].values()}
    if len(seeds) > 1:
        raise ReportError(f"{d}: arms disagree on seed {sorted(seeds)}")
    run["seed"] = seeds.pop() if seeds else None
    run["digest"] = hashlib.sha256(json.dumps([man, run["before"], run["after"], run["arms"]], sort_keys=True)
                                   .encode()).hexdigest()[:16]
    return run


def _compat(run: dict) -> dict:
    c = {"name": run["name"]}
    for a, m in run["arms"].items():
        c[f"arm:{a}"] = {k: m["comparability"][k] for k in COMPAT_KEYS}
    p = run["before"]["provenance"]
    c["stage_d"] = {k: p[k] for k in ("world_config_hash", "split", "split_hash", "catalog_hash",
                                      "observation_schema_hash", "episodes")}
    if run["after"]:
        pa = run["after"]["provenance"]
        c["stage_d_after_matches_before"] = all(pa[k] == p[k] for k in c["stage_d"])
    return c


def check_compat(runs: list[dict]) -> dict:
    """Same experiment, same world/split/catalog/schema everywhere; otherwise refuse (never average mismatches).
    A run that lacks an arm is allowed (reported as MISSING); runs that have it must agree."""
    if not runs:
        raise ReportError("no runs given")
    bad, seen = [], {}
    for r in runs:
        c = _compat(r)
        if c.pop("stage_d_after_matches_before", True) is False:
            bad.append(f"seed {r['seed']}: stage D after-adaptation report differs in provenance from before")
        for k, v in c.items():
            if seen.setdefault(k, v) != v:
                bad.append(f"seed {r['seed']}: {k} {v!r} != {seen[k]!r}")
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        bad.append(f"duplicate seeds {seeds}")
    if len({r["code_revision"] for r in runs}) > 1:
        bad.append("runs come from different code revisions: " + ", ".join(sorted({r['code_revision'][:10] for r in runs})))
    if bad:
        raise ReportError("incompatible runs, refusing to aggregate: " + "; ".join(bad))
    return {"ok": True, "checked": seen}


def alias(actor_id: str) -> str:
    """Learned actors are `role:checkpoint_id` (differs per seed); aggregate them under `learned_<role>`."""
    return f"learned_{actor_id.split(':')[0]}" if ":" in actor_id else actor_id


def _pairs(stage: dict) -> dict[tuple[str, str], dict]:
    return {(alias(m["attacker"]), alias(m["defender"])): m["overall"] for m in stage["matchups"]}


def _stage_section(runs: list[dict], key: str) -> dict | None:
    have = [r for r in runs if r[key]]
    if not have:
        return None
    per = {r["seed"]: _pairs(r[key]) for r in have}
    keys = sorted(set.intersection(*[set(p) for p in per.values()]))
    sec = {"seeds": sorted(per), "missing_seeds": sorted(r["seed"] for r in runs if not r[key]),
           "episodes_per_pair": have[0][key]["episodes"], "split": have[0][key]["split"], "pairs": {}}
    for a, d in keys:
        row = {}
        for m in (*METRICS.values(), "availability_violation_rate", "contained_rate"):
            vals = {s: (p[(a, d)][m]["mean"] if isinstance(p[(a, d)][m], dict) else p[(a, d)][m]) for s, p in per.items()}
            row[m] = {"per_seed": {str(s): v for s, v in sorted(vals.items())}, **stats(list(vals.values()))}
        sec["pairs"][f"{a} x {d}"] = row
    sec["cross_play"] = {}
    for m in METRICS.values():                                       # attacker rows x defender cols, mean over seeds
        atts, dfns = sorted({a for a, _ in keys}), sorted({d for _, d in keys})
        sec["cross_play"][m] = {"rows": atts, "cols": dfns, "values": [
            [sec["pairs"][f"{a} x {d}"][m]["mean"] if (a, d) in keys else None for d in dfns] for a in atts]}
    return sec


def _arm_section(runs: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        have = [r for r in runs if arm in r["arms"]]
        if not have:
            continue
        ms = {r["seed"]: r["arms"][arm] for r in have}
        row = {"seeds": sorted(ms), "missing_seeds": sorted(r["seed"] for r in runs if arm not in r["arms"]),
               "budget": next(iter(ms.values()))["budget"], "metrics": {}}
        for m in (*METRICS.values(), "success", "score"):
            src = "exposure" if m == "weighted_exposure_frac" else "critical_success" if m == "critical_success" else m
            vals = {s: v["validation"].get(src) for s, v in ms.items()}
            vals = {s: x for s, x in vals.items() if x is not None}
            row["metrics"][m] = {"per_seed": {str(s): v for s, v in sorted(vals.items())}, **stats(list(vals.values()))}
        out[arm] = row
    return out


def build(run_dirs: list[str | Path]) -> dict:
    runs = sorted((load_run(d) for d in run_dirs), key=lambda r: (r["seed"] is None, r["seed"]))
    compat = check_compat(runs)
    n = len(runs)
    rejected = [{"seed": r["seed"], "pair": c["pair"], "reason": c["rejected"]}
                for r in runs for c in r["manifest"]["comparisons"] if "rejected" in c]
    return {
        "report_version": REPORT_VERSION, "experiment": runs[0]["name"], "n_seeds": n, "seeds": [r["seed"] for r in runs],
        "inputs": {str(r["seed"]): r["digest"] for r in runs}, "code_revision": runs[0]["code_revision"],
        "evidence": ("INSUFFICIENT: one-seed smoke output; per-seed values only, no spread or CI exists"
                     if n == 1 else
                     f"INSUFFICIENT: {n} seeds < {MIN_SEEDS}; spread and CI are shown but are not a basis for claims"
                     if n < MIN_SEEDS else f"{n} seeds; uncalibrated synthetic simulator, descriptive comparison only"),
        "compatibility": compat, "rejected_comparisons": rejected,
        "adaptation_arms": _arm_section(runs),
        "stage_d_before_adaptation": _stage_section(runs, "before"),
        "stage_d_after_adaptation": _stage_section(runs, "after"),
        "notes": ["Rewards are uncalibrated; scripted opponents and one synthetic world family set are all that is tested.",
                  "Arms were compared only where Agent C's budget-rule compare() accepted them (see rejected_comparisons).",
                  "Cross-play cells with a learned actor aggregate different per-seed checkpoints under one label."]}


# ------------------------------------------------------------------ rendering
def _f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def _cell(s: dict) -> str:
    ci = "" if s["ci95"] is None else f" [{_f(s['ci95'][0])}, {_f(s['ci95'][1])}]"
    sd = "" if s["std"] is None else f" ±{_f(s['std'])}"
    return f"{_f(s['mean'])}{sd}{ci} (n={s['n']})"


def render_md(rep: dict) -> str:
    L = [f"# Asfalia report: {rep['experiment']} (report v{rep['report_version']})", "",
         f"**Evidence level: {rep['evidence']}**", "",
         f"Seeds: {rep['seeds']}. Code revision: `{rep['code_revision'][:12]}`. Compatibility: checked "
         f"(world/split/catalog/schema/league/validation episodes identical across runs).", ""]
    if rep["rejected_comparisons"]:
        L += ["## Rejected comparisons", ""] + [f"- seed {r['seed']} {r['pair']}: {r['reason']}" for r in rep["rejected_comparisons"]] + [""]
    L += ["## Adaptation arms (validation split, last evaluation, per-seed + mean ± std [95% CI])", ""]
    for arm, row in rep["adaptation_arms"].items():
        b = row["budget"]
        L += [f"### {arm} (adapt {b['adapt_env_steps']}, pretrain {b['pretrain_env_steps']}, total {b['total_env_steps']} env steps)"]
        if row["missing_seeds"]:
            L += [f"MISSING for seeds {row['missing_seeds']} (not imputed)"]
        L += [f"- {m}: {_cell(v)}; per-seed {v['per_seed']}" for m, v in row["metrics"].items()] + [""]
    for key, title in (("stage_d_before_adaptation", "Stage D cross-play, before adaptation"),
                       ("stage_d_after_adaptation", "Stage D cross-play, after adaptation")):
        sec = rep[key]
        if not sec:
            L += [f"## {title}", "", "not present in the inputs", ""]
            continue
        L += [f"## {title} ({sec['split']} split, {sec['episodes_per_pair']} episodes per pair)", ""]
        if sec["missing_seeds"]:
            L += [f"MISSING for seeds {sec['missing_seeds']} (not imputed)", ""]
        for m, cp in sec["cross_play"].items():
            L += [f"**{m}** (rows attacker, cols defender; mean over seeds)", "",
                  "| attacker \\ defender | " + " | ".join(cp["cols"]) + " |", "|" + "---|" * (len(cp["cols"]) + 1)]
            L += [f"| {a} | " + " | ".join(_f(v) for v in row) + " |" for a, row in zip(cp["rows"], cp["values"])] + [""]
    L += ["## Figures", "", "- `security.svg`, `availability.svg`, `cost.svg`: Stage D, learned vs scripted defenders",
          "- `arms_security.svg`, `arms_availability.svg`, `arms_cost.svg`: adaptation arms", "",
          "## Notes", ""] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(L) + "\n"


BG, FG, MUT, PAL = "#0f1419", "#e6edf3", "#8b98a5", ["#4fb3d9", "#e3c36e", "#6ec1a3", "#e39a6e", "#b8a1e3"]


def svg_bars(title: str, groups: dict[str, dict[str, list[float]]], note: str, ylabel: str) -> str:
    """groups: category -> {series label -> per-seed values}. Bar = mean, dots = individual seeds."""
    cats = list(groups)
    series = sorted({s for g in groups.values() for s in g})
    allv = [v for g in groups.values() for vs in g.values() for v in vs] or [0.0]
    hi = max(max(allv), 1e-9) * 1.15
    W, H, L, T, Bt = 760, 330, 60, 52, 70
    gw = (W - L - 20) / max(len(cats), 1)
    bw = gw * 0.7 / max(len(series), 1)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="ui-monospace,Menlo,monospace" font-size="11">',
         f'<rect width="100%" height="100%" fill="{BG}"/>',
         f'<text x="{L}" y="20" fill="{FG}" font-size="13">{title}</text>',
         f'<text x="{L}" y="36" fill="{MUT}">{note}</text>',
         f'<text x="8" y="{T + (H - T - Bt) / 2:.0f}" fill="{MUT}" transform="rotate(-90 8 {T + (H - T - Bt) / 2:.0f})" '
         f'text-anchor="middle">{ylabel}</text>',
         f'<line x1="{L}" y1="{H - Bt}" x2="{W - 10}" y2="{H - Bt}" stroke="#2a3441"/>']
    y = lambda v: H - Bt - (v / hi) * (H - T - Bt)
    for k in range(5):
        v = hi * k / 4
        o.append(f'<text x="{L - 6}" y="{y(v) + 4:.1f}" fill="{MUT}" text-anchor="end">{v:.2f}</text>')
    for i, c in enumerate(cats):
        x0 = L + i * gw + gw * 0.15
        o.append(f'<text x="{x0 + gw * 0.35:.1f}" y="{H - Bt + 16}" fill="{FG}" text-anchor="middle">{c}</text>')
        for j, s in enumerate(series):
            vs = groups[c].get(s)
            if not vs:
                continue
            m, x = sum(vs) / len(vs), x0 + j * bw
            o.append(f'<rect x="{x:.1f}" y="{y(m):.1f}" width="{bw * 0.9:.1f}" height="{H - Bt - y(m):.1f}" fill="{PAL[j % 5]}" opacity="0.55"/>')
            for k, v in enumerate(vs):
                o.append(f'<circle cx="{x + bw * 0.45 + (k - (len(vs) - 1) / 2) * 4:.1f}" cy="{y(v):.1f}" r="2.5" fill="{FG}"/>')
    for j, s in enumerate(series):
        o.append(f'<rect x="{L + j * 190}" y="{H - 28}" width="10" height="10" fill="{PAL[j % 5]}"/>'
                 f'<text x="{L + j * 190 + 14}" y="{H - 19}" fill="{FG}">{s}</text>')
    return "\n".join(o + ["</svg>"]) + "\n"


def figures(rep: dict) -> dict[str, str]:
    out, note = {}, f"{rep['evidence'][:90]}"
    sec = rep["stage_d_before_adaptation"]
    for name, m in METRICS.items():
        if sec:
            g: dict = {}
            for pair, row in sec["pairs"].items():
                a, d = pair.split(" x ")
                g.setdefault(d, {})[f"vs {a}"] = list(row[m]["per_seed"].values())
            out[f"{name}.svg"] = svg_bars(f"Stage D {name} ({m}) by defender, before adaptation", g, note,
                                          m + (" (lower better)" if LOWER_IS_BETTER[m] else " (higher better)"))
        arms = rep["adaptation_arms"]
        if arms:
            out[f"arms_{name}.svg"] = svg_bars(
                f"Adaptation arms {name} ({m}), validation", {a: {"last eval": list(r["metrics"][m]["per_seed"].values())}
                                                              for a, r in arms.items()}, note, m)
    return out


def write(rep: dict, out: str | Path) -> list[str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"report.json": json.dumps(rep, indent=1, sort_keys=True) + "\n", "report.md": render_md(rep), **figures(rep)}
    for n, t in files.items():
        (out / n).write_text(t)
    return sorted(files)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hospital_rl.reporting.report", description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="experiment run dirs (one per seed)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        rep = build(a.runs)
        print(json.dumps({"evidence": rep["evidence"], "files": write(rep, a.out)}))
    except (ReportError, KeyError, json.JSONDecodeError) as e:
        print(f"error: {e!r}" if isinstance(e, KeyError) else f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

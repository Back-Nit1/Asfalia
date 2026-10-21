"""Runnable ablation definitions over Agent C's experiment configs, plus comparison validation.

Each ablation changes exactly one factor of a *full* config (pretrained school weights + league + adaptation). Ablations
the engine cannot run yet (workflow cost, domain randomization, full observability, telemetry loss) are real definitions
with a pointer to the request in docs/requests/agent-F.md; they refuse to generate a config rather than fake one.

    python -m hospital_rl.evaluation.ablations list
    python -m hospital_rl.evaluation.ablations write --base configs/experiments/ablation_full.yaml --out configs/experiments/ablations
    python -m hospital_rl.evaluation.ablations check FULL.yaml VARIANT.yaml --ablation no_opponent_league
Exit codes: 0 ok, 2 invalid/unavailable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel

from ..training.experiments import ExperimentConfig, load_experiment
from .strata import StratumError, check_training_config


class AblationError(ValueError):
    pass


class Unavailable(AblationError):
    """The engine/env cannot run this ablation yet."""


class Ablation(BaseModel):
    name: str
    factor: str
    description: str
    changes: dict = {}                    # ExperimentConfig field -> new value (None = drop the block)
    request: str | None = None            # docs/requests/agent-F.md section when the engine cannot run it
    same_run: bool = False                # comparison is within one run (e.g. stage D before vs after adaptation)


ABLATIONS = {a.name: a for a in (
    Ablation(name="no_pretraining", factor="initialization", changes={"arms": ["scratch_same_adapt"]},
             description="random-init defender, same adaptation budget as the pretrained arm (equal_adapt rule)"),
    Ablation(name="no_pretraining_equal_total", factor="initialization", changes={"arms": ["scratch_equal_total"]},
             description="random-init defender given adapt + school steps (equal_total rule)"),
    Ablation(name="no_adaptation", factor="adaptation", changes={"arms": [], "adapt": None}, same_run=True,
             description="frozen Stage D transfer only; compare with the full run's before-adaptation report"),
    Ablation(name="no_opponent_league", factor="opponent league", changes={"league": "off"},
             description="adapt against the single frozen opponent instead of the league"),
    Ablation(name="no_workflow_cost", factor="workflow cost", request="1",
             description="zero the clinical-workflow penalty terms in the defender reward"),
    Ablation(name="no_domain_randomization", factor="domain randomization", request="2",
             description="fixed topology/workload/vulnerability draw instead of per-episode randomization"),
    Ablation(name="full_observability_diagnostic", factor="visibility mode", request="3",
             description="defender sees true state; DIAGNOSTIC upper bound, never a deployable result"),
    *(Ablation(name=f"telemetry_loss_{int(p * 100):02d}", factor="telemetry loss", request="4",
               description=f"defender alert drop probability {p} (engine fixes 0.1 today)") for p in (0.0, 0.3, 0.5)),
)}


def make_config(base: ExperimentConfig, name: str) -> ExperimentConfig:
    ab = ABLATIONS.get(name)
    if ab is None:
        raise AblationError(f"unknown ablation {name!r}; known: {sorted(ABLATIONS)}")
    if ab.request:
        raise Unavailable(f"{name}: engine support missing, see docs/requests/agent-F.md section {ab.request}")
    d = base.model_dump() | ab.changes | {"name": f"{base.name}__{name}"}
    return ExperimentConfig.model_validate(d)


def check_pair(full: ExperimentConfig, variant: ExperimentConfig, name: str) -> None:
    """Reject comparisons that differ in anything but the declared factor, touch the test stratum, or are unrunnable."""
    ab = ABLATIONS.get(name)
    if ab is None:
        raise AblationError(f"unknown ablation {name!r}")
    if ab.request:
        raise Unavailable(f"{name} is not runnable yet (docs/requests/agent-F.md section {ab.request})")
    a, b = full.model_dump(), variant.model_dump()
    allowed = {"name", *ab.changes}
    diff = sorted(k for k in a if a[k] != b[k])
    if sorted(set(diff) - allowed):
        raise AblationError(f"{name}: configs differ in {sorted(set(diff) - allowed)}; only {sorted(allowed)} may change")
    for k, v in ab.changes.items():
        if b[k] != v:
            raise AblationError(f"{name}: variant must set {k}={v!r}, has {b[k]!r}")
    if full.arms and "pretrained" not in full.arms and name != "no_adaptation":
        raise AblationError(f"{name}: the full config must run the pretrained arm")
    for cfg in (full, variant):
        try:
            check_training_config(cfg.model_dump(), cfg.name)
        except StratumError as e:
            raise AblationError(str(e)) from e
        if cfg.transfer.split == "test":
            raise AblationError(f"{cfg.name}: ablation comparisons never use the test stratum")


def check_runs(full: dict, variant: dict, name: str) -> None:
    """Stored adapt manifests (one per run) must agree on everything except the ablated factor."""
    ab = ABLATIONS[name]
    skip = {"league"} if name == "no_opponent_league" else set()
    a, b = full["comparability"], variant["comparability"]
    bad = [f"{k}: {a[k]!r} != {b[k]!r}" for k in a if k not in skip and a[k] != b[k]]
    if full["code_revision"] != variant["code_revision"]:
        bad.append("code_revision differs")
    if bad:
        raise AblationError(f"{name}: runs are not comparable - " + "; ".join(bad))
    if ab.factor == "opponent league" and a["league"] == b["league"]:
        raise AblationError(f"{name}: both runs have league={a['league']}; nothing was ablated")


def write_all(base: ExperimentConfig, out: str | Path) -> dict[str, str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    for n, ab in ABLATIONS.items():
        if ab.request:
            continue
        cfg = make_config(base, n)
        check_pair(base, cfg, n)
        p = out / f"{n}.yaml"
        p.write_text(f"# ablation {n}: {ab.description}\n# generated by hospital_rl.evaluation.ablations from {base.name}; "
                     f"do not edit\n" + yaml.safe_dump(cfg.model_dump(exclude_none=True), sort_keys=False))
        done[n] = str(p)
    return done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hospital_rl.evaluation.ablations", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    w = sub.add_parser("write")
    w.add_argument("--base", required=True)
    w.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("full")
    c.add_argument("variant")
    c.add_argument("--ablation", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "list":
            for n, ab in ABLATIONS.items():
                print(f"{n:32}{'REQUEST ' + ab.request if ab.request else 'runnable':12}{ab.description}")
        elif a.cmd == "write":
            for n, f in write_all(load_experiment(a.base), a.out).items():
                print(f"{n} -> {f}")
        else:
            check_pair(load_experiment(a.full), load_experiment(a.variant), a.ablation)
            print("comparison valid")
    except (AblationError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

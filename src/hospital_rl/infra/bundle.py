"""Resolve and validate the versioned job bundle (``JobSpec``) before anything is launched, locally or remotely.

``resolve`` fills the derived pins (output path, hashes, split/seed range, horizon, labels); ``validate`` refuses a
bundle that is stale, unsafe, or not runnable from a clean container: the environment factory must live in the
``hospital_rl`` package (never a test module), the configs it names must ship with the code, and the code/config
hashes must match what the job will actually run. Remote containers re-validate with ``check_code=True``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from hospital_rl.infra.backend import BUNDLE_VERSION, JobSpec

OWNER_LABEL = {"owner": "asfalia"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")  # ids become path components: no separators, no '..'
PACKAGE = "hospital_rl"
DEPS = ["numpy", "networkx", "gymnasium", "pettingzoo", "pydantic>=2", "pyyaml"]  # runtime deps of a clean container
TORCH_CPU = "https://download.pytorch.org/whl/cpu"  # default linux torch wheel drags in ~3GB of CUDA


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]  # .../hospital_rl


def repo_root() -> Path:
    """Directory that holds ``configs/``: the checkout, or /root inside a Modal container."""
    for d in (package_root().parents[1], package_root().parent, Path.cwd()):
        if (d / "configs").is_dir():
            return d
    return Path.cwd()


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(package_root().rglob("*.py")):
        if "__pycache__" not in p.parts:
            h.update(str(p.relative_to(package_root())).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def config_hash(spec: JobSpec) -> str:
    blob = json.dumps({"kind": spec.kind, "train": spec.train_config, "scenario": spec.scenario}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def total_steps(spec: JobSpec) -> int:
    return int(spec.train_config.get("total_steps" if spec.kind == "train" else "steps", 0))


def resolve(spec: JobSpec) -> JobSpec:
    """Return a copy with every derived pin filled in (idempotent)."""
    from hospital_rl.evaluation.splits import SPLITS

    kw = spec.train_config.get("env_kwargs", {}) if spec.kind == "train" else {}
    scenario = spec.scenario or ({"factory": spec.train_config.get("env_factory"), **kw} if spec.kind == "train"
                                 else {"config": spec.train_config.get("config"), "split": "train"})
    split = spec.split or scenario.get("split")
    r = SPLITS[split]["seeds"] if split in SPLITS else None
    out = spec.model_copy(update={
        "scenario": scenario, "split": split, "seed_range": spec.seed_range or ((r.start, r.stop) if r else None),
        "horizon": spec.horizon if spec.horizon is not None else scenario.get("horizon"),
        "output_path": f"{spec.run_id}/{spec.attempt_id}", "labels": {**OWNER_LABEL, **spec.labels}})
    return out.model_copy(update={"config_hash": config_hash(out), "code_hash": code_hash()})


def validate(spec: JobSpec, *, check_code: bool = False, import_check: bool = True) -> None:
    """Raise ValueError listing every problem. ``check_code`` additionally pins the source tree hash
    (used inside the container to catch a stale image)."""
    from hospital_rl.evaluation.splits import SPLITS
    from hospital_rl.training.train import TrainConfig, load_factory

    bad: list[str] = []
    if spec.bundle_version != BUNDLE_VERSION:
        bad.append(f"bundle_version {spec.bundle_version} != supported {BUNDLE_VERSION}")
    if spec.kind not in ("train", "school"):
        bad.append(f"unknown kind {spec.kind!r}")
    for n in ("run_id", "attempt_id", "resume_from_attempt"):
        v = getattr(spec, n)
        if v is not None and not SAFE_ID.match(v):
            bad.append(f"{n} {v!r} is not a safe path component")
    if spec.resume_from_attempt == spec.attempt_id:
        bad.append("resume_from_attempt equals attempt_id")
    if spec.output_path != f"{spec.run_id}/{spec.attempt_id}":
        bad.append(f"output_path {spec.output_path!r} != '{spec.run_id}/{spec.attempt_id}' (not resolved?)")
    if spec.config_hash != config_hash(spec):
        bad.append("config_hash does not match train_config/scenario")
    if check_code and spec.code_hash != code_hash():
        bad.append(f"code_hash {spec.code_hash} != container code {code_hash()} (stale image?)")
    if spec.split is not None and spec.split not in SPLITS:
        bad.append(f"unknown split {spec.split!r}")
    elif spec.split and spec.seed_range and tuple(spec.seed_range) != (SPLITS[spec.split]["seeds"].start, SPLITS[spec.split]["seeds"].stop):
        bad.append(f"seed_range {spec.seed_range} does not match split {spec.split!r}")
    if total_steps(spec) <= 0:
        bad.append("step cap (total_steps/steps) must be > 0")
    if not bad:
        try:
            if spec.kind == "train":
                cfg = TrainConfig.model_validate(spec.train_config)
                if not cfg.env_factory.startswith(PACKAGE + "."):
                    bad.append(f"env_factory {cfg.env_factory!r} must live in the {PACKAGE} package (no test modules)")
                elif import_check:
                    load_factory(cfg.env_factory)
            else:
                c = spec.train_config
                if c.get("role") not in ("attacker", "defender") or not (repo_root() / c.get("config", "")).is_file():
                    bad.append(f"school needs role and an existing config path (got {c.get('role')!r}, {c.get('config')!r})")
        except Exception as e:  # noqa: BLE001 - report any resolution failure as a bundle problem
            bad.append(f"{type(e).__name__}: {e}")
    if bad:
        raise ValueError("invalid job bundle: " + "; ".join(bad))

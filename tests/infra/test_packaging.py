"""A clean container can run the configured job: only the shipped package + configs, never repo test modules."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from helpers import spec

from hospital_rl.infra import bundle
from hospital_rl.infra.backend import JobSpec

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("cfg", ["cloud_local.yaml", "cloud_modal.yaml"])
def test_presets_use_real_package_environment(cfg):
    raw = yaml.safe_load((ROOT / "configs" / cfg).read_text())
    assert raw["train"]["env_factory"].startswith("hospital_rl.") and "toy_env" not in str(raw)
    sp = bundle.resolve(JobSpec(run_id="r", attempt_id="a", train_config=raw["train"]))
    bundle.validate(sp)
    assert sp.split == "train" and sp.seed_range == (0, 10_000) and sp.horizon == 40
    assert sp.output_path == "r/a" and sp.code_hash and sp.config_hash and sp.labels["owner"] == "asfalia"


def test_clean_interpreter_imports_and_builds_the_configured_env(tmp_path):
    """Fresh interpreter, cwd outside the repo, sys.path = package source only: no tests/ on the path."""
    code = f"""
import sys, yaml
sys.path[:] = [p for p in sys.path if 'tests' not in p]
from hospital_rl.infra import bundle
from hospital_rl.infra.backend import JobSpec
from hospital_rl.training.train import load_factory
raw = yaml.safe_load(open({str(ROOT / 'configs/cloud_modal.yaml')!r}))['train']
sp = bundle.resolve(JobSpec(run_id='r', attempt_id='a', train_config=raw))
bundle.validate(sp, check_code=True)
env = load_factory(raw['env_factory'])(**raw['env_kwargs'])
assert env.metadata['role'] == 'defender'
assert 'toy_env' not in sys.modules
print('ok')
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")})
    assert r.stdout.strip() == "ok", r.stderr[-800:]


def test_stale_code_and_tampered_config_are_rejected():
    sp = bundle.resolve(spec())
    bundle.validate(sp, check_code=True)
    with pytest.raises(ValueError, match="stale image"):
        bundle.validate(sp.model_copy(update={"code_hash": "0" * 16}), check_code=True)
    with pytest.raises(ValueError, match="config_hash"):
        bundle.validate(sp.model_copy(update={"train_config": {**sp.train_config, "total_steps": 1}}))
    with pytest.raises(ValueError, match="seed_range"):
        bundle.validate(sp.model_copy(update={"seed_range": (0, 5)}))
    with pytest.raises(ValueError, match="bundle_version"):
        bundle.validate(sp.model_copy(update={"bundle_version": 99}))


def test_modal_image_ships_package_and_configs_not_tests():
    modal = pytest.importorskip("modal")
    from hospital_rl.infra.modal_app import ModalConfig, build_image
    img = build_image(ModalConfig())
    layers = repr([getattr(l, "__dict__", l) for l in img._deferred_mounts]) if hasattr(img, "_deferred_mounts") else ""
    assert "toy_env" not in layers and "tests" not in layers


def test_real_modal_functions_module_defines_a_global_deployable_app():
    modal = pytest.importorskip("modal")
    from hospital_rl.infra.modal_app import CFG_ENV, FUNCTION, RealModalGateway, ModalConfig
    gw = RealModalGateway(ModalConfig(app_name="hospital-rl-test-import"))  # defines the app; no network
    assert FUNCTION in gw.fns.app.registered_functions and gw.fns.train_job.info.raw_f is not None

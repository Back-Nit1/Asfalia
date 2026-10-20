"""Real Modal smoke run: ONE tiny bounded job. Spends real (small) compute on your Modal account.

  python tests/infra/modal_smoke.py            # needs `modal setup` or MODAL_TOKEN_ID/SECRET
Caps: 1 cpu, 1 GiB, 120 s timeout, 1 container, 0 retries, 2048 env steps.
"""
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from hospital_rl.infra.backend import JobSpec
from hospital_rl.infra.modal_app import ModalBackend, ModalConfig, RealModalGateway
from hospital_rl.infra.plan import credentials_present

if not credentials_present("modal"):
    sys.exit("no Modal credentials: run `modal setup` (see docs/training_infra.md). Falling back to local; nothing launched.")

cfg = ModalConfig(app_name="hospital-rl-smoke", volume_name="hospital-rl-smoke-runs", extra_modules=["toy_env"],
                  caps=dict(cpu=1, memory_mb=1024, timeout_s=120, max_containers=1, max_retries=0, max_steps=2048))
gw = RealModalGateway(cfg)
be = ModalBackend(cfg, gw)
run, attempt = f"smoke-{uuid.uuid4().hex[:8]}", "a1"
spec = JobSpec(run_id=run, attempt_id=attempt, timeout_s=120, cpu=1, memory_mb=1024, grace_s=20,
               train_config=dict(role="defender", total_steps=2048, n_envs=4, rollout_len=64, hidden=32,
                                 checkpoint_every=2, env_factory="toy_env:make_env"))
t0 = time.time()
jid = be.submit(spec)
print("submitted", jid, "(modal call id)")
while not (st := be.status(jid)).state.terminal and time.time() - t0 < 300:
    time.sleep(3)
print("final:", st.state.value, st.detail, st.result, f"{time.time() - t0:.0f}s wall")
files = [e.path for e in gw.vol.listdir(f"/{run}/{attempt}")]
print("volume files:", files)
assert st.state.value == "done" and "manifest.json" in files and "latest.pt" in files
# cleanup: delete this run's files from the Volume
gw.vol.remove_file(f"/{run}", recursive=True)
print("cleanup: removed", f"/{run}", "->", [e.path for e in gw.vol.listdir("/")])

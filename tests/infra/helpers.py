"""Shared test helpers: real hospital env specs and an in-process fake gateway (no network, no credentials)."""
import threading
import time

from hospital_rl.envs.factories import hospital_single
from hospital_rl.infra.backend import JobSpec
from hospital_rl.infra.job import control_path, run_job

ENV_KW = dict(role="defender", size=8, split="train", difficulty=0.5, horizon=20, opponent="scripted")


def spec(attempt="a1", steps=256, run="r1", **kw):
    tc = dict(role="defender", total_steps=steps, n_envs=2, rollout_len=16, hidden=16, checkpoint_every=1,
              env_factory="hospital_rl.envs.factories:hospital_single", env_kwargs=ENV_KW, ppo={"epochs": 1, "minibatches": 1})
    return JobSpec(run_id=run, attempt_id=attempt, train_config=tc, timeout_s=120, **kw)


def ids():
    m = hospital_single(**ENV_KW).metadata
    return m["role"], m["catalog_hash"], m["observation_schema_hash"]


class FakeGateway:
    """In-process stand-in for a remote provider: runs run_job in a thread over a temp dir."""

    def __init__(self, root):
        self.root, self.calls, self.spawned = root, {}, 0

    def spawn(self, sp):
        self.spawned += 1
        cid = f"fc-{self.spawned}"
        slot = self.calls[cid] = {"state": "running", "payload": {}, "key": (sp.run_id, sp.attempt_id)}

        def work():
            try:
                slot.update(state="done", payload=run_job(sp, self.root, log=lambda s: None))
            except Exception as e:  # noqa: BLE001
                slot.update(state="failed", payload=repr(e))

        t = slot["thread"] = threading.Thread(target=work, daemon=True)
        t.start()
        return cid

    def poll(self, call_id):
        s = self.calls[call_id]
        return s["state"], s["payload"]

    def cancel(self, call_id):
        # threads share the process-global torch RNG with the rest of the test run, so a "cancelled"
        # job must really stop before the test returns
        slot = self.calls[call_id]
        self.write_control(*slot["key"], "stop")
        slot["thread"].join(60)
        slot.update(state="failed", payload="cancelled")

    def write_control(self, run_id, attempt_id, word):
        p = control_path(self.root, run_id, attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(word)

    def read_file(self, run_id, attempt_id, name):
        p = self.root / run_id / attempt_id / name
        return p.read_bytes() if p.exists() else None

    def list_attempts(self):
        return [(r.name, a.name) for r in sorted(self.root.iterdir()) if r.is_dir() for a in sorted(r.iterdir())
                if a.is_dir() and "control" not in (r.name, a.name)]

    def delete_attempt(self, run_id, attempt_id):
        import shutil
        shutil.rmtree(self.root / run_id / attempt_id, ignore_errors=True)


class ScriptedGateway(FakeGateway):
    """Never runs anything; poll returns whatever the test scripts. Records every call."""

    def __init__(self, root, outcome=("running", {})):
        super().__init__(root)
        self.outcome, self.log = outcome, []

    def spawn(self, sp):
        self.spawned += 1
        self.log.append(("spawn", sp.attempt_id))
        return f"sc-{self.spawned}"

    def poll(self, call_id):
        return self.outcome

    def cancel(self, call_id):
        self.log.append(("cancel", call_id))

    def write_control(self, run_id, attempt_id, word):
        self.log.append(("control", word))


def wait_state(be, jid, timeout=60):
    end = time.time() + timeout
    while time.time() < end and not (st := be.status(jid)).state.terminal:
        time.sleep(0.1)
    return st

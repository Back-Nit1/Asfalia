import pytest
from fastapi.testclient import TestClient

from hospital_rl.api.app import create_app
from hospital_rl.api.auth import AuthConfig
from hospital_rl.api.storage import Store
from hospital_rl.events import EventEnvelope
from hospital_rl.orchestration.contracts import PlanConfig
from hospital_rl.orchestration.scheduler import Scheduler
from hospital_rl.orchestration.tasks import Task

OP, EV = "op-token-123456", "ev-token-654321"


def ev(seq, *, source="development", worker="w1", visibility="operator", task_id=None, **kw):
    return EventEnvelope(source_kind=source, worker_id=worker, sequence=seq, event_type=kw.pop("event_type", "t"),
                         visibility=visibility, task_id=task_id, **kw)


@pytest.fixture
def store():
    return Store()


@pytest.fixture
def make_client(store):
    def _make(auth=None, tasks=None):
        sched = Scheduler(store, PlanConfig(), tasks or [Task(id="t1", role="r", description="d", owner="human", assigned_paths=["p"])])
        return TestClient(create_app(store, sched, auth or AuthConfig(OP, EV)))
    return _make


@pytest.fixture
def client(make_client):
    return make_client()


def sse(resp):
    """Parse a finished SSE body into [(id, event, data)]."""
    import json
    out = []
    for block in resp.text.split("\n\n"):
        f = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in f:
            out.append((int(f["id"]) if "id" in f else None, f["event"], json.loads(f["data"])))
    return out

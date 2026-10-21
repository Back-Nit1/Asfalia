import json
from pathlib import Path

import pytest

from hospital_rl.api.storage import Store
from hospital_rl.experiments.controller import Controller
from hospital_rl.experiments.fakes import FakeBackend, FakeEvaluator
from hospital_rl.experiments.reader import FsReader

FIXTURE = Path(__file__).parents[2] / "src/hospital_rl/api/fixtures/fixture_tiny_episode.json"


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make(tmp_path, clock):
    """make(db=..., backend=...) -> Controller; reuse tmp_path/db to simulate a process restart."""
    def _make(backend=None, db=None, **kw):
        store = Store(db or tmp_path / "x.sqlite")
        be = backend or FakeBackend(tmp_path / "art")
        kw.setdefault("evaluator", FakeEvaluator(FIXTURE))
        kw.setdefault("episodes_dir", str(tmp_path / "eps" / "x"))
        return Controller(store, be, FsReader(tmp_path / "art"), clock=clock, **kw)
    return _make


@pytest.fixture
def ctl(make):
    return make()


def drive(ctl, n=1):
    for _ in range(n):
        ctl.tick()

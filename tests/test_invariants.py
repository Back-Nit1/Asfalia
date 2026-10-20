import random

import pytest

from hospital_rl.engine.transitions import World
from hospital_rl.scenarios import FAMILIES, generate


@pytest.mark.parametrize("family", FAMILIES)
def test_random_legal_play_preserves_invariants(family):
    for seed in range(4):
        spec = generate(family, 16, seed, difficulty=0.8)
        w, r = World(spec, seed), random.Random(seed)
        total_records = {n.id: n.records for n in spec.nodes}
        while True:
            info = w.step(r.choice(w.legal_attacker()), r.choice(w.legal_defender()))
            st = w.st
            assert st.budget_left >= -1e-9 and st.m.spend <= spec.budget + 1e-9
            assert info["reward"]["defender"] <= 1e-12                    # no profit from cycling controls
            assert st.m.records_exfiltrated <= st.m.records_accessed
            for n, ns in st.nodes.items():
                assert len(ns.taken) <= total_records[n] and ns.staged <= ns.taken and ns.down_jobs >= 0
            assert len(st.jobs) <= spec.action_capacity
            assert sum(len(ns.taken) - len(ns.staged) for ns in st.nodes.values()) == st.m.records_exfiltrated
            if info["done"]:
                break


def test_benign_only_has_no_attacker_activity():
    from hospital_rl.engine.transitions import WAIT
    w = World(generate("tiered", 12, 1), 1)
    while not w.step(WAIT, WAIT)["done"]:
        pass
    assert w.st.m.records_accessed == 0 and w.st.m.first_access_tick is None


def test_defender_return_includes_remaining_horizon():
    """Episodes always run the fixed horizon; no early termination to dodge downtime penalties."""
    from hospital_rl.policies.scripted import IsolateEverything, Noop
    from hospital_rl.runner import run_episode
    spec = generate("hub", 12, 2)
    assert len(run_episode(spec, 2, Noop(), IsolateEverything())["steps"]) == spec.horizon

import random
from collections import Counter

import numpy as np
import pytest

from hospital_rl.training.curriculum import Curriculum
from hospital_rl.training.league import Entry, League, cross_play


def test_league_bounded_disjoint_and_sampling():
    lg = League(max_checkpoints=3, p_scripted=0.3, p_latest=0.5)
    lg.add_train(Entry("greedy", "scripted"))
    for i in range(6):
        lg.add_train(Entry(f"c{i}", "checkpoint", f"p{i}"))
    assert [e.id for e in lg.checkpoints] == ["c3", "c4", "c5"] and len(lg.scripted) == 1
    lg.add_eval(Entry("holdout", "checkpoint", "h"))
    with pytest.raises(ValueError):
        lg.add_train(Entry("holdout", "checkpoint"))
    with pytest.raises(ValueError):
        lg.add_eval(Entry("c5", "checkpoint"))
    rng = random.Random(0)
    counts = Counter(lg.sample(rng).id for _ in range(4000))
    assert "holdout" not in counts and "c0" not in counts
    assert 0.25 < counts["greedy"] / 4000 < 0.35 and counts["c5"] > counts["c3"]


def test_cross_play_matrix():
    mat, full = cross_play(lambda a, d: {"win": float(len(a) > len(d))}, ["aa", "a"], ["d", "ddd"], "win")
    assert np.array_equal(mat, [[1, 0], [0, 0]]) and full["aa", "d"]["win"] == 1.0


def test_curriculum_advances_on_validation_only():
    c = Curriculum(["small", "mid", "big"], threshold=0.8, window=2)
    assert not c.report(0.9) and not c.report(0.5) and c.level == "small"
    assert not c.report(0.85) and c.report(0.9) and c.level == "mid"
    s = c.state_dict()
    c2 = Curriculum(["small", "mid", "big"], 0.8, 2)
    c2.load_state_dict(s)
    assert c2.level == "mid"

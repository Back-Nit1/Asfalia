import pytest

from hospital_rl.orchestration.tasks import TRANSITIONS, IllegalTransition, Status, Task, load_plan, paths_overlap


@pytest.mark.parametrize("frm,to", [(f, t) for f in Status for t in Status if t not in TRANSITIONS[f]])
def test_illegal_transitions_rejected(frm, to):
    t = Task(id="x", role="r", description="d", status=frm)
    with pytest.raises(IllegalTransition):
        t.move(to)


def test_happy_path_is_legal():
    t = Task(id="x", role="r", description="d")
    for s in (Status.ready, Status.running, Status.review, Status.passed, Status.integrated):
        t.move(s)


def test_terminal_states_have_no_exits():
    assert not TRANSITIONS[Status.integrated] and not TRANSITIONS[Status.cancelled]


def test_paths_overlap():
    assert paths_overlap(["src/a/**"], ["src/a/b.py"])
    assert paths_overlap(["pyproject.toml"], ["pyproject.toml"])
    assert not paths_overlap(["src/a"], ["src/ab"])


def test_plan_rejects_cycle(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("tasks:\n - {id: a, role: r, description: d, dependencies: [b]}\n - {id: b, role: r, description: d, dependencies: [a]}\n")
    with pytest.raises(ValueError, match="cycle"):
        load_plan(str(p))

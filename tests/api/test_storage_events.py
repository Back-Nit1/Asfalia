import pytest

from .conftest import ev


def test_dedupe_by_event_id(store):
    e = ev(0)
    assert store.append_event(e) is not None
    assert store.append_event(e) is None
    assert len(store.events_after(0)) == 1


def test_order_within_a_source_regardless_of_arrival(store):
    for seq in (2, 0, 1):
        store.append_event(ev(seq))
    store.append_event(ev(0, worker="w2"))   # other worker: independent sequence space
    assert [e.sequence for e in store.stream_events("development", "w1")] == [0, 1, 2]
    assert [c for c, _ in store.events_after(0)] == [1, 2, 3, 4]   # cursor feed is arrival order


def test_sequence_clash_with_new_event_id_rejected(store):
    store.append_event(ev(0))
    with pytest.raises(ValueError):
        store.append_event(ev(0))


def test_retention_prunes_and_marks_resync_floor(store):
    for i in range(10):
        store.append_event(ev(i))
    assert store.prune_events(keep=4) == 6
    assert store.oldest_cursor() == 7 and store.pruned_through() == 6


def test_next_sequence_survives_restart(store):
    store.append_event(ev(0))
    store.append_event(ev(1))
    assert store.next_sequence("development", "w1") == 2

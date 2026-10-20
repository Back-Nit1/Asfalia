from .conftest import ev, sse


def fill(store, n=5):
    for i in range(n):
        store.append_event(ev(i, event_type=f"e{i}"))


def test_sse_replays_from_start_with_cursor_ids(client, store):
    fill(store)
    r = client.get("/api/events/stream?follow=false")
    assert r.headers["content-type"].startswith("text/event-stream")
    assert [i for i, *_ in sse(r)] == [1, 2, 3, 4, 5]


def test_sse_reconnect_with_last_event_id_resumes(client, store):
    fill(store)
    r = client.get("/api/events/stream?follow=false", headers={"Last-Event-ID": "3"})
    got = sse(r)
    assert [i for i, *_ in got] == [4, 5] and [d["event_type"] for *_, d in got] == ["e3", "e4"]
    store.append_event(ev(5, event_type="e5"))
    assert [i for i, *_ in sse(client.get("/api/events/stream?follow=false", headers={"Last-Event-ID": "5"}))] == [6]


def test_sse_resync_after_retention_expired(client, store):
    fill(store, 10)
    store.prune_events(keep=3)
    got = sse(client.get("/api/events/stream?follow=false", headers={"Last-Event-ID": "2"}))
    assert got[0][1] == "resync" and "tasks" in got[0][2] and got[0][2]["cursor"] == 10
    assert [e for _, e, _ in got[1:]] == []   # nothing newer than the snapshot cursor
    store.append_event(ev(10))
    got = sse(client.get("/api/events/stream?follow=false", headers={"Last-Event-ID": got[0][2]["cursor"].__str__()}))
    assert [i for i, *_ in got] == [11]


def test_cursor_in_query_works_without_header(client, store):
    fill(store)
    assert [i for i, *_ in sse(client.get("/api/events/stream?follow=false&after=4"))] == [5]

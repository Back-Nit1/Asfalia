from hospital_rl.orchestration.tasks import Status


def post(client, path, key="k1", **kw):
    return client.post(path, headers={"Idempotency-Key": key}, **kw)


def test_idempotent_cancel_replays_without_second_effect(client, store):
    first = post(client, "/api/tasks/t1/cancel")
    n_events = len(store.events_after(0))
    again = post(client, "/api/tasks/t1/cancel")
    assert first.status_code == 200 and again.status_code == 200 and again.json() == first.json()
    assert again.headers["idempotent-replay"] == "true"
    assert len(store.events_after(0)) == n_events      # no second state change
    assert client.get("/api/tasks/t1").json()["status"] == Status.cancelled.value


def test_without_key_rejected_and_state_untouched(client):
    assert client.post("/api/tasks/t1/cancel").status_code == 400
    assert client.get("/api/tasks/t1").json()["status"] == "queued"


def test_refusals_are_replayed_too(client):
    a = post(client, "/api/tasks/t1/heartbeat", key="hb")
    b = post(client, "/api/tasks/t1/heartbeat", key="hb")
    assert a.status_code == 409 and b.json() == a.json()   # task not running; the refusal itself is replayed


def test_same_key_on_different_tasks_is_independent(client):
    assert post(client, "/api/tasks/t1/cancel", key="same").status_code == 200
    assert post(client, "/api/tasks/nope/cancel", key="same").status_code == 404


def test_illegal_transition_is_409_and_unknown_task_404(client):
    post(client, "/api/tasks/t1/cancel", key="a")
    assert post(client, "/api/tasks/t1/cancel", key="b").status_code == 409
    assert post(client, "/api/tasks/nope/cancel", key="c").status_code == 404

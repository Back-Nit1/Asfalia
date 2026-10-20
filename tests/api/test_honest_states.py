def test_unwired_surfaces_say_not_configured(client):
    for path in ("/api/runs", "/api/infrastructure", "/api/checkpoints"):
        b = client.get(path).json()
        assert b["configured"] is False and b["state"] == "not_configured", path
    assert client.post("/api/runs", headers={"Idempotency-Key": "x"}).status_code == 501
    o = client.get("/api/overview").json()
    assert o["runs"]["count"] == 0 and o["step_rate"] is None and o["spend_estimate"] is None
    assert o["runtime"] == {"configured": False, "name": None, "detail": "coding runtime not configured"}


def test_episode_list_marks_fixture_and_no_live(client):
    b = client.get("/api/world/episodes").json()
    assert b["episodes"][0]["label"] == "FIXTURE" and b["live"]["configured"] is False


def test_replay_scrub_bounds(client):
    s = client.get("/api/world/defender/episodes/fixture_tiny_episode/snapshot?tick=9999").json()
    assert s["tick"] == s["last_tick"] and s["mode"] == "REPLAY" and s["label"] == "FIXTURE"
    assert client.get("/api/world/defender/episodes/nope/snapshot").status_code == 404

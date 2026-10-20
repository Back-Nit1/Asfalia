from hospital_rl.events import EventEnvelope


def test_envelope_roundtrip():
    e = EventEnvelope(source_kind="world", event_type="x", worker_id="w", sequence=1)
    assert EventEnvelope.model_validate_json(e.model_dump_json()) == e

import re
from pathlib import Path

from hospital_rl.evaluation import attack_mapping as M
from hospital_rl.schemas import ATTACKER_TYPES

DOC = (Path(__file__).resolve().parents[2] / "docs/mitre_mapping.md").read_text()


def test_every_attacker_action_mapped_with_valid_ids_and_documented():
    assert {a.value for a in ATTACKER_TYPES} == set(M.ATTACKER_MAP)
    for act, (ids, fit, _) in M.ATTACKER_MAP.items():
        assert fit in ("partial", "exact") and ids and all(re.fullmatch(r"T\d{4}", i) and i in M.TECHNIQUES for i in ids)
        assert f"`{act}`" in DOC
    for tid, (name, _) in M.TECHNIQUES.items():
        assert tid in DOC and name in DOC
    assert "incomplete" in M.COVERAGE.lower() and "Incomplete" in DOC

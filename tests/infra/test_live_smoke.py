import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("ASFALIA_LIVE_MODAL") != "1", reason="live Modal smoke is opt-in (ASFALIA_LIVE_MODAL=1)")


def test_live_modal_smoke():
    r = subprocess.run([sys.executable, str(Path(__file__).with_name("modal_smoke.py"))], capture_output=True, text=True, timeout=900)
    assert "SMOKE OK" in r.stdout, r.stdout[-800:] + r.stderr[-800:]

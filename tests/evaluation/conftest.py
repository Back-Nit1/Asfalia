import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "training"))
from real_ckpts import school_ckpts, world_path  # noqa: E402,F401  (session fixtures)

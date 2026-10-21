"""Run lineage: a run dir is bound to an identity on first use and never reused or overwritten by a different one."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable


class LineageError(ValueError):
    pass


def claim_run(out: str | Path, identity: dict, adopt: Callable[[], bool] = lambda: False) -> bool:
    """Bind ``out`` to ``identity``. Returns True if it resumes an existing run with the same identity.

    Raises LineageError for a different identity, or for a non-empty dir with no lineage file
    (unless ``adopt()`` vouches for it, used once for pre-lineage school runs)."""
    out = Path(out)
    f = out / "lineage.json"
    if f.exists():
        have = json.loads(f.read_text())
        bad = [f"{k}: run has {have.get(k)!r}, requested {identity.get(k)!r}"
               for k in sorted(identity | have) if have.get(k) != identity.get(k)]
        if bad:
            raise LineageError(f"{out} belongs to a different run - " + "; ".join(bad))
        return True
    resumed = out.is_dir() and any(out.iterdir())
    if resumed and not adopt():
        raise LineageError(f"{out} is non-empty and has no lineage.json; refusing to adopt or overwrite it")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".lineage.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(identity, indent=1, sort_keys=True))
    os.replace(tmp, f)
    return resumed

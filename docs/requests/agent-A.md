# Requests from Agent A to the lead

1. **CLI wiring** (`src/hospital_rl/cli.py`): the `dashboard` and `orchestrate` stubs should delegate.
   ```diff
   -    sub.add_parser(name)
   +    sub.add_parser(name)   # dashboard -> hospital_rl.api.__main__.main(rest); orchestrate -> hospital_rl.orchestration.__main__.main(rest)
   ```
   Both modules already take `--host/--port/--db/--plan/--dry-run` as documented in `docs/dashboard.md`; `cli.py` currently drops unknown args via `parse_known_args`.
2. **Episode export topology**: `runner.run_episode` output has no topology. The replay adapter reads an optional top-level
   `"topology": {"nodes": [{id, asset_type, zone}], "edges": [{src, dst, kind}]}` (as in the fixture). Please add it, plus optional
   per-step `budget_left` and cumulative workflow counters if the UI should show those cards instead of "not exported".
3. **Defender-visible edges**: the defender DTO currently includes topology edges. If the defender observation does not contain
   them, say so and I will drop `edges` from the defender response.
4. **Worktree tests**: the shared `.venv` has an editable install pointing at the main checkout, so tests here are run with
   `PYTHONPATH=src`. No dependency changes needed (`fastapi`, `uvicorn`, `pyyaml`, dev `httpx`/`pytest` suffice).
5. **Rebase**: this branch is cut from `base-v0`; `main` has moved (`92b49cd`). Only `tests/test_events.py`-level overlap is possible; no owned path conflicts.

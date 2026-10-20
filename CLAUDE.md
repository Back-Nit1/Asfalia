# Asfalia — project rules

Asfalia (ἀσφάλεια, "security, safety, certainty, firmness") is a synthetic hospital-security RL simulator. See README.md.

## HARD RULE: commit at least every 4 minutes
Every contributor — lead, specialist agents, humans — must commit work in progress **at least once every 4 minutes** of
active work (small, buildable-or-WIP commits on your own branch; never on `main` unless you are the lead). Do not batch up
long stretches of uncommitted changes. If a command or task will take longer than 4 minutes, commit before starting it.
Commit messages end with the session attribution line the harness supplies. This rule is repeated in every doc on purpose.

## Other rules
- Never read, print, log, or commit `.env` or any API key. `.env` is git-ignored.
- Lead owns `pyproject.toml`, lockfile, `events.py`, `schemas.py`, `cli.py`, engine, envs. Specialists request changes in `docs/requests/`.
- Synthetic data only; typed simulator actions only; no shell/SQL/network tools for learned policies.
- Report results faithfully; no invented numbers.

"""Owned-resource reconciliation and cleanup for local / Modal attempts.

Only attempts that this registry launched (``labels.owner == "asfalia"``) are ever candidates. Default is a dry run.
Never deleted: unresolved attempts (still running, or spawn outcome unknown), the newest terminal attempt of every
run (the resume point), attempts another unresolved attempt resumes from, anything younger than the TTL, and anything
without a registry record (reported as ``orphan``). Artifacts are verified against their manifest before deletion;
attempts with no manifest (failed / force-cancelled) are kept unless ``include_partial`` is set.
Modal calls cannot be enumerated through the SDK, so remote calls the registry does not know are not discoverable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

from hospital_rl.infra.backend import JobState
from hospital_rl.infra.bundle import OWNER_LABEL
from hospital_rl.infra.lifecycle import GatewayBackend

MANIFESTED = {JobState.DONE.value, JobState.DRAINED.value, JobState.STOPPED.value, JobState.TIMED_OUT.value}


def verify_remote(gw, run_id: str, attempt_id: str) -> list[str]:
    """Hash every manifest entry through the gateway (works for any store); empty list = intact."""
    raw = gw.read_file(run_id, attempt_id, "manifest.json")
    if raw is None:
        return ["manifest.json missing"]
    problems = []
    for name, want in json.loads(raw)["files"].items():
        data = gw.read_file(run_id, attempt_id, name)
        if data is None:
            problems.append(f"missing {name}")
        elif hashlib.sha256(data).hexdigest() != want["sha256"]:
            problems.append(f"changed {name}")
    return problems


def reconcile(backend: GatewayBackend, *, dry_run: bool = True, ttl_s: float = 7 * 86400, include_partial: bool = False,
              now: float | None = None) -> dict:
    now = time.time() if now is None else now
    gw, reg = backend.gw, backend.reg
    for rec in reg.all():  # refresh anything still believed live so we never judge from a stale record
        if not JobState(rec["state"]).terminal and rec["backend"] == backend.name:
            backend.status(rec["job_id"])
    recs = [r for r in reg.all() if r["backend"] == backend.name]
    newest = {}
    for r in recs:
        if JobState(r["state"]).terminal and (r["run_id"] not in newest or r["finished_at"] > newest[r["run_id"]]["finished_at"]):
            newest[r["run_id"]] = r
    pinned = {(r["run_id"], r["resume_from_attempt"]) for r in recs
              if r["resume_from_attempt"] and not JobState(r["state"]).terminal}
    actions = []

    def act(r, action, reason):
        actions.append({"job_id": r["job_id"], "action": action, "reason": reason})

    known = {(r["run_id"], r["attempt_id"]) for r in recs}
    for r in recs:
        run, att = r["run_id"], r["attempt_id"]
        if r["labels"].get("owner") != OWNER_LABEL["owner"]:
            act(r, "skip", "unowned: missing owner label")
        elif r.get("artifacts_deleted"):
            act(r, "skip", "already cleaned")
        elif not JobState(r["state"]).terminal:
            act(r, "keep", f"unresolved ({r['state']}): never deleted")
        elif newest[run] is r:
            act(r, "keep", "newest terminal attempt of its run (resume point)")
        elif (run, att) in pinned:
            act(r, "keep", "an unresolved attempt resumes from it")
        elif now - r["finished_at"] < ttl_s:
            act(r, "keep", f"younger than ttl ({int(now - r['finished_at'])}s < {int(ttl_s)}s)")
        elif r["state"] in MANIFESTED:
            if problems := verify_remote(gw, run, att):
                act(r, "keep", f"artifact verification failed, not deleting: {problems}")
            else:
                act(r, "delete", "manifest verified; older than ttl")
        elif include_partial:
            act(r, "delete", f"{r['state']}, no manifest; include_partial set")
        else:
            act(r, "keep", f"{r['state']}: partial artifacts kept (use include_partial)")
    deleted = []
    if not dry_run:
        for a in actions:
            if a["action"] == "delete":
                rec = reg.get(*a["job_id"].split(":", 2)[1:])
                gw.delete_attempt(rec["run_id"], rec["attempt_id"])
                reg.update(rec["run_id"], rec["attempt_id"], artifacts_deleted=True, cleaned_at=now)
                deleted.append(a["job_id"])
    orphans = [f"{r}/{a}" for r, a in gw.list_attempts() if (r, a) not in known]
    missing = [r["job_id"] for r in recs if not r.get("artifacts_deleted") and JobState(r["state"]).terminal
               and gw.read_file(r["run_id"], r["attempt_id"], "result.json") is None and r["state"] in MANIFESTED]
    return {"dry_run": dry_run, "actions": actions, "deleted": deleted, "orphans_not_deleted": orphans,
            "records_missing_artifacts": missing}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital_rl.infra.reconcile", description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=["local", "modal"], required=True)
    p.add_argument("--root", help="local artifact root (local backend)")
    p.add_argument("--config", help="cloud yaml (modal backend)")
    p.add_argument("--registry", help="registry directory (default: backend default)")
    p.add_argument("--ttl-hours", type=float, default=168)
    p.add_argument("--include-partial", action="store_true")
    p.add_argument("--apply", action="store_true", help="actually delete; default is a dry run")
    a = p.parse_args(argv)
    if a.backend == "local":
        from hospital_rl.infra.local_backend import LocalBackend
        be = LocalBackend(a.root, registry=a.registry)
    else:
        from hospital_rl.infra.modal_app import ModalBackend, RealModalGateway, load_modal_config
        cfg = load_modal_config(a.config)
        be = ModalBackend(cfg, RealModalGateway(cfg), a.registry)
    print(json.dumps(reconcile(be, dry_run=not a.apply, ttl_s=a.ttl_hours * 3600, include_partial=a.include_partial), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

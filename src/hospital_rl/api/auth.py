"""Token auth. Local binds are open for operator/attacker/defender views; the evaluator view always needs its own token."""
from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass
class AuthConfig:
    operator_token: str | None = None
    evaluator_token: str | None = None
    require_operator: bool = False     # True for any non-local bind

    @classmethod
    def for_bind(cls, host: str, operator_token: str | None, evaluator_token: str | None) -> "AuthConfig":
        local = host in LOCAL_HOSTS
        if not local and not operator_token:
            raise SystemExit(f"refusing to bind {host} without HOSPITAL_RL_OPERATOR_TOKEN")
        if operator_token and evaluator_token and hmac.compare_digest(operator_token, evaluator_token):
            raise SystemExit("operator and evaluator tokens must differ")
        return cls(operator_token, evaluator_token, require_operator=not local)


def _bearer(request: Request) -> str:
    h = request.headers.get("authorization", "")
    return h[7:] if h.lower().startswith("bearer ") else ""


def _eq(a: str, b: str | None) -> bool:
    return bool(b) and hmac.compare_digest(a.encode(), b.encode())


def require_operator(request: Request) -> None:
    cfg: AuthConfig = request.app.state.auth
    if cfg.require_operator and not _eq(_bearer(request), cfg.operator_token):
        raise HTTPException(401, "operator token required")


def require_evaluator(request: Request) -> None:
    cfg: AuthConfig = request.app.state.auth
    if not cfg.evaluator_token:
        raise HTTPException(503, "evaluator access not configured (set HOSPITAL_RL_EVALUATOR_TOKEN)")
    tok = _bearer(request)
    if not _eq(tok, cfg.evaluator_token):
        raise HTTPException(403 if _eq(tok, cfg.operator_token) else 401, "privileged evaluator token required")

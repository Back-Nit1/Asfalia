"""Local backend: the common lifecycle over a subprocess gateway (same semantics as Modal)."""
from __future__ import annotations

import sys
from pathlib import Path

from hospital_rl.infra.lifecycle import GatewayBackend, LocalGateway, Registry, ResourceCaps


class LocalBackend(GatewayBackend):
    def __init__(self, root: str | Path, python: str = sys.executable, env: dict[str, str] | None = None,
                 registry: str | Path | None = None, caps: ResourceCaps | None = None):
        root = Path(root)
        # default registry sits beside (not inside) the artifact root so manifests never see it
        super().__init__("local", LocalGateway(root, python, env), Registry(registry or Path(str(root) + ".registry")), caps)
        self.root = root

"""Separate response models per perspective. The attacker model has no field a defender or evaluator fact could ride in."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeView(_Base):
    id: str
    asset_type: str | None
    zone: str | None
    status: Literal["unknown", "reachable", "isolated", "degraded"]
    tags: list[str]


class Edge(_Base):
    src: str
    dst: str
    kind: str


class FeedEntry(_Base):
    idx: int
    tick: int
    type: str
    actor: str
    action: str | None
    target: str | None
    outcome: str
    duration: int | None
    scope: str | None


class Cumulative(_Base):
    reward: dict[str, float]
    components: dict[str, dict[str, float]]


class Snapshot(_Base):
    episode_id: str
    label: str                       # FIXTURE | RECORDED
    mode: Literal["REPLAY"]
    scenario: str
    tick: int
    last_tick: int
    nodes: list[NodeView]
    feed: list[FeedEntry]            # all feed entries up to and including `tick`
    reward: dict[str, float]
    cumulative: Cumulative
    topology_exported: bool          # false => node/zone layout unavailable (export has no topology block)


class AttackerSnapshot(Snapshot):
    perspective: Literal["attacker"] = "attacker"
    privileged: Literal[False] = False


class DefenderSnapshot(Snapshot):
    perspective: Literal["defender"] = "defender"
    privileged: Literal[False] = False
    edges: list[Edge]


class EvaluatorSnapshot(Snapshot):
    perspective: Literal["evaluator"] = "evaluator"
    privileged: Literal[True] = True
    notice: str = "PRIVILEGED: ground-truth evaluator view; never show to a policy"
    edges: list[Edge]
    final_metrics: dict


class Frame(_Base):
    tick: int
    nodes: list[NodeView]
    feed: list[FeedEntry]
    reward: dict[str, float]
    cumulative: Cumulative


class _Replay(_Base):
    episode_id: str
    label: str
    scenario: str
    topology_exported: bool
    frames: list[Frame]


class AttackerReplay(_Replay):
    perspective: Literal["attacker"] = "attacker"
    privileged: Literal[False] = False


class DefenderReplay(_Replay):
    perspective: Literal["defender"] = "defender"
    privileged: Literal[False] = False
    edges: list[Edge]


class EvaluatorReplay(_Replay):
    perspective: Literal["evaluator"] = "evaluator"
    privileged: Literal[True] = True
    notice: str = "PRIVILEGED: ground-truth evaluator view; never show to a policy"
    edges: list[Edge]
    final_metrics: dict


class EventsPage(_Base):
    perspective: str
    privileged: bool
    cursor: int
    events: list[FeedEntry]


SNAPSHOT = {"attacker": AttackerSnapshot, "defender": DefenderSnapshot, "evaluator": EvaluatorSnapshot}
REPLAY = {"attacker": AttackerReplay, "defender": DefenderReplay, "evaluator": EvaluatorReplay}

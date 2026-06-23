"""RobotSpec: unified per-(robot + end-effector) configuration object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

MimicRule = tuple[str, str, float, float]
HandBodyPredicate = Callable[[str], bool]
MjcfPatcher = Callable[[Path, float], Path]
QPosWriterFactory = Callable[[Any], Callable[[np.ndarray, np.ndarray], None]]


@dataclass(frozen=True)
class RobotSpec:
    """Immutable description of a robot + end-effector variant."""

    name: str
    display_name: str
    mjcf_path: Path
    action_dim: int
    action_joint_names: list[str]
    mimic_rules: list[MimicRule] = field(default_factory=list)
    hand_body_predicate: HandBodyPredicate = field(default=lambda _name: False)
    patch_mjcf: MjcfPatcher | None = None
    standing_height: float = 0.793
    head_mesh_names: tuple[str, ...] = ("head_link", "logo_link")
    left_shoulder_body: str = "left_shoulder_pitch_link"
    right_shoulder_body: str = "right_shoulder_pitch_link"
    floating_base_joint: str = "floating_base_joint"
    robot_type: str = ""
    modality: dict = field(default_factory=dict)
    hand_retargeter_cls: type | None = None
    qpos_writer_factory: QPosWriterFactory | None = None
    left_hand_clamp_joint_ids: np.ndarray | None = None
    right_hand_clamp_joint_ids: np.ndarray | None = None


_REGISTRY: dict[str, RobotSpec] = {}


def register(spec: RobotSpec) -> RobotSpec:
    """Register a spec by name."""
    _REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> RobotSpec:
    if name not in _REGISTRY:
        from phantom.robots import g1_dex3, g1_inspire  # noqa: F401
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown robot spec {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def all_specs() -> dict[str, RobotSpec]:
    """Return a copy of the registry."""
    from phantom.robots import g1_dex3, g1_inspire  # noqa: F401
    return dict(_REGISTRY)

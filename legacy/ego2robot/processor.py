"""Processor contract. Manager only sees name, artifacts, and resource kind."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class Artifacts:
    SOURCE = "source.json"
    ACTION = "action/eef_cam.npz"
    MASKS = "visual/masks.npz"
    INPAINT = "visual/inpaint.mp4"
    DEPTH = "visual/depth.u16.mp4"
    BASE = "robots/arx/base.json"
    IK = "robots/arx/ik.npz"
    ROBOT_RGB = "robots/arx/robot_rgb.mp4"
    ROBOT_MASK = "robots/arx/robot_mask.npz"
    COMPOSITE = "robots/arx/composite.mp4"
    L1 = "robots/arx/l1.json"


@dataclass(frozen=True)
class ResourceSpec:
    """cpu runs in-process. gpu goes through the single-GPU slot (serialized on a laptop)."""

    kind: str  # "cpu" | "gpu"
    gpu_name: str | None = None


CPU = ResourceSpec("cpu")
GPU_SAM3 = ResourceSpec("gpu", "sam3")
GPU_INPAINT = ResourceSpec("gpu", "inpaint")
GPU_CUROBO = ResourceSpec("gpu", "curobo")


@dataclass
class EpisodeContext:
    split: str
    task: str
    episode_id: str
    hdf5_path: Path
    video_path: Path | None
    out_dir: Path
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.split}/{self.task}/{self.episode_id}"


class Processor(Protocol):
    name: str
    requires: frozenset[str]
    produces: frozenset[str]
    optional_requires: frozenset[str]
    resources: ResourceSpec

    def run(self, ctx: EpisodeContext) -> None: ...

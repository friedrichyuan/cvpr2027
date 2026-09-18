"""Strict, small reader for the calibrated pose fields in an EgoDex episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


@dataclass(frozen=True)
class EgoDexEpisode:
    """Calibrated skeletal motion extracted from one EgoDex HDF5 episode.

    All transforms are stored exactly as supplied by EgoDex.  They are assumed
    to be homogeneous transforms whose translation is in the dataset world
    frame.  Mapping them into the ARX scene is deliberately a separate step.
    """

    path: Path
    intrinsic: np.ndarray
    world_T_camera: np.ndarray
    world_T_joint: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]

    @property
    def frame_count(self) -> int:
        return int(self.world_T_camera.shape[0])

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.world_T_joint)


def load_episode(path: str | Path) -> EgoDexEpisode:
    """Load camera calibration and tracked joint transforms from an episode.

    EgoDex test files contain ``camera/intrinsic`` and a ``transforms`` group
    whose members are ``(T, 4, 4)`` matrices, including ``camera``.  Fail early
    on a different layout: silently guessing units or pose conventions would
    make later retargeting errors very difficult to diagnose.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"EgoDex episode does not exist: {path}")

    with h5py.File(path, "r") as handle:
        try:
            intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
            transforms = handle["transforms"]
            world_T_camera = np.asarray(transforms["camera"], dtype=np.float64)
        except KeyError as error:
            raise KeyError(
                "Expected EgoDex keys 'camera/intrinsic', 'transforms/camera', "
                "and per-joint transforms."
            ) from error

        if intrinsic.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 camera intrinsic, got {intrinsic.shape}")
        _validate_transform_sequence("transforms/camera", world_T_camera)

        joints: dict[str, np.ndarray] = {}
        for name, dataset in transforms.items():
            if name == "camera":
                continue
            sequence = np.asarray(dataset, dtype=np.float64)
            _validate_transform_sequence(f"transforms/{name}", sequence)
            if sequence.shape[0] != world_T_camera.shape[0]:
                raise ValueError(
                    f"transforms/{name} has {sequence.shape[0]} frames, but camera "
                    f"has {world_T_camera.shape[0]}"
                )
            joints[name] = sequence

        if not joints:
            raise ValueError("EgoDex episode contains no tracked body joints")
        metadata = {key: _normalise_attribute(value) for key, value in handle.attrs.items()}

    return EgoDexEpisode(
        path=path,
        intrinsic=intrinsic,
        world_T_camera=world_T_camera,
        world_T_joint=joints,
        metadata=metadata,
    )


def _validate_transform_sequence(name: str, transforms: np.ndarray) -> None:
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected {name} to have shape (T, 4, 4), got {transforms.shape}")
    if transforms.shape[0] == 0:
        raise ValueError(f"{name} contains no frames")
    if not np.isfinite(transforms).all():
        raise ValueError(f"{name} contains non-finite values")


def _normalise_attribute(value: object) -> object:
    """Turn HDF5 byte attributes into ordinary Python values for display."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return tuple(_normalise_attribute(item) for item in value.tolist())
    return value

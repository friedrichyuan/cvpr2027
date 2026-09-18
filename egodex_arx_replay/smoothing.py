"""Temporal smoothing for geometric gripper reference trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .gripper import GripperTrajectory


@dataclass(frozen=True)
class SmoothingConfig:
    """Episode-independent smoothing parameters at EgoDex's 30 Hz frame rate."""

    window: int = 9
    polyorder: int = 2
    orientation_sigma: float = 2.0

    def __post_init__(self) -> None:
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError("window must be an odd integer of at least 3")
        if not 0 <= self.polyorder < self.window:
            raise ValueError("polyorder must be non-negative and smaller than window")
        if self.orientation_sigma <= 0:
            raise ValueError("orientation_sigma must be positive")


def smooth_gripper_trajectory(
    trajectory: GripperTrajectory,
    config: SmoothingConfig = SmoothingConfig(),
) -> GripperTrajectory:
    """Smooth positions/openings with Savitzky-Golay and rotations with SLERP.

    This mirrors the two target-space operations used by Ego2Robot: polynomial
    filtering for Euclidean signals and a local, Gaussian-weighted spherical
    interpolation for orientations.  Each contiguous valid segment is treated
    independently, so missing-hand intervals never interpolate across a gap.
    """
    position = trajectory.position.copy()
    rotation = trajectory.rotation.copy()
    width = trajectory.width.copy()

    for side in range(trajectory.valid.shape[1]):
        for start, end in _valid_segments(trajectory.valid[:, side]):
            length = end - start
            local_window = _usable_window(config.window, config.polyorder, length)
            if local_window is not None:
                position[start:end, side] = savgol_filter(
                    position[start:end, side],
                    window_length=local_window,
                    polyorder=config.polyorder,
                    axis=0,
                    mode="interp",
                )
                width[start:end, side] = np.maximum(
                    0.0,
                    savgol_filter(
                        width[start:end, side],
                        window_length=local_window,
                        polyorder=config.polyorder,
                        mode="interp",
                    ),
                )
            rotation[start:end, side] = _smooth_rotations(
                rotation[start:end, side],
                sigma=config.orientation_sigma,
                radius=config.window // 2,
            )

    return GripperTrajectory(position=position, rotation=rotation, width=width, valid=trajectory.valid.copy())


def _valid_segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals of consecutive valid frames."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(valid)))
    return segments


def _usable_window(requested: int, polyorder: int, length: int) -> int | None:
    window = min(requested, length if length % 2 else length - 1)
    return window if window > polyorder else None


def _smooth_rotations(rotations: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    """Apply a local Gaussian-weighted running mean using incremental SLERP."""
    quaternions = Rotation.from_matrix(rotations).as_quat()
    out = np.empty_like(rotations)
    for center in range(len(rotations)):
        indices = list(range(max(0, center - radius), min(len(rotations), center + radius + 1)))
        # Starting at the center avoids a chronological bias in the incremental
        # mean and handles the quaternion double-cover consistently.
        indices.sort(key=lambda index: abs(index - center))
        mean = quaternions[center].copy()
        accumulated_weight = 0.0
        for index in indices:
            candidate = quaternions[index].copy()
            if np.dot(mean, candidate) < 0.0:
                candidate *= -1.0
            weight = float(np.exp(-0.5 * ((index - center) / sigma) ** 2))
            if accumulated_weight == 0.0:
                mean = candidate
                accumulated_weight = weight
                continue
            mean = _slerp(mean, candidate, weight / (accumulated_weight + weight))
            accumulated_weight += weight
        out[center] = Rotation.from_quat(mean).as_matrix()
    return out


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate two unit quaternions, returning a unit quaternion."""
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    return (
        np.sin((1.0 - fraction) * angle) / sin_angle * first
        + np.sin(fraction * angle) / sin_angle * second
    )

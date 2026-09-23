"""Hand keypoints to a smoothed camera-frame parallel gripper."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from egowhale.step import GRIPPER, Step

FPS = 30.0
WINDOW = 21
POLY = 3
SIGMA = 10.0
_TIPS = ("Hand", "ThumbTip", "IndexFingerTip", "MiddleFingerTip")


class Retarget(Step):
    name = "retarget"
    makes = (GRIPPER,)

    def run(self, src: Path, dst: Path) -> None:
        src = Path(src)
        position, rotation, width, valid, intrinsic = _episode(src)
        position, rotation, width = _smooth(position, rotation, width, valid)
        path = Path(dst) / GRIPPER
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            position=position.astype(np.float32),
            rotation=rotation.astype(np.float32),
            width=width.astype(np.float32),
            valid=valid,
            intrinsic=intrinsic.astype(np.float32),
            fps=np.float32(FPS),
        )


def _episode(path: Path):
    with h5py.File(path, "r") as handle:
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        transforms = handle["transforms"]
        camera = np.asarray(transforms["camera"], dtype=np.float64)
        joints = {
            name: np.asarray(dataset, dtype=np.float64)
            for name, dataset in transforms.items()
            if name != "camera"
        }
    frames = camera.shape[0]
    position = np.full((frames, 2, 3), np.nan, dtype=np.float64)
    rotation = np.full((frames, 2, 3, 3), np.nan, dtype=np.float64)
    width = np.full((frames, 2), np.nan, dtype=np.float64)
    valid = np.zeros((frames, 2), dtype=bool)
    for frame in range(frames):
        cam_from_world = _invert(camera[frame])
        points = {name: value[frame, :3, 3] for name, value in joints.items()}
        for side, name in enumerate(("left", "right")):
            pose = _hand(points, name)
            if pose is None:
                continue
            world_p, world_r, opening = pose
            position[frame, side] = cam_from_world[:3, :3] @ world_p + cam_from_world[:3, 3]
            rotation[frame, side] = cam_from_world[:3, :3] @ world_r
            width[frame, side] = opening
            valid[frame, side] = True
    return position, rotation, width, valid, intrinsic


def _hand(points: dict[str, np.ndarray], side: str):
    names = [f"{side}{tip}" for tip in _TIPS]
    if any(name not in points for name in names):
        return None
    wrist, thumb, index, middle = (points[name] for name in names)
    virtual = 0.7 * index + 0.3 * middle
    jaw = thumb - virtual
    opening = float(np.linalg.norm(jaw))
    if opening < 1e-4:
        return None
    sign = 1.0 if side == "right" else -1.0
    grasp = sign * jaw / opening
    normal = np.cross(grasp, virtual - wrist)
    norm = np.linalg.norm(normal)
    if norm < 1e-4:
        return None
    normal /= norm
    approach = np.cross(normal, grasp)
    return 0.5 * (thumb + virtual), np.column_stack((approach, normal, grasp)), opening


def _invert(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _smooth(position, rotation, width, valid):
    position, rotation, width = position.copy(), rotation.copy(), width.copy()
    for side in range(2):
        for start, end in _segments(valid[:, side]):
            window = _window(end - start)
            if window is not None:
                position[start:end, side] = savgol_filter(position[start:end, side], window, POLY, axis=0)
                width[start:end, side] = np.maximum(0.0, savgol_filter(width[start:end, side], window, POLY))
            rotation[start:end, side] = _slerp_smooth(rotation[start:end, side])
    return position, rotation, width


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    spans, start = [], None
    for index, flag in enumerate(valid):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(valid)))
    return spans


def _window(length: int) -> int | None:
    window = min(WINDOW, length if length % 2 else length - 1)
    return window if window > POLY else None


def _slerp_smooth(rotations: np.ndarray) -> np.ndarray:
    quats = Rotation.from_matrix(rotations).as_quat()
    radius = WINDOW // 2
    out = np.empty_like(rotations)
    for center in range(len(rotations)):
        indices = range(max(0, center - radius), min(len(rotations), center + radius + 1))
        mean = quats[center].copy()
        weight_sum = 0.0
        for index in sorted(indices, key=lambda item: abs(item - center)):
            sample = quats[index].copy()
            if np.dot(mean, sample) < 0.0:
                sample *= -1.0
            weight = float(np.exp(-0.5 * ((index - center) / SIGMA) ** 2))
            if weight_sum == 0.0:
                mean, weight_sum = sample, weight
                continue
            mean = _slerp(mean, sample, weight / (weight_sum + weight))
            weight_sum += weight
        out[center] = Rotation.from_quat(mean).as_matrix()
    return out


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if dot < 0.0:
        second, dot = -second, -dot
    if dot > 0.9995:
        mixed = first + fraction * (second - first)
        return mixed / np.linalg.norm(mixed)
    angle = np.arccos(dot)
    return (
        np.sin((1.0 - fraction) * angle) * first + np.sin(fraction * angle) * second
    ) / np.sin(angle)

"""Packed masks and short RGB clips shared by the visual steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_masks(path: Path, masks: np.ndarray) -> None:
    masks = np.asarray(masks, dtype=bool)
    packed = np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=packed, shape=np.array(masks.shape, dtype=np.int32))


def load_masks(path: Path) -> np.ndarray:
    with np.load(path) as data:
        shape = tuple(int(value) for value in data["shape"])
        packed = data["packed"]
    flat = np.unpackbits(packed, axis=1)[:, : shape[1] * shape[2]]
    return flat.reshape(shape).astype(bool)


def read_rgb(path: Path) -> tuple[np.ndarray, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames), fps


def write_rgb(path: Path, frames: np.ndarray, fps: float) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

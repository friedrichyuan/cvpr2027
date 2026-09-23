"""Load/save packed binary masks and short mp4 clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_masks(path: Path, masks: np.ndarray) -> None:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks (T, H, W), got {masks.shape}")
    packed = np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=packed, shape=np.array(masks.shape, dtype=np.int32))


def load_masks(path: Path) -> np.ndarray:
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        packed = data["packed"]
    flat = np.unpackbits(packed, axis=1)[:, : shape[1] * shape[2]]
    return flat.reshape(shape).astype(bool)


def read_rgb_video(path: Path) -> tuple[np.ndarray, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"Video has no frames: {path}")
    return np.stack(frames), fps


def write_rgb_video(path: Path, frames: np.ndarray, fps: float) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write video: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

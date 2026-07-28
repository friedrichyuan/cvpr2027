from __future__ import annotations

import numpy as np


def parse_resolution(value: str | None) -> tuple[float, float] | None:
    if not value or "x" not in value:
        return None
    left, right = value.lower().split("x", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return None


def scale_intrinsics_to_shape(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    source_width: float,
    source_height: float,
    shape: tuple[int, int, int],
) -> np.ndarray:
    h, w = shape[:2]
    sx = float(w) / max(float(source_width), 1e-6)
    sy = float(h) / max(float(source_height), 1e-6)
    return np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def project_camera_points(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-4)
    uv = np.full((len(points), 2), -100000, dtype=np.int32)
    if valid.any():
        pts = points[valid]
        uv_float = np.stack(
            [K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2], K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]],
            axis=1,
        )
        uv[valid] = np.round(uv_float).astype(np.int32)
    return uv, valid


def bbox_area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_overlap(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_center(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)

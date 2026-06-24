# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Cam-local shoulder synthesis for AoE (neck-mounted camera).

AoE has ZERO body keypoints — only MANO hands + camera trajectory.
We exploit the rigidity of the neck mount: the camera hangs ~20 cm below
the shoulder midpoint, creating a constant offset in camera-local frame.

CRITICAL: The y-component MUST be negative (OpenCV +Y = down, camera is
BELOW shoulder, so shoulder is at negative y). A positive value causes
the "hands above head" failure mode documented in aoe_to_g1 v1→v3 history.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from phantom.constants import G1_ARM_LENGTH
from phantom.constants.aoe import (
    SHOULDER_HALF_WIDTH,
    SHOULDER_OFFSET_FROM_CAMERA_CAM,
)


def synth_shoulders_cam(
    n_frames: int,
    offset: np.ndarray = SHOULDER_OFFSET_FROM_CAMERA_CAM,
    half_width: float = SHOULDER_HALF_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize left/right shoulder positions in camera-local frame.

    Returns:
        (left_shoulder (T, 3), right_shoulder (T, 3)) in cam-local coords.
    """
    offset = np.asarray(offset, dtype=np.float64)
    left = np.tile(offset + np.array([-half_width, 0, 0]), (n_frames, 1))
    right = np.tile(offset + np.array([+half_width, 0, 0]), (n_frames, 1))
    return left.astype(np.float64), right.astype(np.float64)


def refine_shoulder_offset_to_arm_length(
    left_wrist_pos: np.ndarray,
    right_wrist_pos: np.ndarray,
    init_offset: np.ndarray = SHOULDER_OFFSET_FROM_CAMERA_CAM,
    init_half_width: float = SHOULDER_HALF_WIDTH,
    target_arm_length: float = G1_ARM_LENGTH * 1.5,
    valid_left: np.ndarray | None = None,
    valid_right: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Optional per-episode 4-parameter refinement of shoulder offset.

    Fits (dx, dy, dz, half_width) to minimize |p95(arm_length) - target|.
    Hard constraint: dy < 0 (camera is always below shoulder).

    Returns:
        (refined_offset (3,), refined_half_width)
    """
    T = left_wrist_pos.shape[0]
    if valid_left is None:
        valid_left = np.ones(T, dtype=bool)
    if valid_right is None:
        valid_right = np.ones(T, dtype=bool)

    x0 = np.array([init_offset[0], init_offset[1], init_offset[2], init_half_width])

    def _loss(params):
        dx, dy, dz, hw = params
        if dy >= 0:
            return 1e6
        offset = np.array([dx, dy, dz])
        ls = offset + np.array([-hw, 0, 0])
        rs = offset + np.array([+hw, 0, 0])

        dists = []
        for t in range(T):
            if valid_left[t]:
                dists.append(np.linalg.norm(left_wrist_pos[t] - ls))
            if valid_right[t]:
                dists.append(np.linalg.norm(right_wrist_pos[t] - rs))
        if not dists:
            return 1e6
        p95 = np.percentile(dists, 95)
        return (p95 - target_arm_length) ** 2

    result = minimize(_loss, x0, method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 0.001, "fatol": 1e-6})
    dx, dy, dz, hw = result.x
    dy = min(dy, -0.05)
    return np.array([dx, dy, dz], dtype=np.float64), float(hw)

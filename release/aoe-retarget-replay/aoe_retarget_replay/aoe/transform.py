"""AOE cam-local → G1 base coordinate transform.

Converts episode wrist SE(3) poses from OpenCV camera-local frame
to the G1 robot base frame. Uses episode-level 95th-percentile
arm-length scaling (single scalar, preserves bimanual geometry).
"""

from __future__ import annotations

import numpy as np

from aoe_retarget_replay.constants import (
    ARM_LENGTH_PERCENTILE,
    G1_ARM_LENGTH,
    MIN_ARM_LENGTH_SAMPLE,
)
from aoe_retarget_replay.constants.aoe import (
    AOE_T_ALIGN_LEFT_WRIST,
    AOE_T_ALIGN_RIGHT_WRIST,
    R_AOE_CAM_TO_G1,
)


def transform_aoe_wrist_poses(
    left_wrist_poses: np.ndarray,
    right_wrist_poses: np.ndarray,
    left_shoulder_pos: np.ndarray,
    right_shoulder_pos: np.ndarray,
    g1_left_shoulder: np.ndarray,
    g1_right_shoulder: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Transform AoE cam-local wrist poses to G1 base frame.

    Args:
        left_wrist_poses:  (T, 4, 4) SE(3) in AOE cam-local frame
        right_wrist_poses: (T, 4, 4) SE(3) in AOE cam-local frame
        left_shoulder_pos:  (T, 3) synthesized shoulder in cam-local
        right_shoulder_pos: (T, 3) synthesized shoulder in cam-local
        g1_left_shoulder:  (3,) G1 left shoulder position
        g1_right_shoulder: (3,) G1 right shoulder position

    Returns:
        (left_wrist_g1 (T, 4, 4), right_wrist_g1 (T, 4, 4), scale)
    """
    T = left_wrist_poses.shape[0]
    R = R_AOE_CAM_TO_G1

    shoulder_mid_cam = 0.5 * (left_shoulder_pos + right_shoulder_pos)
    g1_shoulder_mid = 0.5 * (g1_left_shoulder + g1_right_shoulder)

    left_pos_cam = left_wrist_poses[:, :3, 3]
    right_pos_cam = right_wrist_poses[:, :3, 3]

    left_rel = left_pos_cam - shoulder_mid_cam
    right_rel = right_pos_cam - shoulder_mid_cam

    left_dist = np.linalg.norm(left_pos_cam - left_shoulder_pos, axis=1)
    right_dist = np.linalg.norm(right_pos_cam - right_shoulder_pos, axis=1)

    all_dists = np.concatenate([
        left_dist[left_dist > MIN_ARM_LENGTH_SAMPLE],
        right_dist[right_dist > MIN_ARM_LENGTH_SAMPLE],
    ])
    if len(all_dists) == 0:
        scale = 1.0
    else:
        p95 = np.percentile(all_dists, ARM_LENGTH_PERCENTILE)
        scale = G1_ARM_LENGTH / p95 if p95 > 0 else 1.0

    left_wrist_g1 = np.zeros((T, 4, 4), dtype=np.float64)
    right_wrist_g1 = np.zeros((T, 4, 4), dtype=np.float64)
    left_wrist_g1[:, 3, 3] = 1.0
    right_wrist_g1[:, 3, 3] = 1.0

    for t in range(T):
        left_wrist_g1[t, :3, 3] = g1_shoulder_mid + R @ (left_rel[t] * scale)
        right_wrist_g1[t, :3, 3] = g1_shoulder_mid + R @ (right_rel[t] * scale)

        R_left = left_wrist_poses[t, :3, :3]
        R_right = right_wrist_poses[t, :3, :3]
        left_wrist_g1[t, :3, :3] = R @ R_left @ AOE_T_ALIGN_LEFT_WRIST
        right_wrist_g1[t, :3, :3] = R @ R_right @ AOE_T_ALIGN_RIGHT_WRIST

    return left_wrist_g1, right_wrist_g1, float(scale)

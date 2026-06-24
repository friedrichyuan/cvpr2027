# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
MANO Hand Parameters → GR00T N1.7 Actions

Two hand representation modes (EEF-first layout, matching NVIDIA GR00T convention):
  sharpa   (62D):  [left_wrist_eef(9), right_wrist_eef(9), left_hand_joints(22), right_hand_joints(22)]
  gripper  (20D):  [left_wrist_eef(9), right_wrist_eef(9), left_gripper(1), right_gripper(1)]

Input: hands.npz with fields:
    - pred_rot: (2, T, 3) axis-angle global rotation [left=0, right=1]
    - pred_trans: (2, T, 3) translation in world frame (meters)
    - pred_hand_pose: (2, T, 45) finger joint poses (15 joints × 3 axis-angle)
    - pred_valid: (2, T) validity mask
"""

import numpy as np
from scipy.spatial.transform import Rotation


def axis_angle_to_rot_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to rotation matrix (3, 3)."""
    return Rotation.from_rotvec(axis_angle).as_matrix()


def rot_matrix_to_rot6d(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (3,3) to 6D representation (first two rows flattened)."""
    return rot_matrix[:2, :].flatten()


def rot6d_to_rot_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation back to 3x3 rotation matrix via Gram-Schmidt."""
    r = rot6d.reshape(2, 3)
    row1 = r[0] / np.linalg.norm(r[0])
    row2 = r[1] - np.dot(row1, r[1]) * row1
    row2 = row2 / np.linalg.norm(row2)
    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3])


def build_homogeneous(rot_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = translation
    return T


def extract_wrist_poses(
    pred_rot: np.ndarray,
    pred_trans: np.ndarray,
    pred_valid: np.ndarray,
    hand_idx: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract wrist 6D pose from MANO global rotation + translation.

    Args:
        pred_rot: (2, T, 3) axis-angle global hand rotation
        pred_trans: (2, T, 3) hand translation (meters)
        pred_valid: (2, T) validity mask
        hand_idx: 0=left, 1=right

    Returns:
        positions: (T, 3) wrist positions in world frame
        rot_matrices: (T, 3, 3) wrist orientations
        valid_mask: (T,) boolean mask
    """
    T = pred_rot.shape[1]
    positions = pred_trans[hand_idx]  # (T, 3)
    valid_mask = pred_valid[hand_idx].astype(bool)  # (T,)

    rot_matrices = np.zeros((T, 3, 3))
    for t in range(T):
        if valid_mask[t]:
            rot_matrices[t] = axis_angle_to_rot_matrix(pred_rot[hand_idx, t])
        else:
            rot_matrices[t] = np.eye(3)

    return positions, rot_matrices, valid_mask


def compute_gripper_from_mano_pose(
    pred_hand_pose: np.ndarray,
    pred_valid: np.ndarray,
    hand_idx: int = 1,
) -> np.ndarray:
    """
    Infer gripper open/close state from MANO finger joint angles.

    Uses the curl of thumb (joints 0-2), index (joints 3-5), and
    middle finger (joints 6-8) to estimate grasp state.

    MANO joint layout (15 joints × 3 axis-angle each = 45D):
        joints 0-2: thumb (CMC, MCP, IP)
        joints 3-5: index (MCP, PIP, DIP)
        joints 6-8: middle
        joints 9-11: ring
        joints 12-14: pinky

    Args:
        pred_hand_pose: (2, T, 45) finger joint axis-angles
        pred_valid: (2, T) validity mask
        hand_idx: 0=left, 1=right

    Returns:
        gripper: (T, 1) grasp state normalized to [0, 1]
    """
    T = pred_hand_pose.shape[1]
    valid = pred_valid[hand_idx].astype(bool)
    hand_pose = pred_hand_pose[hand_idx]  # (T, 45)

    # Index finger (joints 3-5): axis-angle indices 9:18
    index_angles = np.linalg.norm(hand_pose[:, 9:18].reshape(T, 3, 3), axis=2)
    index_curl = index_angles.sum(axis=1)

    # Middle finger (joints 6-8): axis-angle indices 18:27
    middle_angles = np.linalg.norm(hand_pose[:, 18:27].reshape(T, 3, 3), axis=2)
    middle_curl = middle_angles.sum(axis=1)

    # Thumb (joints 0-2): axis-angle indices 0:9
    thumb_angles = np.linalg.norm(hand_pose[:, 0:9].reshape(T, 3, 3), axis=2)
    thumb_curl = thumb_angles.sum(axis=1)

    # Combined grasp metric
    grasp_metric = (index_curl + middle_curl + thumb_curl) / 3.0

    # Normalize using valid frames only
    valid_metrics = grasp_metric[valid]
    if len(valid_metrics) == 0:
        return np.zeros((T, 1))

    d_min = np.percentile(valid_metrics, 5)
    d_max = np.percentile(valid_metrics, 95)

    gripper = np.clip((grasp_metric - d_min) / (d_max - d_min + 1e-6), 0, 1)
    gripper[~valid] = 0.0

    return gripper.reshape(-1, 1)


def wrist_to_eef_9d(
    positions: np.ndarray,
    rot_matrices: np.ndarray,
) -> np.ndarray:
    """
    Convert wrist position + rotation to EEF 9D representation.

    Args:
        positions: (T, 3) xyz
        rot_matrices: (T, 3, 3) rotation matrices

    Returns:
        eef_9d: (T, 9) [x, y, z, rot6d(6)]
    """
    T = positions.shape[0]
    eef_9d = np.zeros((T, 9))
    eef_9d[:, :3] = positions
    for t in range(T):
        eef_9d[t, 3:9] = rot_matrix_to_rot6d(rot_matrices[t])
    return eef_9d


def interpolate_invalid_frames(
    valid_mask: np.ndarray,
    *arrays: np.ndarray,
    max_gap: int = 10,
) -> tuple:
    """
    Linearly interpolate short gaps of invalid frames across all provided arrays.
    Gaps longer than max_gap are left invalid (will be split into separate episodes).

    Args:
        valid_mask: (T,) boolean
        *arrays: any number of (T, D) arrays to interpolate
        max_gap: maximum gap length to interpolate

    Returns:
        (valid_out, arr1_out, arr2_out, ...) — same order as input
    """
    T = len(valid_mask)
    valid_out = valid_mask.copy()
    arr_outs = [a.copy() for a in arrays]

    i = 0
    while i < T:
        if not valid_out[i]:
            j = i
            while j < T and not valid_out[j]:
                j += 1

            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < T:
                for k in range(i, j):
                    alpha = (k - i + 1) / (gap_len + 1)
                    for arr in arr_outs:
                        arr[k] = (1 - alpha) * arr[i - 1] + alpha * arr[j]
                    valid_out[k] = True
            i = j
        else:
            i += 1

    return (valid_out, *arr_outs)


def _process_single_hand(
    data: dict,
    hand_idx: int,
    max_interp_gap: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process one hand from loaded npz data.

    Returns:
        eef_9d: (T, 9)
        gripper: (T, 1)
        valid_mask: (T,) bool
    """
    pred_rot = data["pred_rot"]
    pred_trans = data["pred_trans"]
    pred_valid = data["pred_valid"]
    pred_hand_pose = data["pred_hand_pose"]

    positions, rot_matrices, valid_mask = extract_wrist_poses(
        pred_rot, pred_trans, pred_valid, hand_idx
    )
    eef_9d = wrist_to_eef_9d(positions, rot_matrices)
    gripper = compute_gripper_from_mano_pose(pred_hand_pose, pred_valid, hand_idx)

    valid_mask, eef_9d, gripper = interpolate_invalid_frames(
        valid_mask, eef_9d, gripper, max_gap=max_interp_gap
    )

    return eef_9d, gripper, valid_mask


def process_hands_npz(
    hands_npz_path: str,
    hand_idx: int = 1,
    target_fps: int = 15,
    source_fps: int = 30,
    max_interp_gap: int = 10,
) -> dict:
    """
    Single-hand pipeline: hands.npz → GR00T-ready EEF data.

    Returns:
        dict with keys: "eef_9d", "gripper", "valid_mask", "num_frames", "fps"
    """
    data = np.load(hands_npz_path)

    eef_9d, gripper, valid_mask = _process_single_hand(data, hand_idx, max_interp_gap)

    # Downsample
    step = max(1, source_fps // target_fps)
    if step > 1:
        eef_9d = eef_9d[::step]
        gripper = gripper[::step]
        valid_mask = valid_mask[::step]

    return {
        "eef_9d": eef_9d.astype(np.float32),
        "gripper": gripper.astype(np.float32),
        "valid_mask": valid_mask,
        "num_frames": len(eef_9d),
        "fps": source_fps // step,
    }


def process_hands_npz_gripper(
    hands_npz_path: str,
    target_fps: int = 15,
    source_fps: int = 30,
    max_interp_gap: int = 10,
) -> dict:
    """
    Bimanual pipeline: hands.npz → GR00T-ready dual-EEF data.

    State layout: [left_wrist_eef(9), right_wrist_eef(9), left_gripper(1), right_gripper(1)] = 20D

    Returns:
        dict with keys:
            - "left_wrist_eef": (T', 9)
            - "left_gripper": (T', 1)
            - "right_wrist_eef": (T', 9)
            - "right_gripper": (T', 1)
            - "valid_mask": (T',) bool — intersection of both hands
            - "num_frames": int
            - "fps": int
    """
    data = np.load(hands_npz_path)

    left_eef, left_grip, left_valid = _process_single_hand(data, hand_idx=0, max_interp_gap=max_interp_gap)
    right_eef, right_grip, right_valid = _process_single_hand(data, hand_idx=1, max_interp_gap=max_interp_gap)

    # Combined validity: both hands must be valid
    valid_mask = left_valid & right_valid

    # Downsample
    step = max(1, source_fps // target_fps)
    if step > 1:
        left_eef = left_eef[::step]
        left_grip = left_grip[::step]
        right_eef = right_eef[::step]
        right_grip = right_grip[::step]
        valid_mask = valid_mask[::step]

    return {
        "left_wrist_eef": left_eef.astype(np.float32),
        "left_gripper": left_grip.astype(np.float32),
        "right_wrist_eef": right_eef.astype(np.float32),
        "right_gripper": right_grip.astype(np.float32),
        "valid_mask": valid_mask,
        "num_frames": len(left_eef),
        "fps": source_fps // step,
    }


def _process_hands_npz_retarget(
    hands_npz_path: str,
    retarget_npz_path: str,
    left_key: str,
    right_key: str,
    target_fps: int = 15,
    source_fps: int = 30,
    max_interp_gap: int = 10,
) -> dict:
    """
    Generic retarget pipeline: hands.npz + retarget.npz → bimanual EEF + joint angles.

    Args:
        left_key: key in retarget npz for left hand joints (e.g. "left_sharpa_joints")
        right_key: key in retarget npz for right hand joints

    Returns:
        dict with keys: left_wrist_eef, {left_key}, right_wrist_eef, {right_key},
                        valid_mask, num_frames, fps
    """
    data = np.load(hands_npz_path)
    retarget = np.load(retarget_npz_path)

    pred_rot = data["pred_rot"]
    pred_trans = data["pred_trans"]
    pred_valid = data["pred_valid"]

    left_joints = retarget[left_key]
    right_joints = retarget[right_key]

    left_pos, left_rot, left_valid = extract_wrist_poses(pred_rot, pred_trans, pred_valid, 0)
    right_pos, right_rot, right_valid = extract_wrist_poses(pred_rot, pred_trans, pred_valid, 1)
    left_eef = wrist_to_eef_9d(left_pos, left_rot)
    right_eef = wrist_to_eef_9d(right_pos, right_rot)

    valid_mask = left_valid & right_valid

    valid_mask, left_eef, left_joints = interpolate_invalid_frames(
        valid_mask, left_eef, left_joints, max_gap=max_interp_gap
    )
    _, right_eef, right_joints = interpolate_invalid_frames(
        valid_mask, right_eef, right_joints, max_gap=max_interp_gap
    )

    step = max(1, source_fps // target_fps)
    if step > 1:
        left_eef = left_eef[::step]
        left_joints = left_joints[::step]
        right_eef = right_eef[::step]
        right_joints = right_joints[::step]
        valid_mask = valid_mask[::step]

    return {
        "left_wrist_eef": left_eef.astype(np.float32),
        left_key: left_joints.astype(np.float32),
        "right_wrist_eef": right_eef.astype(np.float32),
        right_key: right_joints.astype(np.float32),
        "valid_mask": valid_mask,
        "num_frames": len(left_eef),
        "fps": source_fps // step,
    }


def process_hands_npz_sharpa(
    hands_npz_path: str,
    retarget_npz_path: str,
    target_fps: int = 15,
    source_fps: int = 30,
    max_interp_gap: int = 10,
) -> dict:
    """
    Sharpa retarget pipeline: hands.npz + hands_retargeted_sharpa.npz → bimanual EEF + Sharpa joints.
    State layout: [left_wrist_eef(9), right_wrist_eef(9), left_hand_joints(22), right_hand_joints(22)] = 62D
    """
    return _process_hands_npz_retarget(
        hands_npz_path, retarget_npz_path,
        "left_hand_joints", "right_hand_joints",
        target_fps, source_fps, max_interp_gap,
    )


def split_by_validity(
    valid_mask: np.ndarray,
    min_segment_len: int = 30,
    **arrays: np.ndarray,
) -> list[dict]:
    """
    Split trajectory into continuous valid segments.
    Segments shorter than min_segment_len are discarded.

    Args:
        valid_mask: (T,) boolean
        min_segment_len: minimum segment length
        **arrays: named arrays to slice (each shape (T, ...))

    Returns:
        list of dicts, each containing sliced arrays + "start_idx" + "end_idx"
    """
    segments = []
    T = len(valid_mask)
    i = 0

    while i < T:
        if valid_mask[i]:
            j = i
            while j < T and valid_mask[j]:
                j += 1
            if (j - i) >= min_segment_len:
                seg = {"start_idx": i, "end_idx": j}
                for name, arr in arrays.items():
                    seg[name] = arr[i:j]
                segments.append(seg)
            i = j
        else:
            i += 1

    return segments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mano_to_eef.py <hands.npz> [bimanual]")
        sys.exit(1)

    npz_path = sys.argv[1]
    bimanual = len(sys.argv) > 2 and sys.argv[2] == "bimanual"

    if bimanual:
        result = process_hands_npz_gripper(npz_path)
        print(f"Bimanual: {result['num_frames']} frames at {result['fps']} FPS")
        print(f"  Left EEF:  {result['left_wrist_eef'].shape}")
        print(f"  Right EEF: {result['right_wrist_eef'].shape}")
        print(f"  Valid (both hands): {result['valid_mask'].sum() / len(result['valid_mask']):.2%}")

        segments = split_by_validity(
            result["valid_mask"],
            left_wrist_eef=result["left_wrist_eef"],
            right_wrist_eef=result["right_wrist_eef"],
        )
        print(f"  Valid segments (>=30 frames): {len(segments)}")
    else:
        result = process_hands_npz(npz_path, hand_idx=1)
        print(f"Single hand (right): {result['num_frames']} frames at {result['fps']} FPS")
        print(f"  EEF 9D shape: {result['eef_9d'].shape}")
        print(f"  Valid ratio: {result['valid_mask'].sum() / len(result['valid_mask']):.2%}")

        segments = split_by_validity(
            result["valid_mask"],
            eef_9d=result["eef_9d"],
            gripper=result["gripper"],
        )
        print(f"  Valid segments (>=30 frames): {len(segments)}")
        for i, seg in enumerate(segments):
            print(f"    Segment {i}: frames {seg['start_idx']}-{seg['end_idx']} "
                  f"({len(seg['eef_9d'])} frames)")

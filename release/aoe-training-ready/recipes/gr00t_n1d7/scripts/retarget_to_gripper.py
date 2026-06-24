# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Retarget MANO 21 keypoints → Gripper open/close value.

Computes normalized thumb-to-index fingertip distance as gripper aperture.
Simpler than dexterous hand retarget — just a scalar per hand per frame.

Gripper value:
  0.0 = fully closed (thumb touching index)
  1.0 = fully open (max distance observed)

Output: hands_retargeted_gripper.npz with:
  - left_gripper: (T, 1) normalized [0, 1]
  - right_gripper: (T, 1) normalized [0, 1]
  - pred_valid: (2, T)

Usage:
    cd scripts
    python retarget_to_gripper.py \
        --data-root /path/to/poc_deliver \
        --keypoints-dir ../output \
        --output-dir ../output \
        --max-episodes 2
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import ema_smooth

# Keypoint indices (OpenPose 21-kp ordering)
KP_THUMB_CMC = 1
KP_THUMB_TIP = 4
KP_INDEX_MCP = 5
KP_INDEX_TIP = 8


def compute_gripper_from_keypoints(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    ema_alpha: float = 0.3,
) -> np.ndarray:
    """
    Compute gripper aperture from thumb-index fingertip distance,
    normalized by max distance at 90° opening angle.

    gripper = ||thumb_tip - index_tip|| / max_distance_at_90deg

    where max_distance_at_90deg = sqrt(thumb_length² + index_length²)
      thumb_length = ||thumb_tip - thumb_CMC||
      index_length = ||index_tip - index_MCP||

    Args:
        keypoints: (T, 21, 3) keypoints for one hand
        valid_mask: (T,) boolean

    Returns:
        gripper: (T, 1) normalized to [0, 1]
            0.0 = closed (thumb touching index)
            1.0 = fully open (90° spread)
    """
    T = len(keypoints)
    thumb_tip = keypoints[:, KP_THUMB_TIP]    # (T, 3)
    index_tip = keypoints[:, KP_INDEX_TIP]    # (T, 3)
    thumb_cmc = keypoints[:, KP_THUMB_CMC]    # (T, 3)
    index_mcp = keypoints[:, KP_INDEX_MCP]    # (T, 3)

    # Actual thumb-index distance
    ti_distance = np.linalg.norm(thumb_tip - index_tip, axis=1)  # (T,)

    # Finger lengths
    thumb_length = np.linalg.norm(thumb_tip - thumb_cmc, axis=1)  # (T,)
    index_length = np.linalg.norm(index_tip - index_mcp, axis=1)  # (T,)

    # Max distance at 90° opening: hypotenuse
    max_distance = np.sqrt(thumb_length**2 + index_length**2)
    max_distance = np.where(max_distance < 1e-4, 1e-4, max_distance)

    # Normalize
    gripper = np.clip(ti_distance / max_distance, 0.0, 1.0)
    gripper[~valid_mask] = 0.0

    # EMA smoothing
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 1:
        gripper[valid_indices] = ema_smooth(
            gripper[valid_indices].reshape(-1, 1), alpha=ema_alpha
        ).flatten()

    return gripper.reshape(-1, 1).astype(np.float32)


def convert_single(keypoints_path: str, output_path: str, ema_alpha: float = 0.3) -> dict:
    """Convert one hands_keypoints.npz → hands_retargeted_gripper.npz."""
    data = np.load(keypoints_path)
    joints_world = data["joints_world"]  # (2, T, 21, 3)
    pred_valid = data["pred_valid"]      # (2, T)

    left_valid = pred_valid[0] > 0.5
    right_valid = pred_valid[1] > 0.5

    left_gripper = compute_gripper_from_keypoints(joints_world[0], left_valid, ema_alpha)
    right_gripper = compute_gripper_from_keypoints(joints_world[1], right_valid, ema_alpha)

    np.savez_compressed(
        output_path,
        left_gripper=left_gripper,      # (T, 1)
        right_gripper=right_gripper,    # (T, 1)
        pred_valid=pred_valid,
    )

    T = joints_world.shape[1]
    both_valid = left_valid & right_valid
    return {"frames": T, "valid_ratio": float(both_valid.sum() / T)}


def main():
    parser = argparse.ArgumentParser(
        description="Retarget MANO 21 keypoints → Gripper aperture (thumb-index distance)"
    )
    parser.add_argument("--data-root", type=str,
                        default="/media/hdd4tb/sankuai/code/07_具身/dataset/07_Ego/poc_deliver")
    parser.add_argument("--keypoints-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.keypoints_dir:
        kp_root = Path(args.keypoints_dir)
        episodes = sorted([
            d for d in data_root.iterdir()
            if d.is_dir()
            and (kp_root / d.name / "hands_keypoints.npz").exists()
        ])
    else:
        episodes = sorted([
            d for d in data_root.iterdir()
            if d.is_dir()
            and (d / "ego_process" / "ego_hands_reconstruction" / "hands_keypoints.npz").exists()
        ])

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"=== Retarget → Gripper (thumb-index distance) ===")
    print(f"  Episodes: {len(episodes)}")
    print(f"  EMA alpha: {args.ema_alpha}")
    print()

    success = 0
    skipped = 0
    failed = []

    for ep_dir in tqdm(episodes, desc="Retargeting"):
        if args.keypoints_dir:
            kp_path = Path(args.keypoints_dir) / ep_dir.name / "hands_keypoints.npz"
        else:
            kp_path = ep_dir / "ego_process" / "ego_hands_reconstruction" / "hands_keypoints.npz"

        if args.output_dir:
            out_path = Path(args.output_dir) / ep_dir.name / "hands_retargeted_gripper.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = kp_path.parent / "hands_retargeted_gripper.npz"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            stats = convert_single(str(kp_path), str(out_path), ema_alpha=args.ema_alpha)
            success += 1
        except Exception as e:
            failed.append(f"{ep_dir.name}: {e}")

    print(f"\nDone: {success} converted, {skipped} skipped, {len(failed)} failed")
    if failed:
        for f in failed[:10]:
            print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()

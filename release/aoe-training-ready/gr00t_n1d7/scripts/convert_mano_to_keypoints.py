# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Batch convert hands.npz (MANO params) → hands_keypoints.npz (21 3D keypoints).

Uses smplx.MANOLayer directly for FK (no HaWoR dependency).

Input:  hands.npz with pred_rot, pred_trans, pred_hand_pose, pred_betas, pred_valid, R_w2c, t_w2c
Output: hands_keypoints.npz with joints_world (2, T, 21, 3), joints_cam (2, T, 21, 3), pred_valid

21 keypoints are in OpenPose hand order:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, TIP)
    5-8: Index (MCP, PIP, DIP, TIP)
    9-12: Middle (MCP, PIP, DIP, TIP)
    13-16: Ring (MCP, PIP, DIP, TIP)
    17-20: Pinky (MCP, PIP, DIP, TIP)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import smplx
from smplx.utils import to_tensor

# MANO model weights — local directory first, then shared assets/mano/
MANO_MODEL_DIR = Path(__file__).resolve().parent / "mano_models"
SHARED_MANO_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "assets" / "mano"
if not (MANO_MODEL_DIR / "MANO_RIGHT.pkl").exists() and (SHARED_MANO_DIR / "MANO_RIGHT.pkl").exists():
    MANO_MODEL_DIR = SHARED_MANO_DIR
MANO_RIGHT_PATH = MANO_MODEL_DIR
MANO_LEFT_PATH = MANO_MODEL_DIR

# MANO internal joint order → OpenPose hand order
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

# smplx vertex IDs for fingertips (from smplx.vertex_ids['mano'])
FINGERTIP_VERTEX_IDS = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky


def _create_mano(is_right: bool, device: torch.device):
    """Create MANO model for left or right hand."""
    model_path = str(MANO_RIGHT_PATH if is_right else MANO_LEFT_PATH)
    mano = smplx.MANOLayer(
        model_path=model_path,
        gender="neutral",
        num_hand_joints=15,
        create_body_pose=False,
        is_rhand=is_right,
    )
    # Fix left hand shapedirs bug: https://github.com/vchoutas/smplx/issues/48
    if not is_right:
        mano.shapedirs[:, 0, :] *= -1

    mano = mano.to(device)
    mano.eval()
    return mano


def _aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (N, 3) → rotation matrix (N, 3, 3)."""
    angle = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    K = torch.zeros(*aa.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    R = I + sin_a * K + (1 - cos_a) * (K @ K)
    return R


def _run_mano_fk(mano, trans, rot, hand_pose, betas, device):
    """Run MANO FK and return 21 OpenPose keypoints.

    Args:
        trans: (T, 3) translation
        rot: (T, 3) global rotation axis-angle
        hand_pose: (T, 45) finger joint axis-angles
        betas: (T, 10) shape params

    Returns:
        joints: (T, 21, 3) in world frame
    """
    T = trans.shape[0]

    global_orient = _aa_to_rotmat(rot.to(device)).view(T, 1, 3, 3)
    hand_pose_rotmat = _aa_to_rotmat(hand_pose.reshape(T * 15, 3).to(device)).view(T, 15, 3, 3)

    with torch.no_grad():
        output = mano(
            global_orient=global_orient,
            hand_pose=hand_pose_rotmat,
            betas=betas.to(device),
            transl=trans.to(device),
            pose2rot=False,
        )

    # Standard MANO joints (16) + extra fingertip vertices
    joints_mano = output.joints  # (T, 16, 3)
    vertices = output.vertices   # (T, 778, 3)
    fingertips = vertices[:, FINGERTIP_VERTEX_IDS, :]  # (T, 5, 3)

    # Concatenate and reorder to OpenPose
    all_joints = torch.cat([joints_mano, fingertips], dim=1)  # (T, 21, 3)
    joints_openpose = all_joints[:, MANO_TO_OPENPOSE, :]  # (T, 21, 3)

    return joints_openpose.cpu().numpy()


def convert_single(hands_npz_path: str, output_path: str, device: torch.device) -> dict:
    """Convert one hands.npz → hands_keypoints.npz."""
    data = np.load(hands_npz_path, allow_pickle=True)

    pred_betas = data["pred_betas"]          # (2, T, 10)
    pred_hand_pose = data["pred_hand_pose"]  # (2, T, 45)
    pred_rot = data["pred_rot"]              # (2, T, 3)
    pred_trans = data["pred_trans"]          # (2, T, 3)
    pred_valid = data["pred_valid"]          # (2, T)
    R_w2c = data["R_w2c"]                    # (T, 3, 3)
    t_w2c = data["t_w2c"]                    # (T, 3)

    # Create left and right MANO models
    mano_left = _create_mano(is_right=False, device=device)
    mano_right = _create_mano(is_right=True, device=device)

    T = pred_rot.shape[1]
    joints_world = np.zeros((2, T, 21, 3), dtype=np.float32)

    # Left hand (index 0)
    joints_world[0] = _run_mano_fk(
        mano_left,
        torch.from_numpy(pred_trans[0]).float(),
        torch.from_numpy(pred_rot[0]).float(),
        torch.from_numpy(pred_hand_pose[0]).float(),
        torch.from_numpy(pred_betas[0]).float(),
        device,
    )

    # Right hand (index 1)
    joints_world[1] = _run_mano_fk(
        mano_right,
        torch.from_numpy(pred_trans[1]).float(),
        torch.from_numpy(pred_rot[1]).float(),
        torch.from_numpy(pred_hand_pose[1]).float(),
        torch.from_numpy(pred_betas[1]).float(),
        device,
    )

    # World → camera
    joints_cam = np.zeros_like(joints_world)
    for hand_idx in range(2):
        joints_cam[hand_idx] = np.einsum(
            "nij,nkj->nki", R_w2c, joints_world[hand_idx]
        ) + t_w2c[:, np.newaxis, :]

    # Zero out invalid frames
    for hand_idx in range(2):
        invalid = pred_valid[hand_idx] < 0.5
        joints_world[hand_idx, invalid] = 0.0
        joints_cam[hand_idx, invalid] = 0.0

    np.savez_compressed(
        output_path,
        joints_world=joints_world,
        joints_cam=joints_cam,
        pred_valid=pred_valid,
        R_c2w=data["R_c2w"],
        t_c2w=data["t_c2w"],
        R_w2c=R_w2c,
        t_w2c=t_w2c,
        focal=data["focal"],
    )

    valid_both = (pred_valid[0] > 0.5) & (pred_valid[1] > 0.5)
    return {"frames": T, "valid_ratio": float(valid_both.sum() / T)}


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert MANO params → 21 keypoints (smplx MANOLayer FK)"
    )
    parser.add_argument(
        "--data-root", type=str,
        default="/media/hdd4tb/sankuai/code/07_具身/dataset/07_Ego/poc_deliver",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output dir. Writes <output-dir>/<episode>/hands_keypoints.npz. "
             "If not set, writes in-place next to source.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)

    episodes = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and (d / "ego_process" / "ego_hands_reconstruction" / "hands.npz").exists()
    ])
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"=== MANO FK → 21 Keypoints (smplx direct) ===")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Device: {device}")
    if args.output_dir:
        print(f"  Output: {args.output_dir}")
    print()

    success = 0
    skipped = 0
    failed = []

    for ep_dir in tqdm(episodes, desc="Converting"):
        hands_path = ep_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"

        if args.output_dir:
            output_path = Path(args.output_dir) / ep_dir.name / "hands_keypoints.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = hands_path.parent / "hands_keypoints.npz"

        if output_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            stats = convert_single(str(hands_path), str(output_path), device)
            success += 1
        except Exception as e:
            failed.append(f"{ep_dir.name}: {e}")

    print(f"\nDone: {success} converted, {skipped} skipped, {len(failed)} failed")
    if failed:
        for f in failed[:10]:
            print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()

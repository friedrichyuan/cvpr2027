#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pre-compute action data from AoE MANO data and segment by annotations.

Converts raw episode data (hands.npz + annotations + undistorted video) into
per-segment HDF5 files ready for H-RDT training.

Output structure:
  $OUTPUT_ROOT/
    <episode_name>/
      0.hdf5   (segment 0)
      0.mp4    (copy of undistorted video)
      1.hdf5   (segment 1)
      1.mp4    ...

Each HDF5 contains:
  - actions: (T_seg, action_dim) precomputed action vectors
  - pred_valid: (2, T_seg) hand validity flags
  - metadata attrs: segment info, action format description

Action: 48D = left(24) + right(24), where 24 = pos(3) + rot6d(6) + fingertips(15)

Uses HaWoR's run_mano_twohands directly for MANO FK (guaranteed numerical
consistency with the golden reference convert/convert_mano_to_keypoints.py).

Usage:
  python H_RDT/precompute_actions.py --data_root /path/to/AoE 数据集 --output_root /path/to/output
  python H_RDT/precompute_actions.py --verify  # verify projection on sample frames
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

import torch

# Patch numpy for chumpy compatibility (numpy>=1.24 removed np.bool etc.)
_NP_COMPAT = {'bool': np.bool_, 'int': np.int_, 'float': np.float64,
              'complex': np.complex128, 'object': np.object_, 'unicode': np.str_,
              'str': np.str_}
for _k, _v in _NP_COMPAT.items():
    if not hasattr(np, _k):
        setattr(np, _k, _v)

# Use HaWoR directly (same as convert/convert_mano_to_keypoints.py)
HAWOR_ROOT = Path(os.path.dirname(os.path.abspath(__file__))) / 'HaWoR'
sys.path.insert(0, str(HAWOR_ROOT))
_prev_cwd = os.getcwd()
os.chdir(str(HAWOR_ROOT))

from hawor.utils.process import run_mano_twohands  # noqa: E402

os.chdir(_prev_cwd)

FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # OpenPose ordering: thumb, index, middle, ring, pinky


# ======================== MANO FK (via HaWoR) ========================

def run_mano_fk_twohands(hands_data, start_frame, end_frame):
    """
    Run MANO FK for both hands via HaWoR's run_mano_twohands.

    Args:
        hands_data: loaded hands.npz dict
        start_frame: start index
        end_frame: end index

    Returns:
        joints_world: (2, T, 21, 3) OpenPose-ordered joints in world frame
    """
    trans_t = torch.from_numpy(
        hands_data['pred_trans'][:, start_frame:end_frame].copy()).float()       # (2, T, 3)
    rot_t = torch.from_numpy(
        hands_data['pred_rot'][:, start_frame:end_frame].copy()).float()         # (2, T, 3)
    pose_t = torch.from_numpy(
        hands_data['pred_hand_pose'][:, start_frame:end_frame].copy()).float()   # (2, T, 45)
    betas_t = torch.from_numpy(
        hands_data['pred_betas'][:, start_frame:end_frame].copy()).float()       # (2, T, 10)

    with torch.no_grad():
        outputs = run_mano_twohands(
            trans_t, rot_t, pose_t,
            is_right=None, init_betas=betas_t,
            use_cuda=torch.cuda.is_available(),
        )

    joints_world = outputs['joints'][:, :, :21, :].cpu().numpy()  # (2, T, 21, 3)
    return joints_world.astype(np.float32)


def joints_world_to_cam(joints_world, R_w2c, t_w2c):
    """
    Transform joints from world frame to camera frame.

    Args:
        joints_world: (T, K, 3) joints in world frame
        R_w2c: (T, 3, 3) rotation matrices
        t_w2c: (T, 3) translation vectors

    Returns:
        joints_cam: (T, K, 3) joints in camera frame
    """
    joints_cam = np.einsum('nij,nkj->nki', R_w2c, joints_world) + t_w2c[:, np.newaxis, :]
    return joints_cam.astype(np.float32)


ACTION_DIM = 48  # 2 hands x (3 + 6 + 15)
ACTION_FORMAT = {
    'total_dim': 48,
    'format': 'left_hand(24d) + right_hand(24d)',
    'hand_format': 'translation(3d) + rotation_6d(6d) + fingertips(15d)',
    'fingertips_detail': '5 fingertips (thumb,index,middle,ring,pinky) xyz, OpenPose order, world frame',
    'mano_pipeline': 'HaWoR run_mano_twohands (direct call, not reimplemented)',
}


# ======================== SEGMENT PROCESSING ========================

def build_segment_actions(hands_data, start_frame, end_frame):
    """
    Build the full action vector for a segment.

    Action = left_hand(24d) + right_hand(24d) = 48D
    Per hand: translation(3d) + rotation_6d(6d) + fingertips(15d)

    Returns:
        actions: (T, 48) float32
        valid: (2, T) float32
    """
    T = end_frame - start_frame
    if T <= 0:
        return None, None

    # Run MANO FK once for both hands -> (2, T, 21, 3)
    joints_world = run_mano_fk_twohands(hands_data, start_frame, end_frame)

    per_hand_actions = []
    for hand_idx in range(2):  # 0=left, 1=right
        trans = hands_data['pred_trans'][hand_idx, start_frame:end_frame].astype(np.float32)  # (T, 3)

        axis_angle = hands_data['pred_rot'][hand_idx, start_frame:end_frame]  # (T, 3)
        R = Rotation.from_rotvec(axis_angle).as_matrix()  # (T, 3, 3)
        rot_6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1).astype(np.float32)  # (T, 6)

        fingertips = joints_world[hand_idx, :, FINGERTIP_INDICES, :]  # (T, 5, 3)
        fingertips = fingertips.reshape(-1, 15).astype(np.float32)  # (T, 15)

        per_hand_actions.append(np.concatenate([trans, rot_6d, fingertips], axis=-1))

    actions = np.concatenate(per_hand_actions, axis=-1)  # (T, 48)
    valid = hands_data['pred_valid'][:, start_frame:end_frame]  # (2, T)

    assert actions.shape == (T, 48), f"Expected ({T}, 48), got {actions.shape}"

    return actions, valid


def process_episode(episode_dir, output_dir, min_segment_frames=16):
    """
    Process one episode: read annotations, split into segments, save HDF5 per segment.

    Returns:
        (num_segments_saved, num_segments_skipped)
    """
    import shutil

    episode_name = os.path.basename(episode_dir)

    # Load annotation
    ann_path = os.path.join(episode_dir, 'ego_annotation/ego_action_annotation.json')
    if not os.path.exists(ann_path):
        return 0, 0

    with open(ann_path, 'r') as f:
        annotations = json.load(f)

    if not annotations:
        return 0, 0

    # Load MANO data
    hands_path = os.path.join(episode_dir,
                              'ego_process/ego_hands_reconstruction/hands.npz')
    if not os.path.exists(hands_path):
        return 0, 0

    hands_data = np.load(hands_path)
    total_frames = hands_data['pred_trans'].shape[1]

    # Video path (undistorted)
    video_path = os.path.join(episode_dir,
                              'ego_process/ego_undistorted_video/raw_video_undistorted.mp4')
    if not os.path.exists(video_path):
        return 0, 0

    episode_output_dir = os.path.join(output_dir, episode_name)
    os.makedirs(episode_output_dir, exist_ok=True)

    saved = 0
    skipped = 0

    for seg_idx, seg in enumerate(annotations):
        start_frame = int(seg['start_frame'])
        end_frame = int(seg['end_frame'])

        # Clamp to available data
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total_frames)

        seg_len = end_frame - start_frame
        if seg_len < min_segment_frames:
            skipped += 1
            continue

        # Check validity: skip segment if either hand has any invalid frame
        valid = hands_data['pred_valid'][:, start_frame:end_frame]  # (2, T)
        if not np.all(valid):
            skipped += 1
            continue

        # Build actions
        actions, _ = build_segment_actions(hands_data, start_frame, end_frame)
        if actions is None:
            skipped += 1
            continue

        # Extract description from annotation
        print(actions.shape)
        description = _build_description(seg)
        print(description)

        # Save HDF5
        import h5py
        hdf5_path = os.path.join(episode_output_dir, f"{seg_idx}.hdf5")
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('actions', data=actions, compression='gzip', compression_opts=4)

            f.attrs['llm_description'] = description
            f.attrs['episode_name'] = episode_name
            f.attrs['segment_id'] = seg_idx
            f.attrs['start_frame'] = start_frame
            f.attrs['end_frame'] = end_frame
            f.attrs['total_episode_frames'] = total_frames
            f.attrs['action_dim'] = ACTION_DIM

            fmt = ACTION_FORMAT
            for k, v in fmt.items():
                f.attrs[f'action_{k}'] = str(v)

        # Copy video
        mp4_dst = os.path.join(episode_output_dir, f"{seg_idx}.mp4")
        if not os.path.exists(mp4_dst):
            shutil.copy2(video_path, mp4_dst)

        # Save segment frame range for the dataset loader
        meta_path = os.path.join(episode_output_dir, f"{seg_idx}.meta.json")
        with open(meta_path, 'w') as f:
            json.dump({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'fps': 30,
                'description': description,
            }, f)

        saved += 1

    return saved, skipped


def _build_description(seg):
    """Build a natural language description from an annotation segment."""
    actions = seg.get('atomic_action', [])
    if not actions:
        return f"Performing a manipulation task in the {seg.get('scene', 'scene')}."

    descriptions = []
    for act in actions:
        desc = act.get('description', '')
        if desc:
            descriptions.append(desc)
        else:
            verb = act.get('verb', 'manipulate')
            obj = act.get('object', 'object')
            hand = act.get('hand', 'hand')
            descriptions.append(f"Use {hand} hand to {verb} the {obj}.")

    return ' '.join(descriptions)


# ======================== VERIFICATION ========================

def verify_projection(data_root, num_episodes=2, num_frames_per_episode=5, output_dir=None):
    """
    Verify fingertip extraction by projecting joints_cam onto video frames.
    Saves visualization images for manual inspection.
    """
    import cv2

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verify_projection')
    os.makedirs(output_dir, exist_ok=True)

    HAND_SKELETON = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    FINGER_COLORS = [
        (255, 128, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255),
    ]
    HAND_COLORS = [(0, 200, 255), (255, 100, 100)]  # left=yellow, right=blue

    episodes = sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if d.startswith('poc_raw_video') and os.path.isdir(os.path.join(data_root, d))
    ])[:num_episodes]

    for ep_dir in episodes:
        ep_name = os.path.basename(ep_dir)
        hands_path = os.path.join(ep_dir, 'ego_process/ego_hands_reconstruction/hands.npz')
        video_path = os.path.join(ep_dir, 'ego_process/ego_undistorted_video/raw_video_undistorted.mp4')
        info_path = os.path.join(ep_dir, 'ego_process/ego_undistorted_video/undistorted_video_info.json')

        if not all(os.path.exists(p) for p in [hands_path, video_path, info_path]):
            print(f"Skipping {ep_name}: missing files")
            continue

        hands_data = np.load(hands_path)
        with open(info_path) as f:
            cam_info = json.load(f)['cameraParams']

        focal = float(hands_data['focal'])
        img_w = cam_info.get('image_width', int(cam_info['resolution'].split('x')[0]))
        img_h = cam_info.get('image_height', int(cam_info['resolution'].split('x')[1]))
        cx, cy = img_w / 2.0, img_h / 2.0

        total_frames = hands_data['pred_trans'].shape[1]
        R_w2c = hands_data['R_w2c']  # (T, 3, 3)
        t_w2c = hands_data['t_w2c']  # (T, 3)
        pred_valid = hands_data['pred_valid']  # (2, T)

        # Compute full 21 joints in world via HaWoR, then transform to cam
        joints_world_both = run_mano_fk_twohands(hands_data, 0, total_frames)  # (2, T, 21, 3)

        joints_cam_both = []
        for hand_idx in range(2):
            jc = joints_world_to_cam(joints_world_both[hand_idx], R_w2c, t_w2c)
            joints_cam_both.append(jc)

        # Sample frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames_per_episode, dtype=int)

        cap = cv2.VideoCapture(video_path)
        ep_out_dir = os.path.join(output_dir, ep_name)
        os.makedirs(ep_out_dir, exist_ok=True)

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                continue

            for hand_idx in range(2):
                if pred_valid[hand_idx, fi] < 0.5:
                    continue
                joints_3d = joints_cam_both[hand_idx][fi]  # (21, 3)

                # Project: pinhole
                z = joints_3d[:, 2]
                z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
                px = focal * joints_3d[:, 0] / z_safe + cx
                py = focal * joints_3d[:, 1] / z_safe + cy
                joints_2d = np.stack([px, py], axis=-1)

                # Draw skeleton
                for bone_i, (s, e) in enumerate(HAND_SKELETON):
                    pt1 = joints_2d[s].astype(int)
                    pt2 = joints_2d[e].astype(int)
                    color = FINGER_COLORS[bone_i // 4]
                    cv2.line(frame, tuple(pt1), tuple(pt2), color, 2, cv2.LINE_AA)

                # Draw joints
                for ji, pt in enumerate(joints_2d):
                    center = tuple(pt.astype(int))
                    r = 5 if ji == 0 else 3
                    cv2.circle(frame, center, r, HAND_COLORS[hand_idx], -1, cv2.LINE_AA)

                # Label
                wrist = joints_2d[0].astype(int)
                label = "L" if hand_idx == 0 else "R"
                cv2.putText(frame, label, (wrist[0] - 10, wrist[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, HAND_COLORS[hand_idx], 2)

            cv2.putText(frame, f"frame {fi}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(ep_out_dir, f"frame_{fi:06d}.jpg"), frame)

        cap.release()
        print(f"Verified {ep_name}: {len(frame_indices)} frames -> {ep_out_dir}")

    print(f"\nVerification images saved to: {output_dir}")


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute actions from AoE 数据集 data and segment by annotations')
    parser.add_argument('--data_root', type=str,
                        default='/root/datasets/AoE_demo_data/AoE 数据集',
                        help='Root directory of AoE 数据集 data')
    parser.add_argument('--output_root', type=str,
                        default='/root/datasets/AoE_processed_data',
                        help='Output directory (default: H_RDT/processed_data)')
    parser.add_argument('--min_segment_frames', type=int, default=16,
                        help='Minimum frames per segment (skip shorter ones)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing output')
    parser.add_argument('--verify', action='store_true',
                        help='Run projection verification on sample frames')
    parser.add_argument('--verify_episodes', type=int, default=2,
                        help='Number of episodes to verify')
    parser.add_argument('--verify_frames', type=int, default=5,
                        help='Number of frames per episode to verify')
    args = parser.parse_args()

    if args.verify:
        verify_projection(
            args.data_root,
            num_episodes=args.verify_episodes,
            num_frames_per_episode=args.verify_frames,
        )
        return

    if args.output_root is None:
        args.output_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'processed_data')

    print(f"Action representation: {ACTION_DIM}D")
    print(f"  Format: {ACTION_FORMAT}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Min segment frames: {args.min_segment_frames}")
    print("=" * 60)

    # Find all episodes
    episodes = sorted([
        os.path.join(args.data_root, d)
        for d in os.listdir(args.data_root)
        if d.startswith('poc_raw_video') and os.path.isdir(os.path.join(args.data_root, d))
    ])

    print(f"Found {len(episodes)} episodes")

    total_saved = 0
    total_skipped = 0
    failed_episodes = []

    for i, ep_dir in enumerate(episodes):
        ep_name = os.path.basename(ep_dir)
        try:
            saved, skipped = process_episode(
                ep_dir, args.output_root,
                min_segment_frames=args.min_segment_frames,
            )
            total_saved += saved
            total_skipped += skipped
            print(f"[{i+1}/{len(episodes)}] {ep_name}: saved={saved}, skipped={skipped}")
        except Exception as e:
            failed_episodes.append((ep_name, str(e)))
            print(f"[{i+1}/{len(episodes)}] {ep_name}: FAILED - {e}")

    print("\n" + "=" * 60)
    print(f"Done! Segments saved: {total_saved}, skipped: {total_skipped}")
    if failed_episodes:
        print(f"Failed episodes ({len(failed_episodes)}):")
        for name, err in failed_episodes[:10]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()

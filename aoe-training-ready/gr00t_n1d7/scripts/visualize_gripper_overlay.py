# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Gripper visualization: thumb-index distance overlay on ego video.

Draws a line between thumb tip and index tip, color-coded by gripper state:
  Green = open, Red = closed, with distance value displayed.

Output: output/<episode>/vis_retargeted_gripper_overlay.mp4

Usage:
    cd scripts
    python visualize_gripper_overlay.py \
        --data-root /path/to/poc_deliver \
        --keypoints-dir ../output \
        --output-dir ../output \
        --max-episodes 1 --max-frames 300
"""

import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from utils import (
    discover_episodes,
    draw_hand_skeleton,
    find_motion_start,
    get_intrinsics,
    project_to_pixel,
)

KP_THUMB_TIP = 4
KP_INDEX_TIP = 8


def draw_gripper_distance(frame, joints_2d, gripper_value):
    """Draw thumb-index distance line with color gradient and value label."""
    h, w = frame.shape[:2]
    thumb = tuple(joints_2d[KP_THUMB_TIP].astype(int))
    index = tuple(joints_2d[KP_INDEX_TIP].astype(int))

    if not (0 <= thumb[0] < w and 0 <= thumb[1] < h):
        return
    if not (0 <= index[0] < w and 0 <= index[1] < h):
        return

    # Color: bright cyan (open) → bright red (closed)
    r = int(255 * (1 - gripper_value))
    g = int(255 * gripper_value)
    b = int(200 * gripper_value)
    color = (r, g, b)

    cv2.line(frame, thumb, index, color, 4, cv2.LINE_AA)
    cv2.circle(frame, thumb, 6, (255, 100, 100), -1, cv2.LINE_AA)
    cv2.circle(frame, index, 6, (100, 255, 255), -1, cv2.LINE_AA)

    # Label with background for readability
    mid = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2 - 12)
    cv2.putText(frame, f"{gripper_value:.2f}", mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"{gripper_value:.2f}", mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)


def process_episode(
    hands_npz_path, keypoints_path, retarget_path,
    video_path, output_path, video_info_path,
    max_frames=300, frame_step=2,
):
    """Generate gripper overlay video for one episode."""
    hands = np.load(hands_npz_path, allow_pickle=True)
    kp_data = np.load(keypoints_path)
    rt_data = np.load(retarget_path)

    pred_trans = hands["pred_trans"]
    pred_valid = hands["pred_valid"]
    R_w2c = hands["R_w2c"]
    t_w2c = hands["t_w2c"]

    joints_cam = kp_data["joints_cam"]  # (2, T, 21, 3)
    kp_valid = kp_data["pred_valid"].astype(bool)

    left_gripper = rt_data["left_gripper"].flatten()   # (T,)
    right_gripper = rt_data["right_gripper"].flatten()  # (T,)

    fx, fy, cx, cy = get_intrinsics(hands, video_info_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    T_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T_data = min(T_total, joints_cam.shape[1])

    start_frame = find_motion_start(pred_trans, pred_valid, R_w2c, t_w2c)

    for _ in range(start_frame):
        cap.read()

    out_frames = []
    t = start_frame

    while t < T_data and len(out_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if (t - start_frame) % frame_step == 0:
            ego_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for hand_idx in range(2):
                if kp_valid[hand_idx, t]:
                    pts_2d = project_to_pixel(joints_cam[hand_idx, t], fx, fy, cx, cy)

                    # Draw skeleton (per-finger colors)
                    draw_hand_skeleton(ego_rgb, pts_2d)

                    # Draw gripper distance line
                    gval = left_gripper[t] if hand_idx == 0 else right_gripper[t]
                    draw_gripper_distance(ego_rgb, pts_2d, gval)

            out_frames.append(ego_rgb)

        t += 1

    cap.release()

    if out_frames:
        iio.imwrite(output_path, np.stack(out_frames),
                    fps=fps / frame_step, codec="h264")

    return len(out_frames)


def main():
    parser = argparse.ArgumentParser(
        description="Gripper visualization: thumb-index distance overlay"
    )
    parser.add_argument("--data-root", type=str,
                        default="/media/hdd4tb/sankuai/code/07_具身/dataset/07_Ego/poc_deliver")
    parser.add_argument("--keypoints-dir", type=str, default=None)
    parser.add_argument("--retarget-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="../output")
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frame-step", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    episodes = discover_episodes(args.data_root)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"=== Gripper Distance Overlay ===")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Max frames: {args.max_frames}, step: {args.frame_step}")
    print()

    for i, ep in enumerate(episodes):
        name = ep["name"]
        ep_out = output_dir / name
        ep_out.mkdir(parents=True, exist_ok=True)
        out_path = str(ep_out / "vis_retargeted_gripper_overlay.mp4")

        # Find keypoints
        kp_path = None
        if args.keypoints_dir:
            p = Path(args.keypoints_dir) / name / "hands_keypoints.npz"
            if p.exists():
                kp_path = str(p)
        if not kp_path:
            p = Path(args.data_root) / name / "ego_process" / "ego_hands_reconstruction" / "hands_keypoints.npz"
            if p.exists():
                kp_path = str(p)

        # Find retarget
        rt_path = None
        if args.retarget_dir:
            p = Path(args.retarget_dir) / name / "hands_retargeted_gripper.npz"
            if p.exists():
                rt_path = str(p)
        if not rt_path:
            p = Path(args.data_root) / name / "ego_process" / "ego_hands_reconstruction" / "hands_retargeted_gripper.npz"
            if p.exists():
                rt_path = str(p)

        if not kp_path or not rt_path:
            print(f"  [{i+1}/{len(episodes)}] {name}: SKIP")
            continue

        print(f"  [{i+1}/{len(episodes)}] {name} ...", end=" ", flush=True)
        try:
            n = process_episode(
                ep["hands_npz"], kp_path, rt_path,
                ep["video_path"], out_path, ep["video_info"],
                max_frames=args.max_frames, frame_step=args.frame_step,
            )
            print(f"OK ({n} frames)")
        except Exception as e:
            print(f"FAILED ({e})")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Output: <output>/<episode>/vis_retargeted_gripper_overlay.mp4")


if __name__ == "__main__":
    main()

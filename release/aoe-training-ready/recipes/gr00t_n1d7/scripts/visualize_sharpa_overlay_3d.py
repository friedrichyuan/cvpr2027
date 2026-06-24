# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
AR Overlay: MuJoCo Sharpa hands composited directly onto ego video.

Renders the Sharpa Wave dual hands at full ego resolution with camera-matched
projection, then alpha-composites onto the ego video frames. The result is
an AR-like visualization where the robotic hands appear in-place on the human hands.

Output: output/<episode>/vis_retargeted_sharpa_overlay.mp4

Usage:
    cd scripts
    python visualize_sharpa_overlay_3d.py \
        --data-root /path/to/poc_deliver \
        --keypoints-dir ../output \
        --retarget-dir ../output \
        --output-dir ../output \
        --max-episodes 1 --max-frames 300
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from utils import (
    R_MANO_TO_SHARPA,
    discover_episodes,
    draw_hand_skeleton,
    find_motion_start,
    get_fovy,
    get_intrinsics,
    project_to_pixel,
)

# Import the same hand frame builder used by retarget
from retarget_to_sharpa import _build_hand_frame

# ============================================================
# Paths & Constants
# ============================================================

URDF_BASE = Path(__file__).resolve().parent / "urdf" / "sharpa-urdf-usd-xml" / "wave_01"
DUAL_MJCF_DIR = URDF_BASE / "dual_sharpa_wave"
DUAL_MJCF = DUAL_MJCF_DIR / "dual_sharpa_wave.xml"


# ============================================================
# Overlay Renderer
# ============================================================

class OverlayRenderer:
    """Renders Sharpa hands at ego camera resolution for pixel-perfect overlay."""

    def __init__(self, width: int, height: int, fovy: float):
        cwd = os.getcwd()
        os.chdir(str(DUAL_MJCF_DIR))
        self.model = mujoco.MjModel.from_xml_path(DUAL_MJCF.name)
        os.chdir(cwd)

        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.model.vis.global_.fovy = fovy

        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 90
        self.camera.elevation = 0
        self.scene_option = mujoco.MjvOption()
        self.width = width
        self.height = height

    def render_hands(
        self,
        left_q22: np.ndarray, left_pos: np.ndarray, left_euler: np.ndarray,
        right_q22: np.ndarray, right_pos: np.ndarray, right_euler: np.ndarray,
        left_valid: bool = True, right_valid: bool = True,
        left_wrist: np.ndarray = None, right_wrist: np.ndarray = None,
    ) -> np.ndarray:
        """Render hands and return (H, W, 3) uint8 RGB on black background."""
        self.data.qpos[:] = 0.0

        if left_valid:
            self.data.qpos[6:9] = left_pos
            self.data.qpos[9:12] = left_euler
            self.data.qpos[12:34] = left_q22

        if right_valid:
            self.data.qpos[34:37] = right_pos
            self.data.qpos[37:40] = right_euler
            self.data.qpos[40:62] = right_q22

        mujoco.mj_kinematics(self.model, self.data)

        # Camera at origin looking along +Z_ego (= +Y_mj direction)
        if left_wrist is not None and right_wrist is not None and left_valid and right_valid:
            mid = (left_wrist + right_wrist) / 2
        elif left_valid and left_wrist is not None:
            mid = left_wrist
        elif right_valid and right_wrist is not None:
            mid = right_wrist
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # az=90, el=0: camera at lookat + dist*[0,-1,0]_mj, looking along +Y_mj
        # To place camera at MJ origin: lookat_y = dist → camera at [lx, ly-dist, lz] = [lx, 0, lz]
        # We want camera at [0, 0, 0] → lookat = [0, dist, 0], dist = mid_y
        self.camera.lookat[:] = [0, mid[1], 0]
        self.camera.distance = mid[1]

        self.renderer.update_scene(
            self.data, camera=self.camera, scene_option=self.scene_option
        )
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def composite(ego_frame: np.ndarray, sim_frame: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Alpha-composite sim_frame onto ego_frame using brightness mask."""
    gray = np.mean(sim_frame, axis=2)
    mask = (gray > 10).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask_3ch = mask[:, :, None]

    blended = (
        ego_frame.astype(np.float32) * (1 - mask_3ch * alpha)
        + sim_frame.astype(np.float32) * mask_3ch
    )
    return blended.clip(0, 255).astype(np.uint8)


# ============================================================
# Pipeline
# ============================================================


def process_episode(
    hands_npz_path, retarget_path, keypoints_path,
    video_path, output_path, video_info_path,
    max_frames=300, frame_step=2, overlay_alpha=0.7,
):
    """Generate AR overlay video for one episode."""
    hands = np.load(hands_npz_path, allow_pickle=True)
    rt = np.load(retarget_path)

    pred_trans = hands["pred_trans"]
    pred_valid = hands["pred_valid"]
    R_w2c = hands["R_w2c"]
    t_w2c = hands["t_w2c"]

    left_q = rt["left_hand_joints"]
    right_q = rt["right_hand_joints"]

    # Load keypoints — used for both skeleton overlay AND wrist pose
    if keypoints_path is None or not Path(keypoints_path).exists():
        print("ERROR: keypoints required for sharpa overlay")
        return 0
    kp_data = np.load(keypoints_path)
    joints_cam = kp_data["joints_cam"]    # (2, T, 21, 3) in camera frame
    kp_valid = kp_data["pred_valid"].astype(bool)

    # Video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T_data = min(T_total, joints_cam.shape[1], left_q.shape[0])

    fovy = get_fovy(hands, video_info_path, vid_h)
    overlay_renderer = OverlayRenderer(vid_w, vid_h, fovy)

    fx, fy, cx, cy = get_intrinsics(hands, video_info_path)

    start_frame = find_motion_start(pred_trans, pred_valid, R_w2c, t_w2c)

    # Sequential read to start
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

            # 1) Draw 21-keypoint skeleton
            for hand_idx in range(2):
                if kp_valid[hand_idx, t]:
                    pts_2d = project_to_pixel(
                        joints_cam[hand_idx, t], fx, fy, cx, cy
                    )
                    draw_hand_skeleton(ego_rgb, pts_2d)

            # 2) Sharpa mesh overlay — use same _build_hand_frame as retarget
            left_valid = kp_valid[0, t]
            right_valid = kp_valid[1, t]

            if left_valid or right_valid:
                # Ego camera → MuJoCo world coordinate transform
                # ego: X=right, Y=down, Z=forward
                # MJ:  X=right, Y=forward, Z=up
                # pos_mj = [ego_x, ego_z, -ego_y]
                R_cam2mj = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float64)

                # Build hand frame in camera coords, then transform to MJ world
                R_left_cam = _build_hand_frame(joints_cam[0, t], side="left")
                R_left_mj = R_cam2mj @ R_left_cam
                left_euler = Rotation.from_matrix(R_left_mj).as_euler("XYZ")

                R_right_cam = _build_hand_frame(joints_cam[1, t], side="right")
                R_right_mj = R_cam2mj @ R_right_cam
                right_euler = Rotation.from_matrix(R_right_mj).as_euler("XYZ")

                # Positions: transform wrist from ego cam to MJ, then subtract root offset
                # In MJ world at rest: left C_MC = root + [0, -0.15, 0], right C_MC = root + [0, +0.15, 0]
                # So: root = wrist_mj - offset
                left_wrist_mj = R_cam2mj @ joints_cam[0, t, 0]
                left_pos = left_wrist_mj - np.array([0.0, -0.15, 0.0])

                right_wrist_mj = R_cam2mj @ joints_cam[1, t, 0]
                right_pos = right_wrist_mj - np.array([0.0, 0.15, 0.0])

                sim_rgb = overlay_renderer.render_hands(
                    left_q[t], left_pos, left_euler,
                    right_q[t], right_pos, right_euler,
                    left_valid=left_valid, right_valid=right_valid,
                    left_wrist=left_wrist_mj,
                    right_wrist=right_wrist_mj,
                )

                combined = composite(ego_rgb, sim_rgb, alpha=overlay_alpha)
            else:
                combined = ego_rgb

            out_frames.append(combined)

        t += 1

    cap.release()
    overlay_renderer.close()

    if out_frames:
        iio.imwrite(output_path, np.stack(out_frames),
                    fps=fps / frame_step, codec="h264")

    return len(out_frames)


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="AR overlay: Sharpa hands composited onto ego video"
    )
    parser.add_argument("--data-root", type=str,
                        default="/media/hdd4tb/sankuai/code/07_具身/dataset/07_Ego/poc_deliver")
    parser.add_argument("--retarget-dir", type=str, default=None)
    parser.add_argument("--keypoints-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="../output")
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Overlay opacity (0=transparent, 1=opaque)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    episodes = discover_episodes(args.data_root)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"=== Sharpa AR Overlay ===")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Max frames: {args.max_frames}, step: {args.frame_step}")
    print(f"  Overlay alpha: {args.alpha}")
    print()

    for i, ep in enumerate(episodes):
        name = ep["name"]
        ep_out = output_dir / name
        ep_out.mkdir(parents=True, exist_ok=True)
        out_path = str(ep_out / "vis_retargeted_sharpa_overlay.mp4")

        rt_path = None
        if args.retarget_dir:
            p = Path(args.retarget_dir) / name / "hands_retargeted_sharpa.npz"
            if p.exists():
                rt_path = str(p)
        if not rt_path:
            p = Path(args.data_root) / name / "ego_process" / "ego_hands_reconstruction" / "hands_retargeted_sharpa.npz"
            if p.exists():
                rt_path = str(p)

        if not rt_path:
            print(f"  [{i+1}/{len(episodes)}] {name}: SKIP (no retarget)")
            continue

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

        print(f"  [{i+1}/{len(episodes)}] {name} ...", end=" ", flush=True)
        try:
            n = process_episode(
                ep["hands_npz"], rt_path, kp_path,
                ep["video_path"], out_path, ep["video_info"],
                max_frames=args.max_frames, frame_step=args.frame_step,
                overlay_alpha=args.alpha,
            )
            print(f"OK ({n} frames)")
        except Exception as e:
            print(f"FAILED ({e})")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Output: <output>/<episode>/vis_retargeted_sharpa_overlay.mp4")


if __name__ == "__main__":
    main()

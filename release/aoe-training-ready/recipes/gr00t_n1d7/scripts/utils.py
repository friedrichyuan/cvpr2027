"""
Shared utilities for the AoE retarget pipeline.

Extracted from visualize_*.py, retarget_to_*.py, and convert_ego_to_lerobot.py
to eliminate cross-file duplication.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# Constants
# ============================================================

HAND_EDGES_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_JOINT_RANGES = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]

FINGER_COLORS = [
    (255, 80, 80),    # thumb - red
    (80, 255, 80),    # index - green
    (80, 180, 255),   # middle - blue
    (255, 200, 80),   # ring - yellow
    (220, 80, 255),   # pinky - magenta
]

# MANO-to-Sharpa rest-frame rotation correction.
# MANO local: X≈fingers, -Y≈palm_normal
# Sharpa local: Z=fingers, -Y≈palm_normal, X=side(thumb)
R_MANO_TO_SHARPA = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
], dtype=np.float64)


# ============================================================
# Data discovery
# ============================================================

def discover_episodes(data_root: str) -> list:
    """Find all valid episodes under data_root.

    Returns list of dicts with keys: name, hands_npz, video_path, video_info.
    """
    data_root = Path(data_root)
    episodes = []
    for ep_dir in sorted(data_root.iterdir()):
        if not ep_dir.is_dir():
            continue
        hands_npz = ep_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
        video_undistorted = ep_dir / "ego_process" / "ego_undistorted_video" / "raw_video_undistorted.mp4"
        video_raw = ep_dir / "raw_video.mp4"
        video_path = video_undistorted if video_undistorted.exists() else video_raw
        if not hands_npz.exists() or not video_path.exists():
            continue
        video_info = ep_dir / "video_info.json"
        annotation = ep_dir / "ego_annotation" / "ego_action_annotation.json"
        episodes.append({
            "name": ep_dir.name,
            "dir": str(ep_dir),
            "hands_npz": str(hands_npz),
            "video_path": str(video_path),
            "video_info": str(video_info) if video_info.exists() else None,
            "annotation": str(annotation) if annotation.exists() else None,
        })
    return episodes


# ============================================================
# Camera & projection
# ============================================================

def get_intrinsics(hands_data, video_info_path) -> tuple:
    """Extract camera intrinsics (fx, fy, cx, cy) from hands.npz and video_info.json."""
    focal = float(hands_data["focal"])
    fx = fy = focal
    cx, cy = 960.0, 540.0
    if video_info_path and Path(video_info_path).exists():
        with open(video_info_path) as f:
            vinfo = json.load(f)
        cam = vinfo.get("cameraParams", {})
        fx = cam.get("fx_pixels", fx)
        fy = cam.get("fy_pixels", fy)
        cx = cam.get("cx_pixels", cx)
        cy = cam.get("cy_pixels", cy)
    return fx, fy, cx, cy


def get_fovy(hands_data, video_info_path, height=1080) -> float:
    """Compute vertical FOV in degrees from camera intrinsics."""
    focal = float(hands_data["focal"])
    if video_info_path and Path(video_info_path).exists():
        with open(video_info_path) as f:
            vinfo = json.load(f)
        cam = vinfo.get("cameraParams", {})
        focal = cam.get("fy_pixels", focal)
        height = cam.get("height", height)
    return float(2 * np.arctan(height / 2 / focal) * 180 / np.pi)


def project_to_pixel(points_cam, fx, fy, cx, cy) -> np.ndarray:
    """Project 3D camera-frame points to 2D pixel coordinates."""
    pts = np.asarray(points_cam)
    z = pts[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = pts[:, 0:1] / z * fx + cx
    v = pts[:, 1:2] / z * fy + cy
    return np.hstack([u, v])


# ============================================================
# Temporal processing
# ============================================================

def ema_smooth(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average smoothing along time axis."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def find_motion_start(
    pred_trans, pred_valid, R_w2c, t_w2c,
    threshold: float = 0.005, window: int = 10,
) -> int:
    """Find first frame with significant hand motion in camera frame."""
    T = pred_trans.shape[1]
    for t in range(window, T - window):
        if not (pred_valid[0, t] > 0.5 and pred_valid[1, t] > 0.5):
            continue
        pos_cam = R_w2c[t] @ pred_trans[0, t] + t_w2c[t]
        pos_prev = R_w2c[t - window] @ pred_trans[0, t - window] + t_w2c[t - window]
        speed = np.linalg.norm(pos_cam - pos_prev) / window
        if speed > threshold:
            return max(0, t - window)
    return 0


# ============================================================
# Visualization
# ============================================================

def draw_hand_skeleton(
    frame: np.ndarray,
    joints_2d: np.ndarray,
    color=None,
    per_finger_color: bool = True,
    alpha: float = 0.8,
    line_width: int = 2,
    point_radius: int = 3,
):
    """Draw 21-keypoint hand skeleton on frame (in-place).

    Args:
        frame: (H, W, 3) uint8 RGB/BGR image.
        joints_2d: (21, 2) pixel coordinates.
        color: single BGR/RGB tuple for all edges (overrides per_finger_color).
        per_finger_color: use FINGER_COLORS for each finger.
        alpha: blending alpha for the overlay.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for finger_idx, (start, end) in enumerate(FINGER_JOINT_RANGES):
        c = FINGER_COLORS[finger_idx] if per_finger_color and color is None else (color or (0, 220, 100))
        edges_in_finger = [(a, b) for (a, b) in HAND_EDGES_21
                           if start <= a <= end or start <= b <= end]
        for (a, b) in edges_in_finger:
            pa = tuple(joints_2d[a].astype(int))
            pb = tuple(joints_2d[b].astype(int))
            if 0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h:
                cv2.line(overlay, pa, pb, c, line_width, cv2.LINE_AA)

    for i, pt in enumerate(joints_2d):
        pt_int = tuple(pt.astype(int))
        if 0 <= pt_int[0] < w and 0 <= pt_int[1] < h:
            if per_finger_color and color is None:
                finger_idx = next(
                    (fi for fi, (s, e) in enumerate(FINGER_JOINT_RANGES) if s <= i <= e), 0
                )
                c = FINGER_COLORS[finger_idx]
            else:
                c = color or (0, 220, 100)
            cv2.circle(overlay, pt_int, point_radius, c, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt_int, point_radius, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

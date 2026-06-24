"""AOE-specific constants.

Coordinate-axis conventions for the OpenCV camera frame, the cam-local
shoulder offset prior, and per-side T_align matrices for MANO wrist frame.

CRITICAL: do NOT reuse phantom/egodex_to_g1 T_align values — those are
calibrated for ARKit wrist conventions (different from MANO).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── AOE dataset file names ───────────────────────────────────────────────────

HANDS_NPZ_NAME = "hands.npz"
SIDECAR_NPZ_NAME = "hands_keypoints.npz"
CAMERA_TRAJ_NPZ_NAME = "camera_traj.npz"

HANDS_RECON_SUBDIRS = (
    "ego_hands_reconstruction",
    Path("ego_process") / "ego_hands_reconstruction",
)

UNDISTORTED_VIDEO_SUBDIRS = (
    "ego_undistorted_video",
    Path("ego_process") / "ego_undistorted_video",
)

# ── AOE camera (OpenCV) → G1 base (x=fwd, y=left, z=up) ────────────────────
#
# AOE cam-local: +X right, +Y down, +Z forward (OpenCV convention)
# G1 base:      +X forward, +Y left, +Z up
R_AOE_CAM_TO_G1 = np.array(
    [
        [0,  0,  1],   # G1 +x = AOE +z (forward)
        [-1, 0,  0],   # G1 +y = -AOE +x (right → left)
        [0, -1,  0],   # G1 +z = -AOE +y (down → up)
    ],
    dtype=np.float64,
)

# ── Cam-local shoulder offset prior ─────────────────────────────────────────
#
# The neck-mounted camera hangs ~20 cm below shoulder midpoint.
# In OpenCV cam-local (+Y = down), shoulder is at NEGATIVE y (upward).
# CRITICAL: y must be negative — a positive value causes "hands above head"
# (see aoe_to_g1 debug history v1→v3).
SHOULDER_OFFSET_FROM_CAMERA_CAM = np.array([0.00, -0.20, -0.05], dtype=np.float64)
SHOULDER_HALF_WIDTH = 0.18

# ── MANO wrist local → G1 wrist local alignment ────────────────────────────
#
# Calibrated on AoE sample data via 21-keypoint dot-product test.
# LEFT:  MANO (distal=+X, palm=+Y, radial=+Z) ≡ G1 → identity
# RIGHT: MANO needs X,Y flip relative to G1
AOE_T_ALIGN_LEFT_WRIST = np.eye(3, dtype=np.float64)
AOE_T_ALIGN_RIGHT_WRIST = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)

# ── Fingertip keypoint indices (OpenPose-21 ordering) ───────────────────────

DEX3_TIPS = (4, 8, 12)       # thumb tip, index tip, middle tip
INSPIRE_TIPS = (4, 8, 12, 16, 20)  # all five fingertips

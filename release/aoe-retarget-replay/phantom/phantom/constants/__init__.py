# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Hand-agnostic constants for the AoE → G1 retargeting pipeline.

Single source of truth for coordinate transforms, scaling parameters, IK
settings, arm joint layouts, and LeRobot format constants.
"""

from pathlib import Path

import numpy as np

# ── Paths (relative to project root) ────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ROBOTS_DIR = ASSETS_DIR / "robots"
MANO_MODELS_DIR = ASSETS_DIR / "mano_models"
# Fallback to the shared assets/mano/ location for MANO models.
# Users should run assets/mano/download_mano.sh before using this package.
SHARED_MANO_DIR = PROJECT_ROOT.parent.parent.parent / "assets" / "mano"

RETARGET_CONFIG_DIR = PROJECT_ROOT / "configs" / "retarget"

# ── Coordinate transforms (ARKit → G1, used by phantom IK internals) ────────

R_ARKIT_TO_G1 = np.array([
    [ 0,  0, -1],
    [-1,  0,  0],
    [ 0,  1,  0],
], dtype=np.float64)

T_ALIGN_LEFT_WRIST = np.diag([+1, -1, -1]).astype(np.float64)
T_ALIGN_RIGHT_WRIST = np.diag([-1, -1, +1]).astype(np.float64)

# ── G1 body constants ───────────────────────────────────────────────────────

G1_STANDING_HEIGHT = 0.793
G1_ARM_LENGTH = 0.4095
G1_SHOULDER_SPACING = 0.2004

G1_LEFT_SHOULDER_BODY = "left_shoulder_pitch_link"
G1_RIGHT_SHOULDER_BODY = "right_shoulder_pitch_link"

G1_LEFT_WRIST_TARGET_BODY = "left_wrist_yaw_link"
G1_RIGHT_WRIST_TARGET_BODY = "right_wrist_yaw_link"

# ── Scaling parameters ──────────────────────────────────────────────────────

ARM_LENGTH_PERCENTILE = 95.0
MIN_ARM_LENGTH_SAMPLE = 0.05

# ── Confidence threshold ────────────────────────────────────────────────────

CONF_THRESHOLD = 0.10

# ── IK parameters ───────────────────────────────────────────────────────────

IK_POSITION_COST = 1.0
IK_ORIENTATION_COST = 0.3
IK_SMOOTHNESS_COST = 0.1
IK_SOLVER = "proxqp"
IK_DAMPING = 1e-3
IK_SEED_ITERS = 80
IK_STEP_ITERS = 10
IK_POS_TOLERANCE = 0.002

# ── Validation / smoothness thresholds ───────────────────────────────────────

ARM_DELTA_THRESH = 0.5
HAND_DELTA_THRESH = 1.0
DEAD_ARM_RANGE_THRESH = 0.1
DEAD_HAND_RANGE_THRESH = 0.05

# ── G1 arm joint structure (shared between Dex3 and Inspire) ────────────────

LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_QPOS_IDX = slice(22, 29)
LEFT_ARM_JOINT_IDS = slice(16, 23)

# ── LeRobot format constants ────────────────────────────────────────────────

FPS = 30
CHUNK_SIZE = 1000
VIDEO_KEY = "observation.images.egoview"
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640
VIDEO_CODEC = "h264"
VIDEO_PIX_FMT = "yuv420p"

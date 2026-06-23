"""Dex3-specific constants: 28-DoF action layout [L_arm(7), R_arm(7), L_hand(7), R_hand(7)]."""

from phantom.constants import (
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    ROBOTS_DIR,
    VIDEO_KEY,
)

# ── Paths ────────────────────────────────────────────────────────────────────

G1_MJCF_PATH = ROBOTS_DIR / "g1_dex3" / "g1_mocap_29dof_with_dex3_hands.xml"

DEX3_URDF_DIR = ROBOTS_DIR / "hands"
DEX3_CONFIG_DIR = ROBOTS_DIR.parent.parent / "configs" / "retarget"

# ── 28-DoF action layout ────────────────────────────────────────────────────

ACTION_DIM = 28

LEFT_HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]
RIGHT_HAND_JOINT_NAMES = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

ACTION_JOINT_NAMES = (
    LEFT_ARM_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
    + LEFT_HAND_JOINT_NAMES
    + RIGHT_HAND_JOINT_NAMES
)

# ── G1+Dex3 MJCF qpos indices ───────────────────────────────────────────────

LEFT_HAND_QPOS_IDX = slice(29, 36)
RIPOS_IDX = slice(36, 43)
RIGHT_HAND_QPOS_IDX = slice(43, 50)

LEFT_HAND_JOINT_IDS = slice(23, 30)
RIGHT_ARM_JOINT_IDS = slice(30, 37)
RIGHT_HAND_JOINT_IDS = slice(37, 44)

# ── LeRobot metadata ────────────────────────────────────────────────────────

ROBOT_TYPE = "unitree_g1_dex3"

MODALITY = {
    "state": {
        "left_arm": {"start": 0, "end": 7},
        "right_arm": {"start": 7, "end": 14},
        "left_hand": {"start": 14, "end": 21},
        "right_hand": {"start": 21, "end": 28},
    },
    "action": {
        "left_arm": {"start": 0, "end": 7},
        "right_arm": {"start": 7, "end": 14},
        "left_hand": {"start": 14, "end": 21},
        "right_hand": {"start": 21, "end": 28},
    },
    "video": {
        "ego_view": {"original_key": VIDEO_KEY},
    },
    "annotation": {
        "human.action.task_description": {},
        "human.validity": {},
    },
}

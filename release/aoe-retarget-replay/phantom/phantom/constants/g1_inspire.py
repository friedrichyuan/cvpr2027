"""Inspire-specific constants: 26-DoF action layout [L_arm(7), R_arm(7), L_hand(6), R_hand(6)]."""

from phantom.constants import (
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    ROBOTS_DIR,
    VIDEO_KEY,
)

# ── Paths ────────────────────────────────────────────────────────────────────

G1_INSPIRE_MJCF_PATH = (
    ROBOTS_DIR / "g1_inspire" / "g1_mocap_29dof_with_inspire_hands.xml"
)

INSPIRE_URDF_DIR = ROBOTS_DIR / "hands"
INSPIRE_CONFIG_DIR = ROBOTS_DIR.parent.parent / "configs" / "retarget"

# ── 26-DoF action layout ────────────────────────────────────────────────────

INSPIRE_ACTION_DIM = 26

LEFT_HAND_INSPIRE_JOINT_NAMES = [
    "l_thumb_proximal_yaw_joint",
    "l_thumb_proximal_pitch_joint",
    "l_index_proximal_joint",
    "l_middle_proximal_joint",
    "l_ring_proximal_joint",
    "l_pinky_proximal_joint",
]
RIGHT_HAND_INSPIRE_JOINT_NAMES = [
    "r_thumb_proximal_yaw_joint",
    "r_thumb_proximal_pitch_joint",
    "r_index_proximal_joint",
    "r_middle_proximal_joint",
    "r_ring_proximal_joint",
    "r_pinky_proximal_joint",
]
INSPIRE_ACTION_JOINT_NAMES = (
    LEFT_ARM_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
    + LEFT_HAND_INSPIRE_JOINT_NAMES
    + RIGHT_HAND_INSPIRE_JOINT_NAMES
)
assert len(INSPIRE_ACTION_JOINT_NAMES) == INSPIRE_ACTION_DIM

# ── G1+Inspire MJCF qpos / joint layout ─────────────────────────────────────

LEFT_HAND_QPOS_RANGE = slice(29, 41)
RIGHT_HAND_QPOS_RANGE = slice(48, 60)
INSPIRE_ACTUATED_OFFSETS = [0, 1, 4, 6, 8, 10]
INSPIRE_MIMIC_OFFSETS = [2, 3, 5, 7, 9, 11]

RIGHT_ARM_QPOS_IDX = slice(41, 48)

RIGHT_ARM_JOINT_IDS = slice(35, 42)
LEFT_HAND_JOINT_IDS_RANGE = slice(23, 35)
RIGHT_HAND_JOINT_IDS_RANGE = slice(42, 54)

# ── Mimic rules ─────────────────────────────────────────────────────────────

INSPIRE_MIMIC_RULES: list[tuple[str, str, float, float]] = []
for _prefix in ("l_", "r_"):
    INSPIRE_MIMIC_RULES.extend([
        (f"{_prefix}thumb_intermediate_joint",
         f"{_prefix}thumb_proximal_pitch_joint", 1.334,    0.0),
        (f"{_prefix}thumb_distal_joint",
         f"{_prefix}thumb_proximal_pitch_joint", 0.667,    0.0),
        (f"{_prefix}index_intermediate_joint",
         f"{_prefix}index_proximal_joint",       1.06399, -0.04545),
        (f"{_prefix}middle_intermediate_joint",
         f"{_prefix}middle_proximal_joint",      1.06399, -0.04545),
        (f"{_prefix}ring_intermediate_joint",
         f"{_prefix}ring_proximal_joint",        1.06399, -0.04545),
        (f"{_prefix}pinky_intermediate_joint",
         f"{_prefix}pinky_proximal_joint",       1.06399, -0.04545),
    ])

# ── LeRobot metadata ────────────────────────────────────────────────────────

INSPIRE_ROBOT_TYPE = "unitree_g1_inspire"

INSPIRE_MODALITY = {
    "state": {
        "left_arm":  {"start":  0, "end":  7},
        "right_arm": {"start":  7, "end": 14},
        "left_hand": {"start": 14, "end": 20},
        "right_hand":{"start": 20, "end": 26},
    },
    "action": {
        "left_arm":  {"start":  0, "end":  7},
        "right_arm": {"start":  7, "end": 14},
        "left_hand": {"start": 14, "end": 20},
        "right_hand":{"start": 20, "end": 26},
    },
    "video": {
        "ego_view": {"original_key": VIDEO_KEY},
    },
    "annotation": {
        "human.action.task_description": {},
        "human.validity": {},
    },
}

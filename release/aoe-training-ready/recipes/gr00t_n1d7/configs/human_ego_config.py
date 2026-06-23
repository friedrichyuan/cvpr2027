"""
GR00T N1.7 Embodiment Configurations for Human Egocentric Hand Data.

Two modes:
  gripper  (20D):  EEF 9D + 1D gripper per hand — simplified for robot transfer
  sharpa   (62D):  EEF 9D + 22D Sharpa Wave joints — full dexterous hand

Embodiment tags (must match GR00T N1.7 pretrained model registry):
  "human_ego_bimanual_gripper_relative_eef"
      → Custom tag (no pretrained projector, requires finetune from scratch)
      → 20D bimanual gripper
      → Naming: {data_source}_{arm_type}_{hand_type}_{action_type}
  "real_r1_pro_sharpa_relative_eef_human"
      → From GR00T N1.7 pretrain weights (projector index 26)
      → 62D bimanual Sharpa Wave, shares pretrained embeddings with R1 Pro robot data
      → Naming convention: {platform}_{hand}_{action_type}_{data_source}

This module is the single source of truth for mode metadata.
Both convert_ego_to_lerobot.py and launch_pretrain.sh derive their
mode-specific settings from the constants defined here.
"""

# =============================================================================
# Mode registry — canonical definitions used by the data pipeline and training
# =============================================================================

MODE_CONFIGS = {
    "gripper": {
        "state_dim": 20,
        "embodiment_tag": "human_ego_bimanual_gripper_relative_eef",
        "modality_keys": ["left_wrist_eef", "right_wrist_eef", "left_gripper", "right_gripper"],
        "state_layout": "[left_wrist_eef(9), right_wrist_eef(9), left_gripper(1), right_gripper(1)]",
    },
    "sharpa": {
        "state_dim": 62,
        "embodiment_tag": "real_r1_pro_sharpa_relative_eef_human",
        "modality_keys": ["left_wrist_eef", "right_wrist_eef", "left_hand_joints", "right_hand_joints"],
        "state_layout": "[left_wrist_eef(9), right_wrist_eef(9), left_hand_joints(22), right_hand_joints(22)]",
    },
}

# Derived constants
GRIPPER_EMBODIMENT_TAG = MODE_CONFIGS["gripper"]["embodiment_tag"]
SHARPA_EMBODIMENT_TAG = MODE_CONFIGS["sharpa"]["embodiment_tag"]

# =============================================================================
# GR00T Training Modality Configs (used by launch_pretrain.sh & model processor)
# =============================================================================

_GRIPPER_KEYS = MODE_CONFIGS["gripper"]["modality_keys"]
_SHARPA_KEYS = MODE_CONFIGS["sharpa"]["modality_keys"]

# Action configs ordered to match EEF-first modality_keys layout:
#   [left_wrist_eef, right_wrist_eef, left_gripper/hand, right_gripper/hand]
# EEF keys get RELATIVE representation; hand keys get ABSOLUTE.
_ACTION_CONFIGS_BIMANUAL = [
    {
        "rep": "RELATIVE",
        "type": "EEF",
        "format": "XYZ_ROT6D",
        "state_key": "left_wrist_eef",
    },
    {
        "rep": "RELATIVE",
        "type": "EEF",
        "format": "XYZ_ROT6D",
        "state_key": "right_wrist_eef",
    },
    {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
    },
    {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
    },
]

GRIPPER_MODALITY_CONFIG = {
    "video": {
        "delta_indices": [-15, 0],
        "modality_keys": ["ego_view"],
    },
    "state": {
        "delta_indices": [0],
        "modality_keys": _GRIPPER_KEYS,
    },
    "action": {
        "delta_indices": list(range(0, 16)),
        "modality_keys": _GRIPPER_KEYS,
        "action_configs": _ACTION_CONFIGS_BIMANUAL,
    },
    "language": {
        "delta_indices": [0],
        "modality_keys": ["annotation.human.task_description"],
    },
}

SHARPA_MODALITY_CONFIG = {
    "video": {
        "delta_indices": [-15, 0],
        "modality_keys": ["ego_view"],
    },
    "state": {
        "delta_indices": [0],
        "modality_keys": _SHARPA_KEYS,
    },
    "action": {
        "delta_indices": list(range(0, 16)),
        "modality_keys": _SHARPA_KEYS,
        "action_configs": _ACTION_CONFIGS_BIMANUAL,
    },
    "language": {
        "delta_indices": [0],
        "modality_keys": ["annotation.human.task_description"],
    },
}

# Map mode name → modality config for easy lookup
MODALITY_CONFIGS = {
    "gripper": GRIPPER_MODALITY_CONFIG,
    "sharpa": SHARPA_MODALITY_CONFIG,
}

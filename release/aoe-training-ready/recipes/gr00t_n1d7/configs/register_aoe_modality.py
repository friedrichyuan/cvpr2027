# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Modality config registrar for AoE ego-hand embodiments (sharpa / gripper).

Loaded by GR00T training via --modality-config-path. Registers custom
embodiment tags that are NOT in the built-in MODALITY_CONFIGS registry:

  real_r1_pro_sharpa_relative_eef_human         → 62D bimanual Sharpa
  human_ego_bimanual_gripper_relative_eef       → 20D bimanual gripper

The modality configs are mirrored here (not imported from configs/) because
GR00T expects ActionConfig dataclasses with enum members, and this file must
be self-contained when imported from an arbitrary cwd.
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Action configs ordered to match EEF-first modality_keys layout:
#   [left_wrist_eef, right_wrist_eef, left_gripper/right_hand, right_gripper/right_hand]
# Both gripper and sharpa share this ordering: EEF keys first, then hand keys.
_BIMANUAL_ACTION_CONFIGS = [
    ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF,
                 format=ActionFormat.XYZ_ROT6D, state_key="left_wrist_eef"),
    ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF,
                 format=ActionFormat.XYZ_ROT6D, state_key="right_wrist_eef"),
    ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                 format=ActionFormat.DEFAULT),
    ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                 format=ActionFormat.DEFAULT),
]


def _build(modality_keys, action_horizon=16):
    return {
        "video": ModalityConfig(delta_indices=[-15, 0], modality_keys=["ego_view"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=modality_keys),
        "action": ModalityConfig(
            delta_indices=list(range(0, action_horizon)),
            modality_keys=modality_keys,
            action_configs=_BIMANUAL_ACTION_CONFIGS,
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"],
        ),
    }


# Sharpa uses the pre-registered R1 Pro sharpa_human config (already in registry
# from pretrain weights / processor_config). Provide standard keys for fallback.
_SHARPA_KEYS = ["left_wrist_eef", "right_wrist_eef", "left_hand_joints", "right_hand_joints"]
_GRIPPER_KEYS = ["left_wrist_eef", "right_wrist_eef", "left_gripper", "right_gripper"]


def _register(tag_enum, config):
    if tag_enum.value in MODALITY_CONFIGS:
        return  # already registered (e.g. sharpa from pretrain weights)
    register_modality_config(config, tag_enum)


# sharpa_human: if not yet in registry (e.g. standalone without pretrain config),
# register with standard R1 Pro keys. No-op if pretrain config already registered it.
_register(EmbodimentTag.REAL_R1_PRO_SHARPA_HUMAN, _build(_SHARPA_KEYS))

# gripper tag is custom — register under the literal string since EmbodimentTag
# enum may not have it. Fall back to direct dict insertion.
_gripper_tag = "human_ego_bimanual_gripper_relative_eef"
if _gripper_tag not in MODALITY_CONFIGS:
    MODALITY_CONFIGS[_gripper_tag] = _build(_GRIPPER_KEYS)

print(f"[register_aoe_modality] sharpa_human in registry="
      f"{EmbodimentTag.REAL_R1_PRO_SHARPA_HUMAN.value in MODALITY_CONFIGS}, "
      f"gripper registered={_gripper_tag in MODALITY_CONFIGS}")

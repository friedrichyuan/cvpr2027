# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""G1 + Dex3 (28-DoF) robot spec."""

from __future__ import annotations

import numpy as np

from phantom.constants.g1_dex3 import (
    ACTION_DIM,
    ACTION_JOINT_NAMES,
    G1_MJCF_PATH,
    LEFT_HAND_JOINT_IDS,
    MODALITY,
    RIGHT_HAND_JOINT_IDS,
    ROBOT_TYPE,
)
from phantom.retarget.hand_retarget_dex3 import Dex3HandRetargeter
from phantom.robots import RobotSpec, register
from phantom.robots.mjcf_patch import patch_mjcf_local

_HAND_BODY_SUBSTRINGS = ("hand_thumb", "hand_middle", "hand_index")


def _dex3_hand_body_predicate(body_name: str) -> bool:
    return any(s in body_name for s in _HAND_BODY_SUBSTRINGS)


def _slice_to_ids(s: slice) -> np.ndarray:
    return np.arange(s.start, s.stop, dtype=int)


def _dex3_qpos_writer_factory(model):
    """Action layout: [L_arm(7), R_arm(7), L_hand(7), R_hand(7)] — 28 DoF."""
    import mujoco
    qpos_idx = np.array(
        [int(model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        ]) for n in ACTION_JOINT_NAMES],
        dtype=int,
    )

    def write(qpos_full: np.ndarray, action: np.ndarray) -> None:
        qpos_full[qpos_idx] = action

    return write


G1_DEX3_SPEC = register(RobotSpec(
    name="g1_dex3",
    display_name="G1 + Dex3",
    mjcf_path=G1_MJCF_PATH,
    action_dim=ACTION_DIM,
    action_joint_names=list(ACTION_JOINT_NAMES),
    mimic_rules=[],
    hand_body_predicate=_dex3_hand_body_predicate,
    patch_mjcf=patch_mjcf_local,
    robot_type=ROBOT_TYPE,
    modality=MODALITY,
    hand_retargeter_cls=Dex3HandRetargeter,
    qpos_writer_factory=_dex3_qpos_writer_factory,
    left_hand_clamp_joint_ids=_slice_to_ids(LEFT_HAND_JOINT_IDS),
    right_hand_clamp_joint_ids=_slice_to_ids(RIGHT_HAND_JOINT_IDS),
))

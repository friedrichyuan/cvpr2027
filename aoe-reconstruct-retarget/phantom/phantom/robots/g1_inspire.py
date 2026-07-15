# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""G1 + Inspire Hand (26-DoF) robot spec."""

from __future__ import annotations

import numpy as np

from phantom.constants.g1_inspire import (
    G1_INSPIRE_MJCF_PATH,
    INSPIRE_ACTION_DIM,
    INSPIRE_ACTION_JOINT_NAMES,
    INSPIRE_ACTUATED_OFFSETS,
    INSPIRE_MIMIC_RULES,
    INSPIRE_MODALITY,
    INSPIRE_ROBOT_TYPE,
    LEFT_HAND_JOINT_IDS_RANGE,
    RIGHT_HAND_JOINT_IDS_RANGE,
)
from phantom.retarget.hand_retarget_inspire import InspireHandRetargeter
from phantom.robots import RobotSpec, register
from phantom.robots.mjcf_patch import patch_mjcf_with_sibling_dirs

_INSPIRE_HAND_BODY_PREFIXES = (
    "l_thumb", "l_index", "l_middle", "l_ring", "l_pinky",
    "r_thumb", "r_index", "r_middle", "r_ring", "r_pinky",
)


def _inspire_hand_body_predicate(body_name: str) -> bool:
    return any(body_name.startswith(p) for p in _INSPIRE_HAND_BODY_PREFIXES)


_INSPIRE_LEFT_ACTUATED_IDS = np.array(
    [LEFT_HAND_JOINT_IDS_RANGE.start + off for off in INSPIRE_ACTUATED_OFFSETS],
    dtype=int,
)
_INSPIRE_RIGHT_ACTUATED_IDS = np.array(
    [RIGHT_HAND_JOINT_IDS_RANGE.start + off for off in INSPIRE_ACTUATED_OFFSETS],
    dtype=int,
)


def _inspire_qpos_writer_factory(model):
    """Action layout: [L_arm(7), R_arm(7), L_hand(6), R_hand(6)] — 26 DoF."""
    import mujoco

    def _jadr(name: str) -> int:
        return int(model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ])

    actuated_idx = np.array(
        [_jadr(n) for n in INSPIRE_ACTION_JOINT_NAMES], dtype=int,
    )
    mimic_table: list[tuple[int, int, float, float]] = [
        (_jadr(mimic), _jadr(driver), float(mult), float(off))
        for (mimic, driver, mult, off) in INSPIRE_MIMIC_RULES
    ]

    def write(qpos_full: np.ndarray, action: np.ndarray) -> None:
        qpos_full[actuated_idx] = action
        for m_adr, d_adr, mult, off in mimic_table:
            qpos_full[m_adr] = mult * qpos_full[d_adr] + off

    return write


G1_INSPIRE_SPEC = register(RobotSpec(
    name="g1_inspire",
    display_name="G1 + Inspire",
    mjcf_path=G1_INSPIRE_MJCF_PATH,
    action_dim=INSPIRE_ACTION_DIM,
    action_joint_names=list(INSPIRE_ACTION_JOINT_NAMES),
    mimic_rules=INSPIRE_MIMIC_RULES,
    hand_body_predicate=_inspire_hand_body_predicate,
    patch_mjcf=patch_mjcf_with_sibling_dirs,
    robot_type=INSPIRE_ROBOT_TYPE,
    modality=INSPIRE_MODALITY,
    hand_retargeter_cls=InspireHandRetargeter,
    qpos_writer_factory=_inspire_qpos_writer_factory,
    left_hand_clamp_joint_ids=_INSPIRE_LEFT_ACTUATED_IDS,
    right_hand_clamp_joint_ids=_INSPIRE_RIGHT_ACTUATED_IDS,
))

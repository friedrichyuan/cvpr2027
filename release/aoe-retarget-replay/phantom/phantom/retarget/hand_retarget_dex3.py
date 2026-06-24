# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Dex3 hand retargeting (3 fingertips → 7 actuated joint angles)."""

from __future__ import annotations

from pathlib import Path

from phantom.constants.g1_dex3 import (
    DEX3_CONFIG_DIR,
    DEX3_URDF_DIR,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)
from phantom.retarget.hand_retarget_base import HandRetargeter


class Dex3HandRetargeter(HandRetargeter):
    """3-finger gripper: thumb, index, middle."""

    N_TIPS = 3
    N_HAND_DOF = 7
    CONFIG_FILENAME_TEMPLATE = "dex3_hand_{side}.yml"

    @classmethod
    def canonical_joint_names(cls, side: str) -> list[str]:
        return LEFT_HAND_JOINT_NAMES if side == "left" else RIGHT_HAND_JOINT_NAMES

    @classmethod
    def urdf_dir(cls) -> Path:
        return DEX3_URDF_DIR

    @classmethod
    def config_dir(cls) -> Path:
        return DEX3_CONFIG_DIR

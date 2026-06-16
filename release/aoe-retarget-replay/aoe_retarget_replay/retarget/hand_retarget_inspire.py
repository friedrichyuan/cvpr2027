"""Inspire-hand retargeting (5 fingertips → 6 actuated joint angles)."""

from __future__ import annotations

from pathlib import Path

from aoe_retarget_replay.constants.g1_inspire import (
    INSPIRE_CONFIG_DIR,
    INSPIRE_URDF_DIR,
    LEFT_HAND_INSPIRE_JOINT_NAMES,
    RIGHT_HAND_INSPIRE_JOINT_NAMES,
)
from aoe_retarget_replay.retarget.hand_retarget_base import HandRetargeter


class InspireHandRetargeter(HandRetargeter):
    """5-finger anthropomorphic hand: 6 actuated + 6 mimic per side."""

    N_TIPS = 5
    N_HAND_DOF = 6
    CONFIG_FILENAME_TEMPLATE = "inspire_hand_{side}.yml"

    @classmethod
    def canonical_joint_names(cls, side: str) -> list[str]:
        return (
            LEFT_HAND_INSPIRE_JOINT_NAMES if side == "left"
            else RIGHT_HAND_INSPIRE_JOINT_NAMES
        )

    @classmethod
    def urdf_dir(cls) -> Path:
        return INSPIRE_URDF_DIR

    @classmethod
    def config_dir(cls) -> Path:
        return INSPIRE_CONFIG_DIR

    def _validate_optimizer(self, side: str) -> None:
        opt_target_names = list(
            getattr(self.retargeting.optimizer, "target_joint_names", [])
        )
        hand_targets = [n for n in opt_target_names if not n.startswith("dummy_")]
        canonical = self.canonical_joint_names(side)
        assert sorted(hand_targets) == sorted(canonical), (
            "Inspire optimizer target_joint_names mismatch: "
            f"expected {canonical}, got {hand_targets}"
        )

        dof_names = list(self.retargeting.optimizer.robot.dof_joint_names)
        hand_dof_names = [n for n in dof_names if not n.startswith("dummy_")]
        assert len(hand_dof_names) == 12, (
            f"Expected 12 hand DOF in pinocchio model, "
            f"got {len(hand_dof_names)}: {hand_dof_names}"
        )

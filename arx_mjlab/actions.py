"""Arm reference-residual action with deterministic reference gripper control."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from .reference_bank import ReferenceBankCommand


@dataclass(kw_only=True)
class ArmResidualActionCfg(BaseActionCfg):
    command_name: str = "reference"

    def __post_init__(self) -> None:
        self.transmission_type = TransmissionType.JOINT

    def build(self, env) -> "ArmResidualAction":
        return ArmResidualAction(self, env)


class ArmResidualAction(BaseAction):
    _GRIPPER_JOINT_NAMES = (
        "left_joint7",
        "left_joint8",
        "right_joint17",
        "right_joint18",
    )

    def __init__(self, cfg: ReferenceResidualActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._gripper_joint_ids = torch.tensor(
            [self._entity.joint_names.index(name) for name in self._GRIPPER_JOINT_NAMES],
            device=self.device,
            dtype=torch.long,
        )

    def apply_actions(self) -> None:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        assert isinstance(command, ReferenceBankCommand)
        # The actor only corrects the arm reference.  A zero action therefore
        # recovers the offline IK controller, making it a safe RL baseline.
        self._entity.set_joint_position_target(
            command.qpos[:, self.target_ids] + self._processed_actions,
            joint_ids=self.target_ids,
        )
        # The parallel-jaw opening comes from the retargeted human width.  It
        # is a deterministic reference command, not an independent PPO action.
        gripper_target = command.qpos[:, self._gripper_joint_ids]
        self._entity.set_joint_position_target(
            gripper_target,
            joint_ids=self._gripper_joint_ids,
        )

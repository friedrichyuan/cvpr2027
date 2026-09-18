"""Reference-residual position action for ARX tracking RL."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from .reference_bank import ReferenceBankCommand


@dataclass(kw_only=True)
class ReferenceResidualActionCfg(BaseActionCfg):
    command_name: str = "reference"
    lead_time: float = 0.06

    def __post_init__(self) -> None:
        self.transmission_type = TransmissionType.JOINT

    def build(self, env) -> "ReferenceResidualAction":
        return ReferenceResidualAction(self, env)


class ReferenceResidualAction(BaseAction):
    def apply_actions(self) -> None:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        assert isinstance(command, ReferenceBankCommand)
        target = (
            command.qpos[:, self.target_ids]
            + self.cfg.lead_time * command.qvel[:, self.target_ids]
            + self._processed_actions
        )
        self._entity.set_joint_position_target(target, joint_ids=self.target_ids)

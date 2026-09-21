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


@dataclass(kw_only=True)
class ArmTransitionActionCfg(BaseActionCfg):
    command_name: str = "reference"

    def __post_init__(self) -> None:
        self.transmission_type = TransmissionType.JOINT

    def build(self, env) -> "ArmTransitionAction":
        return ArmTransitionAction(self, env)


class ArmResidualAction(BaseAction):
    _GRIPPER_JOINT_NAMES = (
        "left_joint7",
        "left_joint8",
        "right_joint17",
        "right_joint18",
    )

    def __init__(self, cfg: ArmResidualActionCfg, env) -> None:
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


class ArmTransitionAction(BaseAction):
    _GRIPPER_JOINT_NAMES = ArmResidualAction._GRIPPER_JOINT_NAMES

    def __init__(self, cfg: ArmTransitionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._gripper_joint_ids = torch.tensor(
            [self._entity.joint_names.index(name) for name in self._GRIPPER_JOINT_NAMES],
            device=self.device,
            dtype=torch.long,
        )

    def apply_actions(self) -> None:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        progress = _episode_progress(self._env, self.device)
        # A zero policy follows a smooth zero-to-first-frame joint ramp; PPO
        # only needs to learn residual corrections for dynamics and limits.
        arm_target = progress * command.qpos[:, self.target_ids]
        self._entity.set_joint_position_target(
            arm_target + self._processed_actions,
            joint_ids=self.target_ids,
        )
        gripper_target = progress * command.qpos[:, self._gripper_joint_ids]
        self._entity.set_joint_position_target(
            gripper_target,
            joint_ids=self._gripper_joint_ids,
        )


def _episode_progress(env, device: torch.device) -> torch.Tensor:
    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is None:
        return torch.ones(env.num_envs, 1, device=device)
    max_episode_length = getattr(env, "max_episode_length", None)
    if max_episode_length is None:
        max_episode_length = int(round(env.cfg.episode_length_s / env.step_dt))
    progress = episode_length.to(device=device, dtype=torch.float32) / max(
        float(max_episode_length) - 1.0,
        1.0,
    )
    return torch.clamp(progress, 0.0, 1.0).unsqueeze(-1)

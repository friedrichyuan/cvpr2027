"""Task-agnostic tracking observations, rewards, and reset for ARX references."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from .reference_bank import ReferenceBankCommand


def reference_command(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, ReferenceBankCommand)
    return command.command


def tcp_target_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, ReferenceBankCommand)
    robot = env.scene[asset_cfg.name]
    return (command.tcp_pos - robot.data.site_pos_w[:, asset_cfg.site_ids]).reshape(env.num_envs, -1)


def joint_tracking_exp(env, command_name: str, std: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, ReferenceBankCommand)
    robot = env.scene["robot"]
    arm_joint_ids = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)
    error = torch.mean(
        torch.square(robot.data.joint_pos[:, arm_joint_ids] - command.qpos[:, arm_joint_ids]),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def tcp_position_tracking_exp(env, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, ReferenceBankCommand)
    robot = env.scene[asset_cfg.name]
    error = torch.mean(torch.sum(torch.square(robot.data.site_pos_w[:, asset_cfg.site_ids] - command.tcp_pos), dim=-1), dim=-1)
    return torch.exp(-error / std**2)


def tcp_orientation_tracking_exp(env, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, ReferenceBankCommand)
    robot = env.scene[asset_cfg.name]
    error = torch.mean(quat_error_magnitude(robot.data.site_quat_w[:, asset_cfg.site_ids], command.tcp_quat) ** 2, dim=-1)
    return torch.exp(-error / std**2)


def reset_to_reference(env, env_ids: torch.Tensor, command_name: str) -> None:
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, ReferenceBankCommand)
    robot = env.scene["robot"]
    robot.write_joint_state_to_sim(command.qpos[env_ids], command.qvel[env_ids], env_ids=env_ids)
    robot.reset(env_ids=env_ids)

"""Task-agnostic tracking observations, rewards, and reset for ARX references."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude


def reference_command(env, command_name: str) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    return command.command


def target_pose_command(env, command_name: str) -> torch.Tensor:
    """Return the dual-TCP pose command, matching IsaacLab Reach's pose command obs."""
    command = _reference_like_command(env, command_name)
    return torch.cat(
        (
            command.tcp_pos.reshape(env.num_envs, -1),
            command.tcp_quat.reshape(env.num_envs, -1),
        ),
        dim=-1,
    )


def tcp_target_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    robot = env.scene[asset_cfg.name]
    return (command.tcp_pos - robot.data.site_pos_w[:, asset_cfg.site_ids]).reshape(env.num_envs, -1)


def joint_tracking_exp(env, command_name: str, std: float) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    robot = env.scene["robot"]
    arm_joint_ids = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)
    error = torch.mean(
        torch.square(robot.data.joint_pos[:, arm_joint_ids] - command.qpos[:, arm_joint_ids]),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def tcp_position_tracking_exp(env, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = torch.mean(torch.sum(torch.square(robot.data.site_pos_w[:, asset_cfg.site_ids] - command.tcp_pos), dim=-1), dim=-1)
    return torch.exp(-error / std**2)


def tcp_orientation_tracking_exp(env, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = torch.mean(quat_error_magnitude(robot.data.site_quat_w[:, asset_cfg.site_ids], command.tcp_quat) ** 2, dim=-1)
    return torch.exp(-error / std**2)


def tcp_position_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = torch.norm(robot.data.site_pos_w[:, asset_cfg.site_ids] - command.tcp_pos, dim=-1)
    return torch.mean(error, dim=-1)


def tcp_orientation_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = quat_error_magnitude(robot.data.site_quat_w[:, asset_cfg.site_ids], command.tcp_quat)
    return torch.mean(error, dim=-1)


def transition_success(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    position_threshold: float,
    orientation_threshold: float,
) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    robot = env.scene[asset_cfg.name]
    position_error = torch.norm(robot.data.site_pos_w[:, asset_cfg.site_ids] - command.tcp_pos, dim=-1)
    orientation_error = quat_error_magnitude(
        robot.data.site_quat_w[:, asset_cfg.site_ids],
        command.tcp_quat,
    )
    return torch.logical_and(
        torch.all(position_error <= position_threshold, dim=-1),
        torch.all(orientation_error <= orientation_threshold, dim=-1),
    )


def transition_success_reward(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    position_threshold: float,
    orientation_threshold: float,
) -> torch.Tensor:
    return transition_success(
        env,
        command_name,
        asset_cfg,
        position_threshold,
        orientation_threshold,
    ).to(dtype=torch.float32)


def action_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def episode_progress(env) -> torch.Tensor:
    """Return normalized [0, 1] episode progress for transition observations."""
    device = env.scene["robot"].data.joint_pos.device
    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is None:
        return torch.zeros(env.num_envs, 1, device=device)
    max_episode_length = float(_max_episode_length(env))
    progress = episode_length.to(device=device, dtype=torch.float32) / max(max_episode_length - 1.0, 1.0)
    return torch.clamp(progress, 0.0, 1.0).unsqueeze(-1)


def joint_velocity_tracking_exp(env, command_name: str, std: float) -> torch.Tensor:
    command = _reference_like_command(env, command_name)
    robot = env.scene["robot"]
    arm_joint_ids = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)
    error = torch.mean(
        torch.square(robot.data.joint_vel[:, arm_joint_ids] - command.qvel[:, arm_joint_ids]),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def final_joint_tracking_exp(env, command_name: str, std: float) -> torch.Tensor:
    return _final_mask(env) * joint_tracking_exp(env, command_name, std)


def final_tcp_position_tracking_exp(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    return _final_mask(env) * tcp_position_tracking_exp(env, command_name, asset_cfg, std)


def final_tcp_orientation_tracking_exp(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    return _final_mask(env) * tcp_orientation_tracking_exp(env, command_name, asset_cfg, std)


def final_joint_velocity_tracking_exp(env, command_name: str, std: float) -> torch.Tensor:
    return _final_mask(env) * joint_velocity_tracking_exp(env, command_name, std)


def reset_to_reference(env, env_ids: torch.Tensor, command_name: str) -> None:
    command = _reference_like_command(env, command_name)
    robot = env.scene["robot"]
    robot.write_joint_state_to_sim(command.qpos[env_ids], command.qvel[env_ids], env_ids=env_ids)
    robot.reset(env_ids=env_ids)


def _reference_like_command(env, command_name: str):
    command = env.command_manager.get_term(command_name)
    required = ("command", "qpos", "qvel", "tcp_pos", "tcp_quat")
    if not all(hasattr(command, name) for name in required):
        raise TypeError(f"Command '{command_name}' does not expose ARX reference targets")
    return command


def _max_episode_length(env) -> int:
    value = getattr(env, "max_episode_length", None)
    if value is not None:
        return int(value)
    return int(round(env.cfg.episode_length_s / env.step_dt))


def _final_mask(env) -> torch.Tensor:
    device = env.scene["robot"].data.joint_pos.device
    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is None:
        return torch.ones(env.num_envs, device=device)
    final_step = _max_episode_length(env) - 1
    return (episode_length.to(device=device) >= final_step).to(dtype=torch.float32)

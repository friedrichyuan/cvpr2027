"""Roll out a transition policy and save a prefix for an EgoDex ARX reference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from .initial_reference_command import InitialReferenceCommand
from .play import runner_cfg
from .transition_env_cfg import make_transition_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episode-length-s", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reference_files = sorted(reference_dir.glob("*.npz"))
    matches = [i for i, path in enumerate(reference_files) if path.stem == Path(args.trajectory).stem]
    if len(matches) != 1:
        raise FileNotFoundError(f"Reference '{args.trajectory}' was not found")
    trajectory_id = matches[0]
    steps = args.steps or int(np.ceil(args.episode_length_s / 0.02))

    cfg = make_transition_env_cfg(
        str(reference_dir),
        num_envs=1,
        episode_length_s=args.episode_length_s,
    )
    cfg.commands["reference"].fixed_trajectory_id = trajectory_id
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    if args.checkpoint is None:
        action_dim = env.action_space.shape[1]
        policy = lambda _obs: torch.zeros((1, action_dim), device=args.device)
        label = "zero_residual"
    else:
        runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg()), device=args.device)
        runner.load(
            str(args.checkpoint.expanduser().resolve()),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        label = str(args.checkpoint)

    obs, _ = wrapped.reset()
    qpos, qvel, tcp_pos, tcp_quat = [], [], [], []
    try:
        for _ in range(steps):
            obs, _, _, _ = wrapped.step(policy(obs))
            robot = env.scene["robot"]
            site_ids = [robot.site_names.index("left_tcp"), robot.site_names.index("right_tcp")]
            qpos.append(robot.data.joint_pos[0].cpu().numpy())
            qvel.append(robot.data.joint_vel[0].cpu().numpy())
            tcp_pos.append(robot.data.site_pos_w[0, site_ids].cpu().numpy())
            tcp_quat.append(robot.data.site_quat_w[0, site_ids].cpu().numpy())

        command = env.command_manager.get_term("reference")
        assert isinstance(command, InitialReferenceCommand)
        robot = env.scene["robot"]
        site_ids = [robot.site_names.index("left_tcp"), robot.site_names.index("right_tcp")]
        final_qpos = robot.data.joint_pos[0]
        final_qvel = robot.data.joint_vel[0]
        final_tcp_pos = robot.data.site_pos_w[0, site_ids]
        final_tcp_quat = robot.data.site_quat_w[0, site_ids]
        joint_mae = torch.mean(torch.abs(final_qpos - command.qpos[0])).item()
        qvel_mae = torch.mean(torch.abs(final_qvel - command.qvel[0])).item()
        tcp_error = torch.norm(final_tcp_pos - command.tcp_pos[0], dim=-1).mean().item()
        quat_dot = torch.sum(final_tcp_quat * command.tcp_quat[0], dim=-1).abs()
        orientation_error = torch.mean(2.0 * torch.acos(torch.clamp(quat_dot, -1.0, 1.0))).item()
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prefix_qpos=np.asarray(qpos, dtype=np.float32),
        prefix_qvel=np.asarray(qvel, dtype=np.float32),
        prefix_tcp_pos=np.asarray(tcp_pos, dtype=np.float32),
        prefix_tcp_quat_wxyz=np.asarray(tcp_quat, dtype=np.float32),
        source_reference=str(reference_files[trajectory_id]),
        agent=label,
        fps=np.float32(50.0),
        joint_mae=np.float32(joint_mae),
        qvel_mae=np.float32(qvel_mae),
        tcp_error=np.float32(tcp_error),
        orientation_error=np.float32(orientation_error),
    )
    print(f"Saved {output_path}")
    print(f"Final joint MAE: {joint_mae:.5f} rad")
    print(f"Final qvel MAE: {qvel_mae:.5f} rad/s")
    print(f"Final TCP error: {tcp_error * 1000.0:.1f} mm")
    print(f"Final orientation error: {np.degrees(orientation_error):.1f} deg")


if __name__ == "__main__":
    main()

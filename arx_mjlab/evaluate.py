"""Roll out a policy or zero-residual baseline and save its ARX qpos sequence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from .env_cfg import make_env_cfg
from .play import runner_cfg
from .reference_bank import ReferenceBankCommand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_files = sorted(args.reference_dir.expanduser().glob("*.npz"))
    matches = [i for i, path in enumerate(reference_files) if path.stem == Path(args.trajectory).stem]
    if len(matches) != 1:
        raise FileNotFoundError(f"Reference '{args.trajectory}' was not found")
    trajectory_id = matches[0]
    reference = np.load(reference_files[trajectory_id], allow_pickle=False)
    steps = args.steps or int(np.ceil(reference["qpos_ref"].shape[0] / 30.0 / 0.02))

    cfg = make_env_cfg(str(args.reference_dir), num_envs=1)
    cfg.commands["reference"].fixed_trajectory_id = trajectory_id
    cfg.episode_length_s = 1.0e6
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    if args.checkpoint is None:
        action_dim = env.action_space.shape[1]
        policy = lambda _obs: torch.zeros((1, action_dim), device=args.device)
        label = "zero_residual"
    else:
        runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg()), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        label = str(args.checkpoint)

    obs, _ = wrapped.reset()
    qpos, joint_error, tcp_error = [], [], []
    try:
        for _ in range(steps):
            obs, _, _, _ = wrapped.step(policy(obs))
            command = env.command_manager.get_term("reference")
            assert isinstance(command, ReferenceBankCommand)
            robot = env.scene["robot"]
            qpos.append(robot.data.joint_pos[0].cpu().numpy())
            joint_error.append(torch.mean(torch.abs(robot.data.joint_pos - command.qpos)).item())
            site_ids = [robot.site_names.index("left_tcp"), robot.site_names.index("right_tcp")]
            tcp_error.append(torch.norm(robot.data.site_pos_w[:, site_ids] - command.tcp_pos, dim=-1).mean().item())
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=np.asarray(qpos, dtype=np.float32),
        joint_error=np.asarray(joint_error, dtype=np.float32),
        tcp_error=np.asarray(tcp_error, dtype=np.float32),
        source_reference=str(reference_files[trajectory_id]),
        agent=label,
        fps=np.float32(50.0),
    )
    print(f"Saved {args.output}")
    print(f"Mean joint MAE: {np.mean(joint_error):.5f} rad")
    print(f"Mean TCP error: {np.mean(tcp_error) * 1000:.1f} mm")


if __name__ == "__main__":
    main()

"""Evaluate a transition checkpoint and replay prefix + EgoDex reference in MuJoCo."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np
import torch

from egodex_arx_replay.defaults import DEFAULT_SCENE
from egodex_arx_replay.reference import load_reference
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from .initial_reference_command import InitialReferenceCommand
from .play import runner_cfg
from .transition_env_cfg import make_transition_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--trajectory", required=True, help="Reference filename or stem")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Transition policy checkpoint")
    parser.add_argument("--output", type=Path, default=None, help="Optional NPZ rollout dump")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--episode-length-s", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--no-viewer", action="store_true", help="Only save/evaluate, do not launch MuJoCo")
    return parser.parse_args()


def rollout_prefix(
    reference_dir: Path,
    trajectory: str,
    checkpoint: Path | None,
    device: str,
    episode_length_s: float,
    steps: int | None,
) -> dict[str, np.ndarray | str | float]:
    reference_files = sorted(reference_dir.expanduser().resolve().glob("*.npz"))
    trajectory_id = _resolve_trajectory_id(reference_files, trajectory)
    steps = steps or int(np.ceil(episode_length_s / 0.02))

    cfg = make_transition_env_cfg(str(reference_dir), num_envs=1, episode_length_s=episode_length_s)
    cfg.commands["reference"].fixed_trajectory_id = trajectory_id
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    policy, agent_label = _make_policy(wrapped, env, checkpoint, device)

    obs, _ = wrapped.reset()
    qpos, qvel = [], []
    try:
        for _ in range(steps):
            obs, _, _, _ = wrapped.step(policy(obs))
            robot = env.scene["robot"]
            qpos.append(robot.data.joint_pos[0].cpu().numpy())
            qvel.append(robot.data.joint_vel[0].cpu().numpy())

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

    reference = load_reference(reference_files[trajectory_id])
    prefix_qpos = np.asarray(qpos, dtype=np.float32)
    full_qpos = np.concatenate((prefix_qpos, reference.qpos_ref[1:]), axis=0)
    return {
        "prefix_qpos": prefix_qpos,
        "prefix_qvel": np.asarray(qvel, dtype=np.float32),
        "full_qpos": full_qpos.astype(np.float32),
        "source_reference": str(reference_files[trajectory_id]),
        "agent": agent_label,
        "fps": np.float32(50.0),
        "joint_mae": np.float32(joint_mae),
        "qvel_mae": np.float32(qvel_mae),
        "tcp_error": np.float32(tcp_error),
        "orientation_error": np.float32(orientation_error),
    }


def replay_qpos(scene_path: Path, qpos: np.ndarray, fps: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path.expanduser().resolve()))
    data = mujoco.MjData(model)
    frame_time = 1.0 / fps
    frame = 0
    paused = False

    def on_key(keycode: int) -> None:
        nonlocal frame, paused
        if keycode == 32:  # Space
            paused = not paused
        elif keycode in (74, 263):  # J or left arrow
            paused = True
            frame = max(0, frame - 1)
        elif keycode in (76, 262):  # L or right arrow
            paused = True
            frame = min(len(qpos) - 1, frame + 1)
        elif keycode in (82, 114):  # R/r
            frame = 0

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        last_update = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            if not paused and now - last_update >= frame_time:
                frame = (frame + 1) % len(qpos)
                last_update = now
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.005)


def _make_policy(wrapped, env, checkpoint: Path | None, device: str):
    if checkpoint is None:
        action_dim = env.action_space.shape[1]
        return lambda _obs: torch.zeros((1, action_dim), device=device), "zero_residual"
    runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg()), device=device)
    checkpoint = checkpoint.expanduser().resolve()
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device), str(checkpoint)


def _resolve_trajectory_id(reference_files: list[Path], trajectory: str) -> int:
    if not reference_files:
        raise FileNotFoundError("No .npz references found")
    requested = Path(trajectory).stem
    matches = [index for index, path in enumerate(reference_files) if path.stem == requested]
    if len(matches) != 1:
        raise FileNotFoundError(f"Reference '{trajectory}' was not found")
    return matches[0]


def main() -> None:
    args = parse_args()
    rollout = rollout_prefix(
        args.reference_dir,
        args.trajectory,
        args.checkpoint,
        args.device,
        args.episode_length_s,
        args.steps,
    )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **rollout)
        print(f"Saved {output}")
    print(f"Source reference: {rollout['source_reference']}")
    print(f"Final joint MAE: {float(rollout['joint_mae']):.5f} rad")
    print(f"Final qvel MAE: {float(rollout['qvel_mae']):.5f} rad/s")
    print(f"Final TCP error: {float(rollout['tcp_error']) * 1000.0:.1f} mm")
    print(f"Final orientation error: {np.degrees(float(rollout['orientation_error'])):.1f} deg")
    if not args.no_viewer:
        replay_qpos(args.scene, rollout["full_qpos"], args.fps)


if __name__ == "__main__":
    main()

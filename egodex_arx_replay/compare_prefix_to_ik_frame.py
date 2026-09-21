"""Fixed viewer comparing EgoDex IK frame 0 with transition prefix final qpos."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

from arx_mjlab.replay_transition import rollout_prefix

from .compare_ik_frames import _draw_body_graph, _draw_tcp_delta
from .data import load_episode
from .defaults import DEFAULT_EPISODE, DEFAULT_SCENE
from .geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from .gripper import convert_episode_to_grippers
from .ik import ARXDualArmIKSolver
from .replay import _print_summary
from .smoothing import SmoothingConfig, smooth_gripper_trajectory


def compare_prefix_to_ik_frame(
    episode_path: Path,
    scene_path: Path,
    scene_anchor: np.ndarray,
    reference_dir: Path,
    trajectory: str,
    frame: int,
    episode_length_s: float,
    checkpoint: Path | None,
    device: str,
    smoothing_window: int,
    no_smoothing: bool,
    no_viewer: bool,
) -> None:
    """Show IK qpos[frame] and the final qpos reached by the transition prefix."""
    episode = load_episode(episode_path)
    scene_T_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    targets = convert_episode_to_grippers(episode, scene_T_egodex)
    if not no_smoothing:
        targets = smooth_gripper_trajectory(targets, SmoothingConfig(window=smoothing_window))

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    ik = ARXDualArmIKSolver(model).solve_episode(targets)
    if not 0 <= frame < ik.qpos.shape[0]:
        raise ValueError(f"Frame must be in [0, {ik.qpos.shape[0] - 1}]")

    rollout = rollout_prefix(
        reference_dir=reference_dir,
        trajectory=trajectory,
        checkpoint=checkpoint,
        device=device,
        episode_length_s=episode_length_s,
        steps=None,
    )
    prefix_qpos = np.asarray(rollout["prefix_qpos"], dtype=np.float64)
    if prefix_qpos.size == 0:
        raise ValueError("Transition prefix did not produce any qpos frames")
    prefix_final_qpos = prefix_qpos[-1]

    ik_data = mujoco.MjData(model)
    prefix_data = mujoco.MjData(model)
    ik_data.qpos[:] = ik.qpos[frame]
    prefix_data.qpos[:] = prefix_final_qpos
    data.qpos[:] = ik.qpos[frame]
    mujoco.mj_forward(model, ik_data)
    mujoco.mj_forward(model, prefix_data)
    mujoco.mj_forward(model, data)

    _print_summary(episode, scene_path, ik)
    _print_prefix_delta(model, ik_data, prefix_data, frame, rollout)
    print("Viewer: solid robot/green is IK frame; red is transition prefix final.")
    if no_viewer:
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.qpos[:] = ik.qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.user_scn.ngeom = 0
            _draw_body_graph(viewer.user_scn, model, ik_data, (0.1, 1.0, 0.25, 0.9))
            _draw_body_graph(viewer.user_scn, model, prefix_data, (1.0, 0.1, 0.1, 0.9))
            _draw_tcp_delta(
                viewer.user_scn,
                model,
                ik_data,
                prefix_data,
                (0.1, 1.0, 0.25, 0.9),
                (1.0, 0.1, 0.1, 0.9),
                (1.0, 1.0, 1.0, 0.85),
            )
            viewer.sync()
            time.sleep(0.02)


def _print_prefix_delta(
    model: mujoco.MjModel,
    ik_data: mujoco.MjData,
    prefix_data: mujoco.MjData,
    frame: int,
    rollout: dict[str, np.ndarray | str | float],
) -> None:
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("left_tcp", "right_tcp")
    ]
    delta = np.linalg.norm(prefix_data.site_xpos[site_ids] - ik_data.site_xpos[site_ids], axis=-1)
    print(
        f"TCP delta IK frame {frame} -> prefix final: "
        f"left {delta[0] * 1000.0:.1f} mm | right {delta[1] * 1000.0:.1f} mm"
    )
    print(
        "Prefix rollout final metrics: "
        f"tcp_error {float(rollout['tcp_error']) * 1000.0:.1f} mm | "
        f"orientation_error {np.degrees(float(rollout['orientation_error'])):.1f} deg"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE, help="EgoDex .hdf5 episode")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="ARX5 MuJoCo scene.xml")
    parser.add_argument("--reference-dir", type=Path, default=Path("references/arx5_transition/stack"))
    parser.add_argument("--trajectory", default="0")
    parser.add_argument("--frame", type=int, default=0, help="IK frame to compare against prefix final")
    parser.add_argument("--episode-length-s", type=float, default=4.0)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional transition policy checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument(
        "--scene-anchor",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=tuple(DEFAULT_SCENE_ANCHOR),
    )
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {scene_path}")
    compare_prefix_to_ik_frame(
        args.episode,
        scene_path,
        np.asarray(args.scene_anchor, dtype=np.float64),
        args.reference_dir.expanduser().resolve(),
        args.trajectory,
        args.frame,
        args.episode_length_s,
        args.checkpoint,
        args.device,
        args.smoothing_window,
        args.no_smoothing,
        args.no_viewer,
    )


if __name__ == "__main__":
    main()

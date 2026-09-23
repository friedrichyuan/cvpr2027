"""Kinematic GLFW replay of ARX IK qpos trajectories for EgoDex episodes."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Keep the backend choice consistent with replay.py.
os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

from .compare_ik_frames import _draw_body_graph
from .data import load_episode
from .defaults import DEFAULT_EPISODE, DEFAULT_SCENE, EGODEX_FPS
from .geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from .gripper import convert_episode_to_grippers
from .ik import ARXDualArmIKSolver
from .replay import ReplayController, _draw_overlay, _print_summary
from .smoothing import SmoothingConfig, smooth_gripper_trajectory


def replay_ik_kinematic(
    episode_path: Path,
    scene_path: Path,
    scene_anchor: np.ndarray,
    smoothing_window: int,
    no_smoothing: bool,
    show_skeleton: bool,
    show_targets: bool,
    pin_frame: int | None,
    no_viewer: bool,
) -> None:
    """Replay IK qpos directly, without actuator controls or MuJoCo dynamics."""
    episode = load_episode(episode_path)
    scene_T_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    target_grippers = convert_episode_to_grippers(episode, scene_T_egodex)
    if not no_smoothing:
        target_grippers = smooth_gripper_trajectory(
            target_grippers,
            SmoothingConfig(window=smoothing_window),
        )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    ik_trajectory = ARXDualArmIKSolver(model).solve_episode(target_grippers)
    pinned_data = None
    if pin_frame is not None:
        if not 0 <= pin_frame < ik_trajectory.qpos.shape[0]:
            raise ValueError(f"pin_frame must be in [0, {ik_trajectory.qpos.shape[0] - 1}]")
        pinned_data = mujoco.MjData(model)
        pinned_data.qpos[:] = ik_trajectory.qpos[pin_frame]
        mujoco.mj_forward(model, pinned_data)

    _print_summary(episode, scene_path, ik_trajectory)
    print("Replay mode: kinematic IK qpos only; no ctrl targets, no mj_step dynamics.")
    if pin_frame is not None:
        print(f"Pinned overlay: fixed IK qpos[{pin_frame}] skeleton in green.")
    if no_viewer:
        return

    controller = ReplayController(episode.frame_count, EGODEX_FPS)
    with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
        while viewer.is_running():
            frame = controller.advance()
            data.qpos[:] = ik_trajectory.qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            _draw_overlay(
                viewer,
                episode,
                scene_T_egodex,
                frame,
                gripper_trajectory=target_grippers if show_targets else None,
                show_skeleton=show_skeleton,
            )
            if pinned_data is not None:
                _draw_body_graph(viewer.user_scn, model, pinned_data, (0.1, 1.0, 0.25, 0.9))
            viewer.sync()
            time.sleep(0.005)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE, help="EgoDex .hdf5 episode")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="ARX5 MuJoCo scene.xml")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=9,
        help="Odd target smoothing window in frames (default: 9)",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Use unsmoothed hand-to-gripper targets.",
    )
    parser.add_argument(
        "--scene-anchor",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=tuple(DEFAULT_SCENE_ANCHOR),
        help="ARX-scene position assigned to the first-frame EgoDex hip.",
    )
    parser.add_argument("--hide-skeleton", action="store_true", help="Do not draw the EgoDex skeleton overlay.")
    parser.add_argument("--hide-targets", action="store_true", help="Do not draw converted gripper target overlays.")
    parser.add_argument(
        "--pin-frame",
        type=int,
        default=0,
        help="Draw a fixed IK qpos frame as a green skeleton overlay (default: 0).",
    )
    parser.add_argument("--no-pin-frame", action="store_true", help="Disable the fixed IK qpos skeleton overlay.")
    parser.add_argument("--no-viewer", action="store_true", help="Compute IK and print summary without opening GLFW.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {scene_path}")
    replay_ik_kinematic(
        args.episode,
        scene_path,
        np.asarray(args.scene_anchor, dtype=np.float64),
        args.smoothing_window,
        args.no_smoothing,
        show_skeleton=not args.hide_skeleton,
        show_targets=not args.hide_targets,
        pin_frame=None if args.no_pin_frame else args.pin_frame,
        no_viewer=args.no_viewer,
    )


if __name__ == "__main__":
    main()

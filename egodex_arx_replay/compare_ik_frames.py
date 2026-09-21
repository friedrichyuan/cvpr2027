"""Fixed MuJoCo viewer comparing two kinematic IK qpos frames."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

from .data import load_episode
from .defaults import DEFAULT_EPISODE, DEFAULT_SCENE
from .geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from .gripper import convert_episode_to_grippers
from .ik import ARXDualArmIKSolver
from .replay import _add_capsule, _add_sphere, _draw_gripper, _print_summary
from .smoothing import SmoothingConfig, smooth_gripper_trajectory


def compare_ik_frames(
    episode_path: Path,
    scene_path: Path,
    scene_anchor: np.ndarray,
    frame_a: int,
    frame_b: int,
    smoothing_window: int,
    no_smoothing: bool,
    no_viewer: bool,
) -> None:
    """Show qpos[frame_a] as the robot and overlay qpos[frame_a/frame_b] skeletons."""
    episode = load_episode(episode_path)
    scene_T_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    targets = convert_episode_to_grippers(episode, scene_T_egodex)
    if not no_smoothing:
        targets = smooth_gripper_trajectory(targets, SmoothingConfig(window=smoothing_window))

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    ghost_a = mujoco.MjData(model)
    ghost_b = mujoco.MjData(model)
    ik = ARXDualArmIKSolver(model).solve_episode(targets)
    if not 0 <= frame_a < ik.qpos.shape[0] or not 0 <= frame_b < ik.qpos.shape[0]:
        raise ValueError(f"Frames must be in [0, {ik.qpos.shape[0] - 1}]")

    data.qpos[:] = ik.qpos[frame_a]
    ghost_a.qpos[:] = ik.qpos[frame_a]
    ghost_b.qpos[:] = ik.qpos[frame_b]
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, ghost_a)
    mujoco.mj_forward(model, ghost_b)

    _print_summary(episode, scene_path, ik)
    _print_frame_delta(model, ghost_a, ghost_b, frame_a, frame_b)
    print("Viewer: solid robot is frame A; green skeleton/TCP is A; red skeleton/TCP is B.")
    if no_viewer:
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.qpos[:] = ik.qpos[frame_a]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            _draw_comparison_overlay(viewer.user_scn, model, ghost_a, ghost_b, targets, frame_a, frame_b)
            viewer.sync()
            time.sleep(0.02)


def _print_frame_delta(
    model: mujoco.MjModel,
    data_a: mujoco.MjData,
    data_b: mujoco.MjData,
    frame_a: int,
    frame_b: int,
) -> None:
    site_ids = [_site_id(model, name) for name in ("left_tcp", "right_tcp")]
    delta = np.linalg.norm(data_b.site_xpos[site_ids] - data_a.site_xpos[site_ids], axis=-1)
    print(
        f"FK TCP delta frame {frame_a}->{frame_b}: "
        f"left {delta[0] * 1000.0:.1f} mm | right {delta[1] * 1000.0:.1f} mm"
    )


def _draw_comparison_overlay(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data_a: mujoco.MjData,
    data_b: mujoco.MjData,
    targets,
    frame_a: int,
    frame_b: int,
) -> None:
    scene.ngeom = 0
    green = (0.1, 1.0, 0.25, 0.9)
    red = (1.0, 0.1, 0.1, 0.9)
    white = (1.0, 1.0, 1.0, 0.85)
    _draw_body_graph(scene, model, data_a, green)
    _draw_body_graph(scene, model, data_b, red)
    _draw_tcp_delta(scene, model, data_a, data_b, green, red, white)
    for side_index, side in enumerate(("left", "right")):
        if targets.valid[frame_a, side_index]:
            _draw_gripper(
                scene,
                targets.position[frame_a, side_index],
                targets.rotation[frame_a, side_index],
                float(targets.width[frame_a, side_index]),
                side,
            )
        if targets.valid[frame_b, side_index]:
            _draw_gripper(
                scene,
                targets.position[frame_b, side_index],
                targets.rotation[frame_b, side_index],
                float(targets.width[frame_b, side_index]),
                side,
            )


def _draw_body_graph(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rgba: tuple[float, float, float, float],
) -> None:
    for body_id in range(1, model.nbody):
        position = data.xpos[body_id]
        _add_sphere(scene, position, rgba)
        parent_id = int(model.body_parentid[body_id])
        if parent_id > 0:
            _add_capsule(scene, data.xpos[parent_id], position, 0.003, rgba)


def _draw_tcp_delta(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data_a: mujoco.MjData,
    data_b: mujoco.MjData,
    rgba_a: tuple[float, float, float, float],
    rgba_b: tuple[float, float, float, float],
    connector_rgba: tuple[float, float, float, float],
) -> None:
    for name in ("left_tcp", "right_tcp"):
        site_id = _site_id(model, name)
        start = data_a.site_xpos[site_id]
        end = data_b.site_xpos[site_id]
        _add_sphere(scene, start, rgba_a)
        _add_sphere(scene, end, rgba_b)
        _add_capsule(scene, start, end, 0.004, connector_rgba)


def _site_id(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"Scene is missing site '{name}'")
    return site_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE, help="EgoDex .hdf5 episode")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="ARX5 MuJoCo scene.xml")
    parser.add_argument("--frame-a", type=int, default=0, help="First IK qpos frame to compare")
    parser.add_argument("--frame-b", type=int, default=1, help="Second IK qpos frame to compare")
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
    compare_ik_frames(
        args.episode,
        scene_path,
        np.asarray(args.scene_anchor, dtype=np.float64),
        args.frame_a,
        args.frame_b,
        args.smoothing_window,
        args.no_smoothing,
        args.no_viewer,
    )


if __name__ == "__main__":
    main()

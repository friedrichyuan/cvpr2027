"""Replay transition prefix first, then the EgoDex IK qpos trajectory."""

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
from .defaults import DEFAULT_EPISODE, DEFAULT_SCENE, EGODEX_FPS
from .geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from .gripper import convert_episode_to_grippers
from .ik import ARXDualArmIKSolver
from .replay import _print_summary
from .replay_prefix_vs_ik_frame import _draw_eef_targets, _print_final_delta
from .smoothing import SmoothingConfig, smooth_gripper_trajectory


class VariableRateController:
    """Pauseable frame controller for a precomputed per-frame time array."""

    def __init__(self, frame_time: np.ndarray, loop: bool = True) -> None:
        self.frame_time = frame_time
        self.loop = loop
        self.frame = 0
        self.playing = True
        self.speed = 1.0
        self._last_update = time.monotonic()
        self._time_credit = 0.0

    def on_key(self, keycode: int) -> None:
        if keycode == 32:
            self.playing = not self.playing
        elif keycode in (74, 263):  # J or left arrow
            self.playing = False
            self.frame = max(0, self.frame - 1)
            self._time_credit = 0.0
        elif keycode in (76, 262):  # L or right arrow
            self.playing = False
            self.frame = min(len(self.frame_time) - 1, self.frame + 1)
            self._time_credit = 0.0
        elif keycode in (82, 114):  # R or r
            self.frame = 0
            self._time_credit = 0.0
        elif keycode in (45, 95):  # - or _
            self.speed = max(0.125, self.speed / 2.0)
        elif keycode in (61, 43):  # = or +
            self.speed = min(8.0, self.speed * 2.0)

    def advance(self) -> int:
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        if not self.playing:
            return self.frame
        self._time_credit += elapsed * self.speed
        while self.frame < len(self.frame_time) - 1 or self.loop:
            if self.frame >= len(self.frame_time) - 1:
                self.frame = 0
                continue
            duration = max(self.frame_time[self.frame + 1] - self.frame_time[self.frame], 1e-4)
            if self._time_credit < duration:
                break
            self._time_credit -= duration
            self.frame += 1
        return self.frame


def replay_prefix_then_ik(
    episode_path: Path,
    scene_path: Path,
    scene_anchor: np.ndarray,
    reference_dir: Path,
    trajectory: str,
    ik_start_frame: int,
    episode_length_s: float,
    checkpoint: Path | None,
    device: str,
    control_fps: float,
    smoothing_window: int,
    no_smoothing: bool,
    loop: bool,
    no_viewer: bool,
) -> None:
    """Play prefix rollout followed by live IK qpos frames."""
    episode = load_episode(episode_path)
    scene_T_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    targets = convert_episode_to_grippers(episode, scene_T_egodex)
    if not no_smoothing:
        targets = smooth_gripper_trajectory(targets, SmoothingConfig(window=smoothing_window))

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    ik = ARXDualArmIKSolver(model).solve_episode(targets)
    if not 0 <= ik_start_frame < ik.qpos.shape[0]:
        raise ValueError(f"ik_start_frame must be in [0, {ik.qpos.shape[0] - 1}]")

    rollout = rollout_prefix(
        reference_dir=reference_dir,
        trajectory=trajectory,
        checkpoint=checkpoint,
        device=device,
        episode_length_s=episode_length_s,
        steps=None,
        control_fps=control_fps,
    )
    prefix_qpos = np.asarray(rollout["prefix_qpos"], dtype=np.float64)
    ik_qpos = ik.qpos[ik_start_frame:].astype(np.float64)
    sequence_qpos = np.concatenate((prefix_qpos, ik_qpos), axis=0)
    frame_time = _combined_frame_time(len(prefix_qpos), len(ik_qpos), control_fps, EGODEX_FPS)

    pinned_data = mujoco.MjData(model)
    final_prefix_data = mujoco.MjData(model)
    pinned_data.qpos[:] = ik.qpos[ik_start_frame]
    final_prefix_data.qpos[:] = prefix_qpos[-1]
    mujoco.mj_forward(model, pinned_data)
    mujoco.mj_forward(model, final_prefix_data)

    _print_summary(episode, scene_path, ik)
    _print_final_delta(model, pinned_data, final_prefix_data, targets, ik_start_frame, rollout, trajectory)
    _print_stitch_delta(model, prefix_qpos[-1], ik.qpos[ik_start_frame])
    print(
        f"Replay sequence: {len(prefix_qpos)} prefix frames at {control_fps:g}Hz, "
        f"then {len(ik_qpos)} IK frames at {EGODEX_FPS:g}Hz."
    )
    if no_viewer:
        return

    controller = VariableRateController(frame_time, loop=loop)
    with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
        while viewer.is_running():
            frame = controller.advance()
            data.qpos[:] = sequence_qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            viewer.user_scn.ngeom = 0
            _draw_body_graph(viewer.user_scn, model, pinned_data, (0.1, 1.0, 0.25, 0.9))
            _draw_body_graph(viewer.user_scn, model, final_prefix_data, (1.0, 0.1, 0.1, 0.55))
            _draw_eef_targets(viewer.user_scn, targets, ik_start_frame)
            if frame >= len(prefix_qpos) - 1:
                _draw_tcp_delta(
                    viewer.user_scn,
                    model,
                    pinned_data,
                    final_prefix_data,
                    (0.1, 1.0, 0.25, 0.9),
                    (1.0, 0.1, 0.1, 0.9),
                    (1.0, 1.0, 1.0, 0.85),
                )
            viewer.sync()
            time.sleep(0.005)


def _combined_frame_time(prefix_count: int, ik_count: int, control_fps: float, ik_fps: float) -> np.ndarray:
    prefix_time = np.arange(prefix_count, dtype=np.float64) / control_fps
    ik_start_time = prefix_time[-1] if prefix_count else 0.0
    ik_time = ik_start_time + np.arange(1, ik_count + 1, dtype=np.float64) / ik_fps
    return np.concatenate((prefix_time, ik_time), axis=0)


def _print_stitch_delta(model: mujoco.MjModel, prefix_qpos: np.ndarray, ik_qpos: np.ndarray) -> None:
    prefix_data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    prefix_data.qpos[:] = prefix_qpos
    ik_data.qpos[:] = ik_qpos
    mujoco.mj_forward(model, prefix_data)
    mujoco.mj_forward(model, ik_data)
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("left_tcp", "right_tcp")
    ]
    delta = np.linalg.norm(ik_data.site_xpos[site_ids] - prefix_data.site_xpos[site_ids], axis=-1)
    print(f"Stitch delta prefix final -> first IK frame: left {delta[0] * 1000:.1f} mm | right {delta[1] * 1000:.1f} mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE, help="EgoDex .hdf5 episode")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="ARX5 MuJoCo scene.xml")
    parser.add_argument("--reference-dir", type=Path, default=Path("references/arx5_transition/stack"))
    parser.add_argument("--trajectory", default="0.untrimmed")
    parser.add_argument("--ik-start-frame", type=int, default=0)
    parser.add_argument("--episode-length-s", type=float, default=4.0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-fps", type=float, default=50.0)
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--no-loop", action="store_true", help="Stop at the last IK frame instead of looping.")
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
    replay_prefix_then_ik(
        args.episode,
        scene_path,
        np.asarray(args.scene_anchor, dtype=np.float64),
        args.reference_dir.expanduser().resolve(),
        args.trajectory,
        args.ik_start_frame,
        args.episode_length_s,
        args.checkpoint,
        args.device,
        args.control_fps,
        args.smoothing_window,
        args.no_smoothing,
        loop=not args.no_loop,
        no_viewer=args.no_viewer,
    )


if __name__ == "__main__":
    main()

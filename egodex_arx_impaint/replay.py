"""Kinematic replay of an imputed ARX trajectory with raw EgoDex EEF markers."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

from egodex_arx_replay.defaults import DEFAULT_SCENE
from egodex_arx_replay.data import load_episode
from egodex_arx_replay.geometry import DEFAULT_SCENE_ANCHOR, frame_joint_positions, make_scene_T_egodex
from egodex_arx_replay.replay import _draw_gripper, _draw_skeleton

from .base_search import _apply_base_offset


class TimeController:
    """Looping playback controller for a trajectory with per-frame timestamps."""

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
            duration = max(self.frame_time[self.frame + 1] - self.frame_time[self.frame], 1.0e-4)
            if self._time_credit < duration:
                break
            self._time_credit -= duration
            self.frame += 1
        return self.frame


def replay_impainted(
    trajectory_path: Path,
    scene_path: Path,
    loop: bool,
    show_raw_eef: bool,
    show_skeleton: bool,
    no_viewer: bool,
) -> None:
    trajectory = np.load(trajectory_path, allow_pickle=False)
    qpos = trajectory["full_qpos"]
    frame_time = trajectory["full_time"]
    eef_frame = trajectory["full_eef_frame"]
    raw_pos = trajectory["raw_eef_pos"]
    raw_rot = trajectory["raw_eef_rot"]
    raw_width = trajectory["raw_eef_width"]
    raw_valid = trajectory["raw_eef_valid"]
    metadata = _load_metadata(trajectory)
    episode, scene_t_egodex = _load_source_episode(metadata) if show_skeleton else (None, None)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _apply_base_offset(model, _metadata_base_offset(metadata))
    data = mujoco.MjData(model)
    _print_summary(trajectory_path, qpos, frame_time, raw_pos)
    if no_viewer:
        return

    controller = TimeController(frame_time.astype(np.float64), loop=loop)
    with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
        while viewer.is_running():
            frame = controller.advance()
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.user_scn.ngeom = 0
            raw_frame = int(eef_frame[frame])
            if episode is not None and scene_t_egodex is not None:
                _draw_skeleton(
                    viewer.user_scn,
                    frame_joint_positions(episode.world_T_joint, raw_frame, scene_t_egodex),
                )
            if show_raw_eef:
                _draw_raw_eef(viewer.user_scn, raw_pos, raw_rot, raw_width, raw_valid, raw_frame)
            viewer.sync()
            time.sleep(0.005)


def _draw_raw_eef(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rotation: np.ndarray,
    width: np.ndarray,
    valid: np.ndarray,
    frame: int,
) -> None:
    for side_index, side in enumerate(("left", "right")):
        if not valid[frame, side_index]:
            continue
        _draw_gripper(
            scene,
            position[frame, side_index],
            rotation[frame, side_index],
            float(width[frame, side_index]),
            side,
        )


def _print_summary(
    trajectory_path: Path,
    qpos: np.ndarray,
    frame_time: np.ndarray,
    raw_pos: np.ndarray,
) -> None:
    print(f"Trajectory: {trajectory_path}")
    print(f"Frames: {qpos.shape[0]} | duration: {frame_time[-1]:.3f}s | qpos dim: {qpos.shape[1]}")
    print(f"Raw EEF frames: {raw_pos.shape[0]}")
    print("Controls: space play/pause | J/L or arrows step | R restart | -/+ speed | viewer mouse to orbit")


def _load_metadata(trajectory: np.lib.npyio.NpzFile) -> dict:
    if "metadata" not in trajectory.files:
        return {}
    return json.loads(str(trajectory["metadata"].item()))


def _load_source_episode(metadata: dict):
    source_episode = metadata.get("source_episode")
    if not source_episode:
        print("[WARN] No source_episode in trajectory metadata; skeleton overlay disabled.")
        return None, None
    episode = load_episode(source_episode)
    scene_anchor = np.asarray(metadata.get("scene_anchor", DEFAULT_SCENE_ANCHOR), dtype=np.float64)
    return episode, make_scene_T_egodex(episode.world_T_joint, scene_anchor)


def _metadata_base_offset(metadata: dict) -> np.ndarray:
    offset = metadata.get("base_offset", {})
    return np.array(
        [
            float(offset.get("dx", 0.0)),
            float(offset.get("dy", 0.0)),
            float(offset.get("yaw_rad", 0.0)),
        ],
        dtype=np.float64,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True, help="Output .npz from impaint.py")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="ARX5 MuJoCo scene.xml")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--hide-raw-eef", action="store_true")
    parser.add_argument("--hide-skeleton", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {scene_path}")
    replay_impainted(
        args.trajectory.expanduser().resolve(),
        scene_path,
        loop=not args.no_loop,
        show_raw_eef=not args.hide_raw_eef,
        show_skeleton=not args.hide_skeleton,
        no_viewer=args.no_viewer,
    )


if __name__ == "__main__":
    main()

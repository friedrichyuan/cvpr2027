"""Interactive GLFW replay of calibrated EgoDex skeleton motion in ARX5."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# This must precede importing mujoco.  A caller can still explicitly select a
# different backend by exporting MUJOCO_GL before invoking this module.
os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

from .data import EgoDexEpisode, load_episode
from .defaults import DEFAULT_EPISODE, DEFAULT_SCENE, EGODEX_FPS
from .geometry import (
    DEFAULT_SCENE_ANCHOR,
    SKELETON_BONES,
    frame_joint_positions,
    make_scene_T_egodex,
    transform_pose,
)
from .gripper import GripperTrajectory, convert_episode_to_grippers
from .ik import ARXDualArmIKSolver, IKTrajectory
from .smoothing import SmoothingConfig, smooth_gripper_trajectory

FPS = EGODEX_FPS


class ReplayController:
    """Clock and keyboard state for frame-accurate, pauseable episode replay."""

    def __init__(self, frame_count: int, fps: float) -> None:
        self.frame_count = frame_count
        self.fps = fps
        self.frame = 0
        self.playing = True
        self.speed = 1.0
        self._last_update = time.monotonic()
        self._frame_credit = 0.0
        self._advanced_frames = 0
        self._seeked = False

    def on_key(self, keycode: int) -> None:
        # GLFW key codes: SPACE=32, R=82, J=74, L=76, -=45, +=61.
        if keycode == 32:
            self.playing = not self.playing
        elif keycode in (74, 263):  # J or left arrow
            self.playing = False
            self.frame = max(0, self.frame - 1)
            self._seeked = True
        elif keycode in (76, 262):  # L or right arrow
            self.playing = False
            self.frame = min(self.frame_count - 1, self.frame + 1)
            self._seeked = True
        elif keycode in (82, 114):  # R or r
            self.frame = 0
            self._frame_credit = 0.0
            self._seeked = True
        elif keycode in (45, 95):  # - or _
            self.speed = max(0.125, self.speed / 2.0)
        elif keycode in (61, 43):  # = or +
            self.speed = min(8.0, self.speed * 2.0)

    def advance(self) -> int:
        self._advanced_frames = 0
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        if self.playing:
            # The render loop commonly runs faster than 30 Hz.  Preserve the
            # fractional remainder rather than truncating every 5 ms update.
            self._frame_credit += elapsed * self.fps * self.speed
            step_count = int(self._frame_credit)
            if step_count:
                self.frame = (self.frame + step_count) % self.frame_count
                self._frame_credit -= step_count
                self._advanced_frames = step_count
        return self.frame

    def consume_advanced_frames(self) -> int:
        """Return the number of forward video frames advanced this update."""
        advanced_frames = self._advanced_frames
        self._advanced_frames = 0
        return advanced_frames

    def consume_seek(self) -> bool:
        """Report and clear a keyboard seek, which cannot be simulated backward."""
        seeked = self._seeked
        self._seeked = False
        return seeked


class DynamicsFrameStepper:
    """Advance MuJoCo by one 30 Hz control interval using fixed-size substeps."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, control_fps: float) -> None:
        if control_fps <= 0.0:
            raise ValueError("Control FPS must be positive")
        if model.opt.timestep <= 0.0:
            raise ValueError("MuJoCo timestep must be positive")
        self.model = model
        self.data = data
        self._substeps_per_frame = 1.0 / (control_fps * model.opt.timestep)
        self._substep_credit = 0.0

    def reset(self) -> None:
        self._substep_credit = 0.0

    def step_frame(self, ctrl: np.ndarray) -> None:
        """Hold ``ctrl`` for one video frame while advancing real dynamics."""
        self.data.ctrl[:] = ctrl
        self._substep_credit += self._substeps_per_frame
        substep_count = int(self._substep_credit)
        self._substep_credit -= substep_count
        for _ in range(substep_count):
            mujoco.mj_step(self.model, self.data)


def _add_sphere(scene: mujoco.MjvScene, position: np.ndarray, rgba: tuple[float, ...]) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.009, 0.0, 0.0]),
        position,
        np.eye(3).flatten(),
        np.asarray(rgba),
    )
    scene.ngeom += 1


def _add_capsule(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: tuple[float, ...],
) -> None:
    if np.linalg.norm(end - start) < 1e-7:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
    scene.ngeom += 1


def _add_box(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rotation: np.ndarray,
    half_size: tuple[float, float, float],
    rgba: tuple[float, ...],
) -> None:
    """Append an oriented box to MuJoCo's viewer-only user scene."""
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(half_size),
        position,
        rotation.flatten(),
        np.asarray(rgba),
    )
    scene.ngeom += 1


def _draw_skeleton(scene: mujoco.MjvScene, positions: dict[str, np.ndarray]) -> None:
    left_rgba = (0.98, 0.33, 0.78, 0.92)
    right_rgba = (0.22, 0.65, 1.0, 0.92)
    body_rgba = (1.0, 0.78, 0.15, 0.88)

    for start_name, end_name in SKELETON_BONES:
        if start_name not in positions or end_name not in positions:
            continue
        rgba = left_rgba if start_name.startswith("left") else right_rgba
        if not (start_name.startswith("left") or start_name.startswith("right")):
            rgba = body_rgba
        _add_capsule(scene, positions[start_name], positions[end_name], 0.004, rgba)

    for name, position in positions.items():
        rgba = left_rgba if name.startswith("left") else right_rgba
        if not (name.startswith("left") or name.startswith("right")):
            rgba = body_rgba
        _add_sphere(scene, position, rgba)


def _draw_camera(scene: mujoco.MjvScene, scene_T_egodex: np.ndarray, world_T_camera: np.ndarray) -> None:
    scene_T_camera = transform_pose(scene_T_egodex, world_T_camera)
    origin = scene_T_camera[:3, 3]
    rotation = scene_T_camera[:3, :3]
    axis_colours = ((1.0, 0.1, 0.1, 0.95), (0.1, 0.9, 0.15, 0.95), (0.15, 0.35, 1.0, 0.95))
    for axis, colour in zip(rotation.T, axis_colours):
        _add_capsule(scene, origin, origin + 0.06 * axis, 0.003, colour)
    _add_sphere(scene, origin, (0.95, 0.95, 0.95, 0.95))


def _draw_gripper(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rotation: np.ndarray,
    width: float,
    side: str,
) -> None:
    """Draw a TCP frame and two flat parallel jaws for one converted hand."""
    colour = (0.98, 0.33, 0.78, 0.95) if side == "left" else (0.22, 0.65, 1.0, 0.95)
    axis_colours = ((1.0, 0.1, 0.1, 0.95), (0.1, 0.9, 0.15, 0.95), (0.15, 0.35, 1.0, 0.95))
    for axis, axis_colour in zip(rotation.T, axis_colours):
        _add_capsule(scene, position, position + 0.065 * axis, 0.003, axis_colour)

    # Local +Z/-Z are the two contact normals.  Each plate spans approach (X)
    # and palm-normal (Y), so its inner faces are exactly ``width`` apart.
    plate_thickness = 0.006
    plate_offset = 0.5 * width + 0.5 * plate_thickness
    for sign in (-1.0, 1.0):
        _add_box(
            scene,
            position + sign * plate_offset * rotation[:, 2],
            rotation,
            half_size=(0.035, 0.020, 0.5 * plate_thickness),
            rgba=colour,
        )


def _draw_grippers(scene: mujoco.MjvScene, trajectory: GripperTrajectory, frame: int) -> None:
    for side_index, side in enumerate(("left", "right")):
        if trajectory.valid[frame, side_index]:
            _draw_gripper(
                scene,
                trajectory.position[frame, side_index],
                trajectory.rotation[frame, side_index],
                float(trajectory.width[frame, side_index]),
                side,
            )


def _draw_overlay(
    viewer: mujoco.viewer.Handle,
    episode: EgoDexEpisode,
    scene_T_egodex: np.ndarray,
    frame: int,
    gripper_trajectory: GripperTrajectory | None = None,
    show_skeleton: bool = True,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    if show_skeleton:
        positions = frame_joint_positions(episode.world_T_joint, frame, scene_T_egodex)
        _draw_skeleton(scene, positions)
    if gripper_trajectory is not None:
        _draw_grippers(scene, gripper_trajectory, frame)
    _draw_camera(scene, scene_T_egodex, episode.world_T_camera[frame])


def _print_summary(
    episode: EgoDexEpisode,
    scene_path: Path,
    ik_trajectory: IKTrajectory | None = None,
) -> None:
    description = episode.metadata.get("llm_description", "")
    print(f"EgoDex episode: {episode.path}")
    print(f"Task: {episode.metadata.get('task', 'unknown')} | frames: {episode.frame_count} | FPS: {FPS:g}")
    print(f"Description: {description}")
    print(f"MuJoCo scene: {scene_path}")
    if ik_trajectory is not None:
        valid = ~np.isnan(ik_trajectory.position_error)
        position_mm = 1000.0 * ik_trajectory.position_error[valid]
        orientation_deg = np.degrees(ik_trajectory.orientation_error[valid])
        print(
            "IK target error: "
            f"median {np.median(position_mm):.1f} mm / {np.median(orientation_deg):.1f} deg, "
            f"converged {ik_trajectory.converged[valid].mean() * 100.0:.1f}%"
        )
    print("Controls: space play/pause | J/L or arrows step | R restart | -/+ speed | viewer mouse to orbit")


def replay(
    episode_path: Path,
    scene_path: Path,
    scene_anchor: np.ndarray,
    mode: str,
    smoothing_window: int,
    no_smoothing: bool,
) -> None:
    episode = load_episode(episode_path)
    scene_T_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    target_grippers = (
        convert_episode_to_grippers(episode, scene_T_egodex)
        if mode in {"gripper", "both", "ik", "all"}
        else None
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    if target_grippers is not None and not no_smoothing:
        target_grippers = smooth_gripper_trajectory(
            target_grippers,
            SmoothingConfig(window=smoothing_window),
        )
    ik_trajectory = None
    if mode in {"ik", "all"}:
        assert target_grippers is not None
        ik_trajectory = ARXDualArmIKSolver(model).solve_episode(target_grippers)

    controller = ReplayController(episode.frame_count, FPS)
    dynamics_stepper = DynamicsFrameStepper(model, data, FPS) if ik_trajectory is not None else None
    simulated_frame = controller.frame
    if ik_trajectory is not None:
        # qpos remains exclusively owned by MuJoCo after this point.  The IK
        # result supplies position-actuator setpoints; mj_step integrates the
        # response to them, including joint limits and any contacts.
        data.ctrl[:] = ik_trajectory.ctrl[simulated_frame]
        mujoco.mj_forward(model, data)

    _print_summary(episode, scene_path, ik_trajectory)
    with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
        while viewer.is_running():
            frame = controller.advance()
            if ik_trajectory is not None:
                assert dynamics_stepper is not None
                if controller.consume_seek():
                    # Stepping backward in a dynamics simulation is not
                    # meaningful.  Recreate the requested state from the
                    # keyframe by replaying the preceding 30 Hz controls.
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                    dynamics_stepper.reset()
                    simulated_frame = 0
                    for _ in range(frame):
                        dynamics_stepper.step_frame(ik_trajectory.ctrl[simulated_frame])
                        simulated_frame = (simulated_frame + 1) % episode.frame_count
                    data.ctrl[:] = ik_trajectory.ctrl[simulated_frame]
                    mujoco.mj_forward(model, data)
                else:
                    for _ in range(controller.consume_advanced_frames()):
                        dynamics_stepper.step_frame(ik_trajectory.ctrl[simulated_frame])
                        simulated_frame = (simulated_frame + 1) % episode.frame_count
                    data.ctrl[:] = ik_trajectory.ctrl[simulated_frame]
            _draw_overlay(
                viewer,
                episode,
                scene_T_egodex,
                frame,
                gripper_trajectory=target_grippers,
                show_skeleton=mode in {"skeleton", "both", "all"},
            )
            viewer.sync()
            time.sleep(0.005)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE, help="EgoDex .hdf5 episode")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="ARX5 MuJoCo scene.xml")
    parser.add_argument(
        "--mode",
        choices=("skeleton", "gripper", "both", "ik", "all"),
        default="skeleton",
        help="Draw skeleton, gripper targets, target+robot IK, or all overlays (default: skeleton)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=9,
        help="Odd target smoothing window in frames (default: 9)",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Use unsmoothed hand-to-gripper targets (useful for an ablation)",
    )
    parser.add_argument(
        "--scene-anchor",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=tuple(DEFAULT_SCENE_ANCHOR),
        help="ARX-scene position assigned to the first-frame EgoDex hip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {scene_path}")
    replay(
        args.episode,
        scene_path,
        np.asarray(args.scene_anchor, dtype=np.float64),
        args.mode,
        args.smoothing_window,
        args.no_smoothing,
    )


if __name__ == "__main__":
    main()

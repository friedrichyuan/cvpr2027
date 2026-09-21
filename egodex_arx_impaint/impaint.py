"""Generate a zero-to-EgoDex-prefix plus per-frame ARX IK trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from egodex_arx_replay.data import load_episode
from egodex_arx_replay.defaults import DEFAULT_EPISODE, DEFAULT_SCENE, EGODEX_FPS
from egodex_arx_replay.geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from egodex_arx_replay.gripper import GripperTrajectory, convert_episode_to_grippers
from egodex_arx_replay.ik import ARXDualArmIKSolver
from egodex_arx_replay.smoothing import SmoothingConfig, smooth_gripper_trajectory


def generate_impainted_trajectory(
    episode_path: Path,
    scene_path: Path,
    output_path: Path,
    scene_anchor: np.ndarray,
    prefix_duration_s: float,
    control_fps: float,
    smoothing_window: int,
    no_smoothing: bool,
    end_velocity_scale: float,
    max_end_velocity: float,
) -> None:
    """Solve per-frame IK, then prepend a smooth zero-to-first-frame prefix."""
    episode = load_episode(episode_path)
    scene_t_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    raw_targets = convert_episode_to_grippers(episode, scene_t_egodex)
    targets = raw_targets if no_smoothing else smooth_gripper_trajectory(
        raw_targets,
        SmoothingConfig(window=smoothing_window),
    )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    ik = ARXDualArmIKSolver(model).solve_episode(targets)
    qvel_ref = np.gradient(ik.qpos, 1.0 / EGODEX_FPS, axis=0, edge_order=1)

    prefix_time, prefix_qpos = _make_prefix(
        target_qpos=ik.qpos[0],
        target_qvel=np.clip(end_velocity_scale * qvel_ref[0], -max_end_velocity, max_end_velocity),
        duration_s=prefix_duration_s,
        control_fps=control_fps,
    )
    ik_time = prefix_time[-1] + np.arange(1, ik.qpos.shape[0], dtype=np.float64) / EGODEX_FPS
    full_qpos = np.concatenate((prefix_qpos, ik.qpos[1:]), axis=0)
    full_time = np.concatenate((prefix_time, ik_time), axis=0)
    full_eef_frame = np.concatenate((
        np.zeros(prefix_qpos.shape[0], dtype=np.int32),
        np.arange(1, ik.qpos.shape[0], dtype=np.int32),
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_episode": str(episode.path),
        "scene_path": str(scene_path),
        "fps": EGODEX_FPS,
        "control_fps": control_fps,
        "prefix_duration_s": prefix_duration_s,
        "scene_anchor": np.asarray(scene_anchor, dtype=np.float64).tolist(),
        "smoothing_window": None if no_smoothing else smoothing_window,
        "ik_solver": "first_frame_independent_then_warm_start",
    }
    np.savez_compressed(
        output_path,
        prefix_qpos=prefix_qpos.astype(np.float32),
        prefix_time=prefix_time.astype(np.float32),
        ik_qpos=ik.qpos.astype(np.float32),
        ik_qvel=qvel_ref.astype(np.float32),
        ik_position_error=ik.position_error.astype(np.float32),
        ik_orientation_error=ik.orientation_error.astype(np.float32),
        ik_converged=ik.converged,
        full_qpos=full_qpos.astype(np.float32),
        full_time=full_time.astype(np.float32),
        full_eef_frame=full_eef_frame,
        raw_eef_pos=raw_targets.position.astype(np.float32),
        raw_eef_rot=raw_targets.rotation.astype(np.float32),
        raw_eef_width=raw_targets.width.astype(np.float32),
        raw_eef_valid=raw_targets.valid,
        target_eef_pos=targets.position.astype(np.float32),
        target_eef_rot=targets.rotation.astype(np.float32),
        target_eef_width=targets.width.astype(np.float32),
        target_eef_valid=targets.valid,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    _print_summary(output_path, ik, prefix_qpos, raw_targets, targets)


def _make_prefix(
    target_qpos: np.ndarray,
    target_qvel: np.ndarray,
    duration_s: float,
    control_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if duration_s <= 0.0:
        raise ValueError("prefix duration must be positive")
    steps = max(2, int(round(duration_s * control_fps)) + 1)
    time = np.arange(steps, dtype=np.float64) / control_fps
    time[-1] = duration_s
    progress = np.clip(time / duration_s, 0.0, 1.0)[:, None]
    min_jerk = progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)
    velocity_basis = progress**3 * (-4.0 + 7.0 * progress - 3.0 * progress**2)
    qpos = min_jerk * target_qpos[None, :] + velocity_basis * duration_s * target_qvel[None, :]
    qpos[0] = 0.0
    qpos[-1] = target_qpos
    return time, qpos


def _print_summary(
    output_path: Path,
    ik,
    prefix_qpos: np.ndarray,
    raw_targets: GripperTrajectory,
    targets: GripperTrajectory,
) -> None:
    valid = targets.valid
    pos_mm = 1000.0 * ik.position_error[valid]
    ori_deg = np.degrees(ik.orientation_error[valid])
    first_target_delta = np.linalg.norm(targets.position[0] - raw_targets.position[0], axis=-1)
    print(f"Saved {output_path}")
    print(f"Prefix frames: {prefix_qpos.shape[0]} | IK frames: {ik.qpos.shape[0]}")
    print(
        "IK target error: "
        f"median {np.median(pos_mm):.3f} mm / {np.median(ori_deg):.3f} deg, "
        f"converged {ik.converged[valid].mean() * 100.0:.1f}%"
    )
    print(
        "First target smoothing delta: "
        f"left {first_target_delta[0] * 1000.0:.3f} mm | "
        f"right {first_target_delta[1] * 1000.0:.3f} mm"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE, help="EgoDex .hdf5 episode")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="ARX5 MuJoCo scene.xml")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz trajectory")
    parser.add_argument("--prefix-duration-s", type=float, default=1.0)
    parser.add_argument("--control-fps", type=float, default=50.0)
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--end-velocity-scale", type=float, default=1.0)
    parser.add_argument("--max-end-velocity", type=float, default=5.0)
    parser.add_argument(
        "--scene-anchor",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=tuple(DEFAULT_SCENE_ANCHOR),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {scene_path}")
    generate_impainted_trajectory(
        args.episode,
        scene_path,
        args.output.expanduser().resolve(),
        np.asarray(args.scene_anchor, dtype=np.float64),
        args.prefix_duration_s,
        args.control_fps,
        args.smoothing_window,
        args.no_smoothing,
        args.end_velocity_scale,
        args.max_end_velocity,
    )


if __name__ == "__main__":
    main()

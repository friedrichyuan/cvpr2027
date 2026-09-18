"""Versioned offline ARX reference trajectories for later batched RL training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .data import load_episode
from .defaults import DEFAULT_SCENE, EGODEX_FPS
from .geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from .gripper import convert_episode_to_grippers
from .ik import ARXDualArmIKSolver, IKConfig
from .smoothing import SmoothingConfig, smooth_gripper_trajectory

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReferenceTrajectory:
    """One self-contained ARX reference trajectory at the source frame rate.

    The file deliberately stores both the robot IK solution and its Cartesian
    target.  RL can therefore use the IK sequence as a residual-control prior
    while independently measuring target fidelity and reference feasibility.
    """

    qpos_ref: np.ndarray
    qvel_ref: np.ndarray
    ctrl_ref: np.ndarray
    tcp_pos_ref: np.ndarray
    tcp_quat_wxyz_ref: np.ndarray
    gripper_width_ref: np.ndarray
    target_valid: np.ndarray
    ik_position_error: np.ndarray
    ik_orientation_error: np.ndarray
    ik_converged: np.ndarray
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.qpos_ref.shape[0])

    @property
    def fps(self) -> float:
        return float(self.metadata["fps"])


def build_reference(
    episode_path: str | Path,
    scene_path: str | Path = DEFAULT_SCENE,
    scene_anchor: np.ndarray = DEFAULT_SCENE_ANCHOR,
    smoothing: SmoothingConfig = SmoothingConfig(),
    ik_config: IKConfig = IKConfig(),
) -> ReferenceTrajectory:
    """Derive a smoothed gripper target and warm-started ARX IK trajectory."""
    episode = load_episode(episode_path)
    scene_path = Path(scene_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    scene_T_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    raw_targets = convert_episode_to_grippers(episode, scene_T_egodex)
    targets = smooth_gripper_trajectory(raw_targets, smoothing)
    ik = ARXDualArmIKSolver(model, ik_config).solve_episode(targets)

    quaternions_xyzw = Rotation.from_matrix(targets.rotation.reshape(-1, 3, 3)).as_quat()
    quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]].reshape(
        episode.frame_count, 2, 4
    )
    qvel_ref = np.gradient(ik.qpos, 1.0 / EGODEX_FPS, axis=0, edge_order=1)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fps": EGODEX_FPS,
        "source_episode": str(episode.path),
        "scene_path": str(scene_path),
        "scene_anchor": np.asarray(scene_anchor, dtype=np.float64).tolist(),
        "task": _json_value(episode.metadata.get("task", "")),
        "object": _json_value(episode.metadata.get("object", "")),
        "description": _json_value(episode.metadata.get("llm_description", "")),
        "joint_names": _joint_names(model),
        "actuator_names": _actuator_names(model),
        "smoothing": {
            "window": smoothing.window,
            "polyorder": smoothing.polyorder,
            "orientation_sigma": smoothing.orientation_sigma,
        },
        "ik": {
            "seed_iterations": ik_config.seed_iterations,
            "step_iterations": ik_config.step_iterations,
            "position_weight": ik_config.position_weight,
            "orientation_weight": ik_config.orientation_weight,
            "damping": ik_config.damping,
            "posture_weight": ik_config.posture_weight,
        },
    }
    return ReferenceTrajectory(
        qpos_ref=ik.qpos.astype(np.float32),
        qvel_ref=qvel_ref.astype(np.float32),
        ctrl_ref=ik.ctrl.astype(np.float32),
        tcp_pos_ref=targets.position.astype(np.float32),
        tcp_quat_wxyz_ref=quaternions_wxyz.astype(np.float32),
        gripper_width_ref=targets.width.astype(np.float32),
        target_valid=targets.valid,
        ik_position_error=ik.position_error.astype(np.float32),
        ik_orientation_error=ik.orientation_error.astype(np.float32),
        ik_converged=ik.converged,
        metadata=metadata,
    )


def save_reference(reference: ReferenceTrajectory, path: str | Path) -> Path:
    """Write a portable compressed NPZ reference without mutating source data."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos_ref=reference.qpos_ref,
        qvel_ref=reference.qvel_ref,
        ctrl_ref=reference.ctrl_ref,
        tcp_pos_ref=reference.tcp_pos_ref,
        tcp_quat_wxyz_ref=reference.tcp_quat_wxyz_ref,
        gripper_width_ref=reference.gripper_width_ref,
        target_valid=reference.target_valid,
        ik_position_error=reference.ik_position_error,
        ik_orientation_error=reference.ik_orientation_error,
        ik_converged=reference.ik_converged,
        metadata=np.array(json.dumps(reference.metadata, sort_keys=True)),
    )
    return path


def load_reference(path: str | Path) -> ReferenceTrajectory:
    """Load and validate a trajectory exported by :func:`save_reference`."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported reference schema {metadata.get('schema_version')!r} in {path}"
            )
        reference = ReferenceTrajectory(
            qpos_ref=data["qpos_ref"],
            qvel_ref=data["qvel_ref"],
            ctrl_ref=data["ctrl_ref"],
            tcp_pos_ref=data["tcp_pos_ref"],
            tcp_quat_wxyz_ref=data["tcp_quat_wxyz_ref"],
            gripper_width_ref=data["gripper_width_ref"],
            target_valid=data["target_valid"],
            ik_position_error=data["ik_position_error"],
            ik_orientation_error=data["ik_orientation_error"],
            ik_converged=data["ik_converged"],
            metadata=metadata,
        )
    _validate_reference(reference, path)
    return reference


def reference_summary(reference: ReferenceTrajectory) -> dict[str, float | int]:
    """Return manifest-friendly quality statistics for one reference."""
    valid = reference.target_valid
    return {
        "frames": reference.frame_count,
        "valid_target_fraction": float(valid.mean()),
        "ik_converged_fraction": float(reference.ik_converged[valid].mean()),
        "median_ik_position_error_m": float(np.median(reference.ik_position_error[valid])),
        "median_ik_orientation_error_rad": float(
            np.median(reference.ik_orientation_error[valid])
        ),
    }


def _validate_reference(reference: ReferenceTrajectory, path: Path) -> None:
    frames = reference.frame_count
    expected_shapes = {
        "qvel_ref": reference.qvel_ref.shape[0],
        "ctrl_ref": reference.ctrl_ref.shape[0],
        "tcp_pos_ref": reference.tcp_pos_ref.shape[0],
        "tcp_quat_wxyz_ref": reference.tcp_quat_wxyz_ref.shape[0],
        "gripper_width_ref": reference.gripper_width_ref.shape[0],
        "target_valid": reference.target_valid.shape[0],
        "ik_position_error": reference.ik_position_error.shape[0],
        "ik_orientation_error": reference.ik_orientation_error.shape[0],
        "ik_converged": reference.ik_converged.shape[0],
    }
    mismatched = {name: count for name, count in expected_shapes.items() if count != frames}
    if mismatched:
        raise ValueError(f"Inconsistent frame count in {path}: qpos={frames}, others={mismatched}")
    if reference.qpos_ref.ndim != 2 or reference.ctrl_ref.ndim != 2:
        raise ValueError(f"Reference qpos/ctrl must be rank 2: {path}")
    if reference.tcp_pos_ref.shape[1:] != (2, 3):
        raise ValueError(f"Expected TCP positions (T, 2, 3): {path}")
    if reference.tcp_quat_wxyz_ref.shape[1:] != (2, 4):
        raise ValueError(f"Expected TCP quaternions (T, 2, 4): {path}")
    if reference.gripper_width_ref.shape[1:] != (2,):
        raise ValueError(f"Expected gripper widths (T, 2): {path}")
    if not np.isfinite(reference.qpos_ref).all() or not np.isfinite(reference.ctrl_ref).all():
        raise ValueError(f"Reference contains non-finite robot commands: {path}")


def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        for joint_id in range(model.njnt)
    ]


def _actuator_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
        for actuator_id in range(model.nu)
    ]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    return value

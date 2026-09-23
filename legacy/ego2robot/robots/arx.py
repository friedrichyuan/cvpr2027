"""ARX dual-arm placement and IK in the ego camera frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egodex_arx_replay.defaults import DEFAULT_SCENE
from egodex_arx_replay.gripper import GripperTrajectory
from egodex_arx_replay.ik import ARXDualArmIKSolver, IKConfig

ARX_REACH = 0.855
# Camera (OpenCV: X right, Y down, Z forward) vs robot (X forward, Y left, Z up).
R_CAM_FROM_ROBOT = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
# Hands sit this far along robot +X / +Z from the base in a typical table grasp.
_WORKSPACE_IN_ROBOT = np.array([0.55, 0.0, 0.18], dtype=np.float64)


@dataclass
class BaseIkResult:
    base: dict
    qpos: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    converged: np.ndarray
    contacts: np.ndarray


def load_arx_model(scene_path: Path | None = None) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene_path or DEFAULT_SCENE))


def apply_base_pose(model: mujoco.MjModel, transform: np.ndarray) -> None:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        raise ValueError("Scene is missing body 'base_link'")
    model.body_pos[base_id] = transform[:3, 3]
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    model.body_quat[base_id] = xyzw[[3, 0, 1, 2]]


def se3_from_rotation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def seed_base_pose(targets: GripperTrajectory, reach: float = ARX_REACH) -> np.ndarray:
    """Place one dual-arm base so both hands sit in the ACone workspace.

    Same idea as ``egodex_arx_impaint`` keeping the robot at the table origin:
    put the workspace in front of the robot, then search small (dx, dy, yaw)
    offsets.  Here the 'table origin' is the camera-aligned seed.
    """
    valid = targets.valid.all(axis=1)
    if not valid.any():
        valid = targets.valid.any(axis=1)
    midpoint = 0.5 * (targets.position[valid, 0] + targets.position[valid, 1]).mean(axis=0)
    translation = midpoint - R_CAM_FROM_ROBOT @ (_WORKSPACE_IN_ROBOT * np.array([reach, 1.0, 1.0]))
    return se3_from_rotation(R_CAM_FROM_ROBOT, translation)


def apply_robot_offset(seed: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Offset in the robot frame: dx forward, dy left, dz up, yaw about +Z."""
    dx, dy, dz, yaw = np.asarray(offset, dtype=np.float64)
    rotation = seed[:3, :3] @ Rotation.from_euler("z", float(yaw)).as_matrix()
    translation = seed[:3, 3] + seed[:3, :3] @ np.array([dx, dy, dz], dtype=np.float64)
    return se3_from_rotation(rotation, translation)


def candidate_bases(
    targets: GripperTrajectory,
    reach: float = ARX_REACH,
    coarse: bool = True,
) -> list[np.ndarray]:
    """Grid around the camera-aligned seed, used by the optional cuRobo path."""
    seed = seed_base_pose(targets, reach)
    if coarse:
        lateral = (-0.15, 0.0, 0.15)
        forward = (-0.10, 0.0, 0.15)
        vertical = (-0.10, 0.0, 0.10)
        yaws = np.deg2rad([-20.0, 0.0, 20.0])
    else:
        lateral = (-0.30, -0.15, 0.0, 0.15, 0.30)
        forward = (-0.20, -0.05, 0.10, 0.25)
        vertical = (-0.20, -0.05, 0.10)
        yaws = np.deg2rad([-35.0, -15.0, 0.0, 15.0, 35.0])
    poses = []
    for dy in lateral:
        for dx in forward:
            for dz in vertical:
                for yaw in yaws:
                    poses.append(apply_robot_offset(seed, np.array([dx, dy, dz, yaw])))
    return poses


def search_base_and_solve(
    targets: GripperTrajectory,
    reach: float = ARX_REACH,
    coarse: bool = True,
    keyframes: int = 12,
) -> BaseIkResult:
    try:
        from ego2robot.actors.curobo_ik import solve_with_curobo

        return solve_with_curobo(targets, reach=reach, coarse=coarse)
    except ImportError:
        pass
    return _solve_with_mujoco(targets, reach, coarse, keyframes)


def _solve_with_mujoco(
    targets: GripperTrajectory,
    reach: float,
    coarse: bool,
    keyframes: int,
) -> BaseIkResult:
    keys = _select_keyframes(targets, keyframes)
    key_targets = GripperTrajectory(
        position=targets.position[keys],
        rotation=targets.rotation[keys],
        width=targets.width[keys],
        valid=targets.valid[keys],
    )
    seed = seed_base_pose(targets, reach)
    offset, cost, mean_pos, max_pos, mean_ori = _cem_search(seed, key_targets, coarse)
    pose = apply_robot_offset(seed, offset)
    model = load_arx_model()
    apply_base_pose(model, pose)
    ik = ARXDualArmIKSolver(model, IKConfig()).solve_episode(targets)
    contacts = _count_contacts(model, ik.qpos)
    xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return BaseIkResult(
        base={
            "translation": pose[:3, 3].tolist(),
            "quat_wxyz": [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])],
            "offset": {
                "dx": float(offset[0]),
                "dy": float(offset[1]),
                "dz": float(offset[2]),
                "yaw_deg": float(np.degrees(offset[3])),
            },
            "cost": cost,
            "keyframe_mean_pos_error_m": mean_pos,
            "keyframe_max_pos_error_m": max_pos,
            "keyframe_mean_ori_error_rad": mean_ori,
            "backend": "mujoco",
            "reach": reach,
        },
        qpos=ik.qpos,
        position_error=ik.position_error,
        orientation_error=ik.orientation_error,
        converged=ik.converged,
        contacts=contacts,
    )


def _cem_search(
    seed: np.ndarray,
    key_targets: GripperTrajectory,
    coarse: bool,
) -> tuple[np.ndarray, float, float, float, float]:
    """CEM on robot-frame (dx, dy, dz, yaw), scored by IK error like impaint."""
    samples = 16 if coarse else 32
    iterations = 4
    elite_fraction = 0.25
    bounds = np.array(
        [
            [-0.30, 0.30],
            [-0.30, 0.30],
            [-0.20, 0.20],
            [np.deg2rad(-35.0), np.deg2rad(35.0)],
        ]
    )
    mean = np.zeros(4, dtype=np.float64)
    std = np.array([0.12, 0.12, 0.08, np.deg2rad(15.0)], dtype=np.float64)
    rng = np.random.default_rng(0)
    model = load_arx_model()
    ik_cfg = IKConfig(step_iterations=40, independent_random_restarts=4, independent_max_nfev=800)
    best_offset = mean.copy()
    best_cost = np.inf
    best_stats = (np.inf, np.inf, np.inf)

    for iteration in range(iterations):
        offsets = rng.normal(mean[None, :], std[None, :], size=(samples, 4))
        offsets[0] = mean
        offsets = np.clip(offsets, bounds[:, 0], bounds[:, 1])
        scored = [_score_offset(model, seed, key_targets, ik_cfg, offset) for offset in offsets]
        scored.sort(key=lambda item: item[1])
        if scored[0][1] < best_cost:
            best_offset, best_cost = scored[0][0], scored[0][1]
            best_stats = scored[0][2:]
        elite_count = max(2, int(round(samples * elite_fraction)))
        elites = np.stack([item[0] for item in scored[:elite_count]])
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), np.array([0.01, 0.01, 0.01, np.deg2rad(1.0)]))
        mean = np.clip(mean, bounds[:, 0], bounds[:, 1])
        print(
            f"base cem {iteration + 1}/{iterations}: "
            f"cost={scored[0][1]:.4f}, "
            f"mean_pos={scored[0][2] * 1000.0:.1f}mm, "
            f"max_pos={scored[0][3] * 1000.0:.1f}mm, "
            f"offset=({scored[0][0][0]:+.3f}, {scored[0][0][1]:+.3f}, "
            f"{scored[0][0][2]:+.3f}, {np.degrees(scored[0][0][3]):+.1f}deg)"
        )

    return best_offset, float(best_cost), float(best_stats[0]), float(best_stats[1]), float(best_stats[2])


def _score_offset(
    model: mujoco.MjModel,
    seed: np.ndarray,
    targets: GripperTrajectory,
    ik_cfg: IKConfig,
    offset: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    pose = apply_robot_offset(seed, offset)
    apply_base_pose(model, pose)
    ik = ARXDualArmIKSolver(model, ik_cfg).solve_episode(targets)
    valid = targets.valid
    pos_errors = ik.position_error[valid]
    ori_errors = ik.orientation_error[valid]
    mean_pos = float(np.nanmean(pos_errors))
    max_pos = float(np.nanmax(pos_errors))
    mean_ori = float(np.nanmean(ori_errors))
    contacts = int(_count_contacts(model, ik.qpos).sum())
    regularization = 0.02 * float((offset[0] / 0.30) ** 2 + (offset[1] / 0.30) ** 2 + (offset[2] / 0.20) ** 2)
    regularization += 0.01 * float((offset[3] / np.deg2rad(35.0)) ** 2)
    behind_camera = 0.05 * max(0.0, -float(pose[2, 3]))
    cost = mean_pos + 0.5 * max_pos + 0.02 * mean_ori + 0.002 * contacts + regularization + behind_camera
    return offset.copy(), cost, mean_pos, max_pos, mean_ori


def _select_keyframes(targets: GripperTrajectory, count: int) -> np.ndarray:
    frames = targets.position.shape[0]
    if count >= frames:
        return np.arange(frames, dtype=np.int32)
    uniform = np.linspace(0, frames - 1, count, dtype=np.int32)
    speed = np.linalg.norm(np.diff(targets.position, axis=0), axis=-1).mean(axis=-1)
    high_motion = np.argsort(speed)[-max(1, count // 4) :] + 1
    selected = np.unique(np.concatenate(([0, frames - 1], uniform, high_motion))).astype(np.int32)
    if len(selected) > count:
        selected = selected[np.linspace(0, len(selected) - 1, count, dtype=np.int32)]
    return selected


def _count_contacts(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    counts = np.zeros(qpos.shape[0], dtype=np.int32)
    for index, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        counts[index] = int(data.ncon)
    return counts

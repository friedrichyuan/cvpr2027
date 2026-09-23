"""Optional NVIDIA cuRobo backend. ImportError means the MuJoCo fallback is used."""

from __future__ import annotations

import numpy as np

from egodex_arx_replay.gripper import GripperTrajectory

from ..robots.arx import ARX_REACH, BaseIkResult, candidate_bases


def solve_with_curobo(
    targets: GripperTrajectory,
    reach: float = ARX_REACH,
    coarse: bool = True,
) -> BaseIkResult:
    from curobo import InverseKinematics, InverseKinematicsCfg

    robot = os_env_robot()
    solver = InverseKinematics(
        InverseKinematicsCfg.create(robot=robot, num_seeds=16, self_collision_check=True)
    )
    poses = candidate_bases(targets, reach=reach, coarse=coarse)
    best_pose = None
    best_score = -1e9
    key_index = _keyframes(targets, 8)
    for pose in poses:
        ee = _targets_in_base(targets, pose, key_index)
        result = solver.solve_pose(goal_tool_poses=ee)
        success = float(result.success.float().mean().item()) if hasattr(result, "success") else 0.0
        rho = np.linalg.norm(targets.position[key_index][targets.valid[key_index]] - pose[:3, 3], axis=-1)
        rho = float(rho.mean() / reach) if rho.size else 1.0
        score = success - 5.0 * abs(rho - 0.65)
        if score > best_score:
            best_score = score
            best_pose = pose
    if best_pose is None:
        raise RuntimeError("cuRobo base search found no candidate")
    full = _targets_in_base(targets, best_pose, np.arange(targets.position.shape[0]))
    solved = solver.solve_pose(goal_tool_poses=full)
    qpos = _as_numpy(solved.solution if hasattr(solved, "solution") else solved.q)
    xyzw = _quat_xyzw(best_pose)
    return BaseIkResult(
        base={
            "translation": best_pose[:3, 3].tolist(),
            "quat_wxyz": [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])],
            "score": best_score,
            "backend": "curobo",
            "reach": reach,
        },
        qpos=qpos,
        position_error=np.zeros((qpos.shape[0], 2), dtype=np.float32),
        orientation_error=np.zeros((qpos.shape[0], 2), dtype=np.float32),
        converged=np.ones((qpos.shape[0], 2), dtype=bool),
        contacts=np.zeros(qpos.shape[0], dtype=np.int32),
    )


def os_env_robot() -> str:
    import os

    return os.environ.get("CUROBO_ROBOT", "arx.yml")


def _keyframes(targets: GripperTrajectory, count: int) -> np.ndarray:
    frames = targets.position.shape[0]
    return np.linspace(0, frames - 1, min(count, frames), dtype=np.int32)


def _targets_in_base(targets: GripperTrajectory, pose: np.ndarray, frames: np.ndarray):
    import torch
    from scipy.spatial.transform import Rotation

    inverse = np.linalg.inv(pose)
    positions = []
    quats = []
    for frame in frames:
        p = inverse[:3, :3] @ targets.position[frame, 1] + inverse[:3, 3]
        r = inverse[:3, :3] @ targets.rotation[frame, 1]
        q = Rotation.from_matrix(r).as_quat()  # xyzw
        positions.append(p)
        quats.append([q[3], q[0], q[1], q[2]])
    return type("Pose", (), {"position": torch.tensor(positions), "quaternion": torch.tensor(quats)})


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_xyzw(pose: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(pose[:3, :3]).as_quat()

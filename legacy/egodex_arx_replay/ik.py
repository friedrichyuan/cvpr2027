"""Warm-started dual-arm MuJoCo Jacobian IK for ARX5 gripper references."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .gripper import GripperTrajectory


@dataclass(frozen=True)
class IKConfig:
    """Shared, task-independent solver settings for all EgoDex episodes."""

    warm_start: bool = False
    temporal_after_first: bool = True
    seed_iterations: int = 80
    step_iterations: int = 30
    position_weight: float = 1.0
    orientation_weight: float = 0.10
    damping: float = 0.02
    posture_weight: float = 0.0
    max_joint_step: float = 0.15
    position_tolerance: float = 0.004
    orientation_tolerance: float = 0.12
    independent_random_restarts: int = 8
    independent_max_nfev: int = 2000
    independent_continuity_weight: float = 0.05


@dataclass(frozen=True)
class IKTrajectory:
    """ARX joint targets and per-frame diagnostics after inverse kinematics."""

    qpos: np.ndarray
    ctrl: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    converged: np.ndarray


class ARXDualArmIKSolver:
    """Solve both ARX arms together while retaining the prior frame as posture."""

    _ARM_JOINTS = (
        ("left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6"),
        ("right_joint11", "right_joint12", "right_joint13", "right_joint14", "right_joint15", "right_joint16"),
    )
    _GRIPPER_JOINTS = (("left_joint7", "left_joint8"), ("right_joint17", "right_joint18"))
    _TCP_SITES = ("left_tcp", "right_tcp")

    def __init__(self, model: mujoco.MjModel, config: IKConfig = IKConfig()) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config
        self.arm_joint_ids = tuple(tuple(self._joint_id(name) for name in names) for names in self._ARM_JOINTS)
        self.gripper_joint_ids = tuple(tuple(self._joint_id(name) for name in names) for names in self._GRIPPER_JOINTS)
        self.arm_qpos_indices_by_side = tuple(
            np.array([int(model.jnt_qposadr[joint_id]) for joint_id in side], dtype=int)
            for side in self.arm_joint_ids
        )
        self.arm_qpos_indices = np.array(
            [int(model.jnt_qposadr[joint_id]) for side in self.arm_joint_ids for joint_id in side], dtype=int
        )
        self.arm_dof_indices = np.array(
            [int(model.jnt_dofadr[joint_id]) for side in self.arm_joint_ids for joint_id in side], dtype=int
        )
        self.tcp_site_ids = tuple(self._site_id(name) for name in self._TCP_SITES)
        arm_joint_ids = np.array(self.arm_joint_ids).reshape(-1)
        self._arm_lower = model.jnt_range[arm_joint_ids, 0]
        self._arm_upper = model.jnt_range[arm_joint_ids, 1]
        self._arm_lower_by_side = tuple(model.jnt_range[np.array(side), 0] for side in self.arm_joint_ids)
        self._arm_upper_by_side = tuple(model.jnt_range[np.array(side), 1] for side in self.arm_joint_ids)
        self._gripper_min_gap = self._measure_minimum_gripper_gaps()
        self._ctrl_qpos_indices = self._actuator_qpos_indices()

    def solve_episode(self, targets: GripperTrajectory) -> IKTrajectory:
        """Solve a smoothed sequence, warm-starting every frame from the prior."""
        if not self.config.warm_start:
            if self.config.temporal_after_first:
                return self._solve_episode_seeded_warm_start(targets)
            return self._solve_episode_independent(targets)
        frame_count = targets.position.shape[0]
        qpos_out = np.zeros((frame_count, self.model.nq), dtype=np.float64)
        ctrl_out = np.zeros((frame_count, self.model.nu), dtype=np.float64)
        position_error = np.full((frame_count, 2), np.nan, dtype=np.float64)
        orientation_error = np.full((frame_count, 2), np.nan, dtype=np.float64)
        converged = np.zeros((frame_count, 2), dtype=bool)

        initial = self.model.key_qpos[0].copy() if self.model.nkey else np.zeros(self.model.nq)
        previous = initial.copy()
        for frame in range(frame_count):
            qpos = previous.copy()
            self._write_grippers(qpos, targets.width[frame], targets.valid[frame])
            iterations = self.config.seed_iterations if frame == 0 else self.config.step_iterations
            qpos = self._solve_frame(qpos, previous, targets, frame, iterations)
            self._write_grippers(qpos, targets.width[frame], targets.valid[frame])

            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            frame_position_error, frame_orientation_error = self._task_errors(targets, frame)
            position_error[frame] = frame_position_error
            orientation_error[frame] = frame_orientation_error
            valid = targets.valid[frame]
            converged[frame, valid] = (
                (frame_position_error[valid] <= self.config.position_tolerance)
                & (frame_orientation_error[valid] <= self.config.orientation_tolerance)
            )
            qpos_out[frame] = qpos
            ctrl_out[frame] = qpos[self._ctrl_qpos_indices]
            previous = qpos

        return IKTrajectory(qpos_out, ctrl_out, position_error, orientation_error, converged)

    def _solve_episode_seeded_warm_start(self, targets: GripperTrajectory) -> IKTrajectory:
        """Solve frame zero accurately, then warm-start subsequent frames from it."""
        frame_count = targets.position.shape[0]
        qpos_out = np.zeros((frame_count, self.model.nq), dtype=np.float64)
        ctrl_out = np.zeros((frame_count, self.model.nu), dtype=np.float64)
        position_error = np.full((frame_count, 2), np.nan, dtype=np.float64)
        orientation_error = np.full((frame_count, 2), np.nan, dtype=np.float64)
        converged = np.zeros((frame_count, 2), dtype=bool)
        initial = self.model.key_qpos[0].copy() if self.model.nkey else np.zeros(self.model.nq)
        previous = initial.copy()

        for frame in range(frame_count):
            if frame == 0:
                qpos = initial.copy()
                self._write_grippers(qpos, targets.width[frame], targets.valid[frame])
                for side in range(2):
                    if targets.valid[frame, side]:
                        self._solve_side_independent(qpos, targets, frame, side, previous)
            else:
                qpos = previous.copy()
                self._write_grippers(qpos, targets.width[frame], targets.valid[frame])
                qpos = self._solve_frame(qpos, previous, targets, frame, self.config.step_iterations)
            self._write_grippers(qpos, targets.width[frame], targets.valid[frame])

            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            frame_position_error, frame_orientation_error = self._task_errors(targets, frame)
            position_error[frame] = frame_position_error
            orientation_error[frame] = frame_orientation_error
            valid = targets.valid[frame]
            converged[frame, valid] = (
                (frame_position_error[valid] <= self.config.position_tolerance)
                & (frame_orientation_error[valid] <= self.config.orientation_tolerance)
            )
            qpos_out[frame] = qpos
            ctrl_out[frame] = qpos[self._ctrl_qpos_indices]
            previous = qpos

        return IKTrajectory(qpos_out, ctrl_out, position_error, orientation_error, converged)

    def _solve_episode_independent(self, targets: GripperTrajectory) -> IKTrajectory:
        """Solve each frame directly from fixed seeds instead of previous-frame warm-starts."""
        frame_count = targets.position.shape[0]
        qpos_out = np.zeros((frame_count, self.model.nq), dtype=np.float64)
        ctrl_out = np.zeros((frame_count, self.model.nu), dtype=np.float64)
        position_error = np.full((frame_count, 2), np.nan, dtype=np.float64)
        orientation_error = np.full((frame_count, 2), np.nan, dtype=np.float64)
        converged = np.zeros((frame_count, 2), dtype=bool)
        initial = self.model.key_qpos[0].copy() if self.model.nkey else np.zeros(self.model.nq)
        previous_qpos = initial.copy()

        for frame in range(frame_count):
            qpos = initial.copy()
            self._write_grippers(qpos, targets.width[frame], targets.valid[frame])
            for side in range(2):
                if targets.valid[frame, side]:
                    self._solve_side_independent(qpos, targets, frame, side, previous_qpos)
            self._write_grippers(qpos, targets.width[frame], targets.valid[frame])

            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            frame_position_error, frame_orientation_error = self._task_errors(targets, frame)
            position_error[frame] = frame_position_error
            orientation_error[frame] = frame_orientation_error
            valid = targets.valid[frame]
            converged[frame, valid] = (
                (frame_position_error[valid] <= self.config.position_tolerance)
                & (frame_orientation_error[valid] <= self.config.orientation_tolerance)
            )
            qpos_out[frame] = qpos
            ctrl_out[frame] = qpos[self._ctrl_qpos_indices]
            previous_qpos = qpos.copy()

        return IKTrajectory(qpos_out, ctrl_out, position_error, orientation_error, converged)

    def _solve_side_independent(
        self,
        qpos: np.ndarray,
        targets: GripperTrajectory,
        frame: int,
        side: int,
        continuity_qpos: np.ndarray,
    ) -> None:
        qpos_indices = self.arm_qpos_indices_by_side[side]
        lower = self._arm_lower_by_side[side]
        upper = self._arm_upper_by_side[side]
        site_id = self.tcp_site_ids[side]
        target_position = targets.position[frame, side]
        target_rotation = targets.rotation[frame, side]
        base_qpos = qpos.copy()

        def residual(arm_qpos: np.ndarray) -> np.ndarray:
            probe_qpos = base_qpos.copy()
            probe_qpos[qpos_indices] = arm_qpos
            self.data.qpos[:] = probe_qpos
            mujoco.mj_forward(self.model, self.data)
            position_residual = self.data.site_xpos[site_id] - target_position
            current_rotation = self.data.site_xmat[site_id].reshape(3, 3)
            rotation_residual = Rotation.from_matrix(
                target_rotation @ current_rotation.T
            ).as_rotvec()
            return np.concatenate((
                self.config.position_weight * position_residual,
                self.config.orientation_weight * rotation_residual,
            ))

        best: tuple[tuple[int, float, float], np.ndarray] | None = None
        seeds = [np.clip(continuity_qpos[qpos_indices], lower, upper)]
        seeds.extend(self._independent_seeds(qpos[qpos_indices], lower, upper, frame, side))
        for seed in seeds:
            result = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
                max_nfev=self.config.independent_max_nfev,
            )
            position_norm = float(np.linalg.norm(residual(result.x)[:3]))
            continuity_norm = float(np.linalg.norm(result.x - continuity_qpos[qpos_indices]))
            # First prefer any candidate inside the position tolerance, then
            # choose the most joint-continuous branch.  This avoids framewise
            # IK branch flicker without using the previous frame as an initial
            # guess for the nonlinear solve.
            outside_tolerance = int(position_norm > self.config.position_tolerance)
            position_score = (
                position_norm + self.config.independent_continuity_weight * continuity_norm
                if outside_tolerance
                else 0.0
            )
            score = (outside_tolerance, position_score, continuity_norm)
            if best is None or score < best[0]:
                best = (score, result.x.copy())

        assert best is not None
        qpos[qpos_indices] = best[1]

    def _independent_seeds(
        self,
        nominal: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        frame: int,
        side: int,
    ) -> list[np.ndarray]:
        seeds = [
            np.clip(nominal, lower, upper),
            0.5 * (lower + upper),
            0.25 * lower + 0.75 * upper,
            0.75 * lower + 0.25 * upper,
        ]
        rng = np.random.default_rng(7919 + 31 * frame + side)
        for _ in range(self.config.independent_random_restarts):
            seeds.append(rng.uniform(lower, upper))
        return seeds

    def _solve_frame(self, qpos, previous, targets, frame, iterations) -> np.ndarray:
        for _ in range(iterations):
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            jacobian_blocks: list[np.ndarray] = []
            residual_blocks: list[np.ndarray] = []
            for side, site_id in enumerate(self.tcp_site_ids):
                if not targets.valid[frame, side]:
                    continue
                jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
                jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
                current_rotation = self.data.site_xmat[site_id].reshape(3, 3)
                position_residual = targets.position[frame, side] - self.data.site_xpos[site_id]
                rotation_residual = Rotation.from_matrix(
                    targets.rotation[frame, side] @ current_rotation.T
                ).as_rotvec()
                jacobian_blocks.append(np.vstack((
                    self.config.position_weight * jac_pos[:, self.arm_dof_indices],
                    self.config.orientation_weight * jac_rot[:, self.arm_dof_indices],
                )))
                residual_blocks.append(np.concatenate((
                    self.config.position_weight * position_residual,
                    self.config.orientation_weight * rotation_residual,
                )))
            if not jacobian_blocks:
                break

            jacobian = np.vstack(jacobian_blocks)
            residual = np.concatenate(residual_blocks)
            posture_delta = qpos[self.arm_qpos_indices] - previous[self.arm_qpos_indices]
            hessian = jacobian.T @ jacobian
            hessian += (self.config.damping**2 + self.config.posture_weight) * np.eye(len(self.arm_dof_indices))
            gradient = jacobian.T @ residual - self.config.posture_weight * posture_delta
            delta = np.linalg.solve(hessian, gradient)
            delta = np.clip(delta, -self.config.max_joint_step, self.config.max_joint_step)
            qpos[self.arm_qpos_indices] = np.clip(
                qpos[self.arm_qpos_indices] + delta,
                self._arm_lower,
                self._arm_upper,
            )

            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            position_error, orientation_error = self._task_errors(targets, frame)
            valid = targets.valid[frame]
            if valid.any() and (
                np.max(position_error[valid]) <= self.config.position_tolerance
                and np.max(orientation_error[valid]) <= self.config.orientation_tolerance
            ):
                break
        return qpos

    def _task_errors(self, targets: GripperTrajectory, frame: int) -> tuple[np.ndarray, np.ndarray]:
        position_error = np.full(2, np.nan, dtype=np.float64)
        orientation_error = np.full(2, np.nan, dtype=np.float64)
        for side, site_id in enumerate(self.tcp_site_ids):
            if not targets.valid[frame, side]:
                continue
            position_error[side] = np.linalg.norm(targets.position[frame, side] - self.data.site_xpos[site_id])
            current_rotation = self.data.site_xmat[site_id].reshape(3, 3)
            orientation_error[side] = np.linalg.norm(
                Rotation.from_matrix(targets.rotation[frame, side] @ current_rotation.T).as_rotvec()
            )
        return position_error, orientation_error

    def _write_grippers(self, qpos: np.ndarray, widths: np.ndarray, valid: np.ndarray) -> None:
        """Map target jaw separation to symmetric ARX slide joints, with limits."""
        for side, joint_ids in enumerate(self.gripper_joint_ids):
            if not valid[side]:
                continue
            lower = self.model.jnt_range[np.array(joint_ids), 0]
            upper = self.model.jnt_range[np.array(joint_ids), 1]
            displacement = 0.5 * (widths[side] - self._gripper_min_gap[side])
            symmetric_value = np.clip(displacement, max(lower), min(upper))
            for joint_id in joint_ids:
                qpos[self.model.jnt_qposadr[joint_id]] = symmetric_value

    def _measure_minimum_gripper_gaps(self) -> np.ndarray:
        probe = mujoco.MjData(self.model)
        gaps = np.zeros(2, dtype=np.float64)
        for side, joint_ids in enumerate(self.gripper_joint_ids):
            probe.qpos[:] = 0.0
            body_ids = []
            for joint_id in joint_ids:
                probe.qpos[self.model.jnt_qposadr[joint_id]] = self.model.jnt_range[joint_id, 0]
                body_ids.append(int(self.model.jnt_bodyid[joint_id]))
            mujoco.mj_forward(self.model, probe)
            gaps[side] = np.linalg.norm(probe.xpos[body_ids[0]] - probe.xpos[body_ids[1]])
        return gaps

    def _actuator_qpos_indices(self) -> np.ndarray:
        indices = []
        for actuator in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator, 0])
            if joint_id < 0:
                raise ValueError("ARX replay expects every actuator to target one joint")
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(indices, dtype=int)

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"ARX scene is missing required joint '{name}'")
        return joint_id

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"ARX scene is missing required TCP site '{name}'")
        return site_id

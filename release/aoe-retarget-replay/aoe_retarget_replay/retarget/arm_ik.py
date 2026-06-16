"""G1 7-DOF arm IK solver using mink (MuJoCo-based IK)."""

import logging
from pathlib import Path

import mink
import mujoco
import numpy as np

from aoe_retarget_replay.constants import (
    CONF_THRESHOLD,
    G1_LEFT_WRIST_TARGET_BODY,
    G1_RIGHT_WRIST_TARGET_BODY,
    G1_STANDING_HEIGHT,
    IK_DAMPING,
    IK_ORIENTATION_COST,
    IK_POS_TOLERANCE,
    IK_POSITION_COST,
    IK_SEED_ITERS,
    IK_SMOOTHNESS_COST,
    IK_SOLVER,
    IK_STEP_ITERS,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)

logger = logging.getLogger(__name__)

_ARM_JOINT_NAMES_FOR_IK = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES


class G1ArmIKSolver:
    """Solves G1 7-DOF arm IK for both arms using mink."""

    def __init__(self, mjcf_path: str | Path | None = None, dt: float = 1.0 / 30):
        if mjcf_path is None:
            from aoe_retarget_replay.constants.g1_dex3 import G1_MJCF_PATH
            mjcf_path = G1_MJCF_PATH
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.dt = dt

        self.data.qpos[:] = 0.0
        self.data.qpos[2] = G1_STANDING_HEIGHT
        mujoco.mj_forward(self.model, self.data)

        self.config = mink.Configuration(self.model)

        self.left_wrist_task = mink.FrameTask(
            frame_name=G1_LEFT_WRIST_TARGET_BODY,
            frame_type="body",
            position_cost=np.array([IK_POSITION_COST]),
            orientation_cost=np.array([IK_ORIENTATION_COST]),
        )
        self.right_wrist_task = mink.FrameTask(
            frame_name=G1_RIGHT_WRIST_TARGET_BODY,
            frame_type="body",
            position_cost=np.array([IK_POSITION_COST]),
            orientation_cost=np.array([IK_ORIENTATION_COST]),
        )
        self.posture_task = mink.PostureTask(
            model=self.model,
            cost=np.array([IK_SMOOTHNESS_COST]),
        )

        arm_dofs = set()
        for jname in _ARM_JOINT_NAMES_FOR_IK:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            arm_dofs.add(int(self.model.jnt_dofadr[jid]))
        frozen_dofs = sorted(set(range(self.model.nv)) - arm_dofs)
        self.freeze_task = mink.DofFreezingTask(
            model=self.model,
            dof_indices=frozen_dofs,
        )

        left_arm_names = _ARM_JOINT_NAMES_FOR_IK[:7]
        right_arm_names = _ARM_JOINT_NAMES_FOR_IK[7:]
        self._left_arm_qpos_idx = np.array(
            [int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]) for n in left_arm_names], dtype=int,
        )
        self._right_arm_qpos_idx = np.array(
            [int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]) for n in right_arm_names], dtype=int,
        )
        self._left_arm_jnt_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
             for n in left_arm_names], dtype=int,
        )
        self._right_arm_jnt_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
             for n in right_arm_names], dtype=int,
        )

        self._standing_qpos = self.data.qpos.copy()

    def set_standing_pose(self) -> None:
        """Reset the robot to standing pose with arms at sides."""
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = G1_STANDING_HEIGHT
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.config.update(self.data.qpos)

    def solve_episode(
        self,
        left_wrist_targets: np.ndarray,
        right_wrist_targets: np.ndarray,
        left_wrist_conf: np.ndarray | None = None,
        right_wrist_conf: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve IK for an entire episode.

        Returns:
            (T, 14) array of arm joint angles: [left_arm(7), right_arm(7)].
        """
        T = left_wrist_targets.shape[0]
        if left_wrist_conf is None:
            left_wrist_conf = np.ones(T, dtype=np.float32)
        if right_wrist_conf is None:
            right_wrist_conf = np.ones(T, dtype=np.float32)

        arm_qpos = np.zeros((T, 14), dtype=np.float32)

        self.set_standing_pose()
        prev_qpos = self.data.qpos.copy()

        for t in range(T):
            self.posture_task.set_target(prev_qpos)

            if left_wrist_conf[t] >= CONF_THRESHOLD:
                left_target = mink.SE3.from_matrix(left_wrist_targets[t])
                self.left_wrist_task.set_target(left_target)

            if right_wrist_conf[t] >= CONF_THRESHOLD:
                right_target = mink.SE3.from_matrix(right_wrist_targets[t])
                self.right_wrist_task.set_target(right_target)

            tasks = [self.left_wrist_task, self.right_wrist_task, self.posture_task]
            max_iters = IK_SEED_ITERS if t == 0 else IK_STEP_ITERS

            for _ in range(max_iters):
                vel = mink.solve_ik(
                    self.config,
                    tasks,
                    self.dt,
                    solver=IK_SOLVER,
                    damping=IK_DAMPING,
                    constraints=[self.freeze_task],
                )
                self.config.integrate_inplace(vel, self.dt)

                left_err = self.left_wrist_task.compute_error(self.config)
                right_err = self.right_wrist_task.compute_error(self.config)
                max_pos_err = max(
                    np.linalg.norm(left_err[:3]),
                    np.linalg.norm(right_err[:3]),
                )
                if max_pos_err < IK_POS_TOLERANCE:
                    break

            qpos = self.config.q.copy()
            arm_qpos[t, :7] = qpos[self._left_arm_qpos_idx]
            arm_qpos[t, 7:] = qpos[self._right_arm_qpos_idx]

            arm_qpos[t, :7] = np.clip(
                arm_qpos[t, :7],
                self.model.jnt_range[self._left_arm_jnt_ids, 0],
                self.model.jnt_range[self._left_arm_jnt_ids, 1],
            )
            arm_qpos[t, 7:] = np.clip(
                arm_qpos[t, 7:],
                self.model.jnt_range[self._right_arm_jnt_ids, 0],
                self.model.jnt_range[self._right_arm_jnt_ids, 1],
            )

            prev_qpos = self.config.q.copy()

        return arm_qpos

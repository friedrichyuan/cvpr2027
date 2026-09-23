"""Dual-arm ARX IK with NVIDIA cuRobo, matching the existing MuJoCo IK interface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egodex_arx_replay.defaults import EGODEX_FPS, PROJECT_ROOT
from egodex_arx_replay.gripper import GripperTrajectory
from egodex_arx_replay.ik import ARXDualArmIKSolver, IKConfig, IKTrajectory

DEFAULT_CUROBO_URDF = PROJECT_ROOT / "assets" / "curobo_arx" / "arx_acone_kin.urdf"
DEFAULT_CUROBO_YAML = PROJECT_ROOT / "assets" / "curobo_arx" / "arx_acone.yml"
TCP_FRAMES = ("left_tcp", "right_tcp")
ARM_JOINTS = (
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "right_joint11",
    "right_joint12",
    "right_joint13",
    "right_joint14",
    "right_joint15",
    "right_joint16",
)
CuroboMode = Literal["seq", "batch", "trajopt", "mpc", "joint"]


@dataclass(frozen=True)
class JointOptConfig:
    """Weights and solver settings for joint base + full-trajectory IK."""

    lbfgs_steps: int = 8
    lbfgs_max_iter: int = 25
    history_size: int = 32
    adam_steps: int = 150
    base_inits: int = 8
    position_weight: float = 20.0
    orientation_weight: float = 5.0
    continuity_weight: float = 0.05
    velocity_weight: float = 2.0e-4
    jerk_weight: float = 0.20
    joint_limit_weight: float = 5.0
    base_reg_weight: float = 0.02
    base_dx_bound: float = 0.30
    base_dy_bound: float = 0.30
    base_yaw_bound: float = float(np.deg2rad(35.0))
    seed: int = 0


@dataclass(frozen=True)
class CuroboIKConfig:
    """Solver settings for the cuRobo dual-arm backend."""

    mode: CuroboMode = "seq"
    num_seeds: int = 32
    batch_size: int = 128
    position_tolerance: float = 0.004
    orientation_tolerance: float = 0.12
    orientation_weight: float = 0.10
    optimization_dt: float = 1.0 / EGODEX_FPS
    steps_per_target: int = 2
    trajopt_seeds: int = 4
    joint: JointOptConfig = JointOptConfig()
    urdf_path: Path = DEFAULT_CUROBO_URDF
    yaml_path: Path = DEFAULT_CUROBO_YAML


@dataclass
class CuroboSolveStats:
    setup_s: float
    solve_s: float
    warmup_s: float = 0.0
    base_offset: np.ndarray | None = None
    joint_losses: dict[str, float] | None = None


class ARXDualArmCuroboIKSolver:
    """Solve both ARX TCP sites with cuRobo, then evaluate in the MuJoCo scene."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config: CuroboIKConfig = CuroboIKConfig(),
        ik_config: IKConfig = IKConfig(),
    ) -> None:
        self.model = model
        self.config = config
        self.ik_config = ik_config
        self.evaluator = ARXDualArmIKSolver(model, ik_config)
        self.stats = CuroboSolveStats(setup_s=0.0, solve_s=0.0)
        self._robot_yaml = _materialize_robot_yaml(config.yaml_path, config.urdf_path)
        self._curobo = _import_curobo()

    def solve_episode(self, targets: GripperTrajectory) -> IKTrajectory:
        if self.config.mode == "batch":
            arm_qpos, solve_s, warmup_s = self._solve_batch(targets)
        elif self.config.mode == "trajopt":
            arm_qpos, solve_s, warmup_s = self._solve_trajopt(targets)
        elif self.config.mode == "mpc":
            arm_qpos, solve_s, warmup_s = self._solve_mpc(targets)
        elif self.config.mode == "joint":
            from .curobo_joint import solve_joint_episode

            arm_qpos, solve_s, warmup_s = solve_joint_episode(self, targets)
        else:
            arm_qpos, solve_s, warmup_s = self._solve_sequence(targets)
        self.stats.solve_s = solve_s
        self.stats.warmup_s = warmup_s
        return self._to_mujoco_trajectory(targets, arm_qpos)

    def _solve_sequence(self, targets: GripperTrajectory) -> tuple[np.ndarray, float, float]:
        """Warm-start cuRobo IK from the previous frame, without a velocity cap."""
        curobo = self._curobo
        setup_t0 = time.perf_counter()
        ik = _make_batch_ik_solver(
            curobo,
            str(self._robot_yaml),
            CuroboIKConfig(
                mode="seq",
                num_seeds=self.config.num_seeds,
                batch_size=1,
                position_tolerance=self.config.position_tolerance,
                orientation_tolerance=self.config.orientation_tolerance,
                orientation_weight=self.config.orientation_weight,
                urdf_path=self.config.urdf_path,
                yaml_path=self.config.yaml_path,
            ),
            use_cuda_graph=True,
        )
        self.stats.setup_s = time.perf_counter() - setup_t0

        pose_dict = _batch_pose_dict(curobo, targets)
        frame_count = targets.position.shape[0]
        joint_names = list(getattr(ik, "joint_names", ARM_JOINTS))
        previous = np.zeros(len(joint_names), dtype=np.float32)
        warmup_goal = _goal_from_poses(
            curobo,
            {name: _pad_pose(curobo, pose, 0, 1, 1) for name, pose in pose_dict.items()},
        )
        warmup_state, warmup_seed = _warm_start(
            curobo, previous, joint_names, self.config.num_seeds, noise_std=0.8
        )
        warmup_t0 = time.perf_counter()
        ik.solve_pose(
            goal_tool_poses=warmup_goal,
            seed_config=warmup_seed,
            current_state=warmup_state,
        )
        if hasattr(curobo.torch, "cuda") and curobo.torch.cuda.is_available():
            curobo.torch.cuda.synchronize()
        warmup_s = time.perf_counter() - warmup_t0

        solutions: list[np.ndarray] = []
        solve_t0 = time.perf_counter()
        for frame in range(frame_count):
            goal = _goal_from_poses(
                curobo,
                {name: _pad_pose(curobo, pose, frame, frame + 1, 1) for name, pose in pose_dict.items()},
            )
            current_state, seed_config = _warm_start(
                curobo,
                previous,
                joint_names,
                self.config.num_seeds,
                noise_std=0.8 if frame == 0 else 0.05,
            )
            result = ik.solve_pose(
                goal_tool_poses=goal,
                seed_config=seed_config,
                current_state=current_state,
            )
            joint_names = _result_joint_names(ik, result)
            qpos = _raw_joint_positions(result)[0]
            solutions.append(qpos)
            previous = qpos
        if hasattr(curobo.torch, "cuda") and curobo.torch.cuda.is_available():
            curobo.torch.cuda.synchronize()
        solve_s = time.perf_counter() - solve_t0
        return _reorder_joints(np.stack(solutions, axis=0), joint_names or list(ARM_JOINTS)), solve_s, warmup_s

    def _solve_batch(self, targets: GripperTrajectory) -> tuple[np.ndarray, float, float]:
        curobo = self._curobo
        setup_t0 = time.perf_counter()
        ik = _make_batch_ik_solver(curobo, str(self._robot_yaml), self.config)
        self.stats.setup_s = time.perf_counter() - setup_t0

        pose_dict = _batch_pose_dict(curobo, targets)
        frame_count = targets.position.shape[0]
        chunk = self.config.batch_size
        warmup_goal = _goal_from_poses(
            curobo,
            {name: _pad_pose(curobo, pose, 0, min(chunk, frame_count), chunk) for name, pose in pose_dict.items()},
        )
        warmup_t0 = time.perf_counter()
        ik.solve_pose(goal_tool_poses=warmup_goal)
        if hasattr(curobo.torch, "cuda") and curobo.torch.cuda.is_available():
            curobo.torch.cuda.synchronize()
        warmup_s = time.perf_counter() - warmup_t0

        solve_t0 = time.perf_counter()
        arm_qpos, _ = run_batch_ik(curobo, ik, targets, chunk)
        if hasattr(curobo.torch, "cuda") and curobo.torch.cuda.is_available():
            curobo.torch.cuda.synchronize()
        solve_s = time.perf_counter() - solve_t0
        return arm_qpos, solve_s, warmup_s

    def _solve_trajopt(self, targets: GripperTrajectory) -> tuple[np.ndarray, float, float]:
        """Reach each TCP pair from the previous joints with B-spline TrajOpt."""
        curobo = self._curobo
        pose_dict = _batch_pose_dict(curobo, targets)
        frame_count = targets.position.shape[0]
        setup_t0 = time.perf_counter()
        ik = _make_batch_ik_solver(
            curobo,
            str(self._robot_yaml),
            CuroboIKConfig(
                mode="seq",
                num_seeds=self.config.num_seeds,
                batch_size=1,
                position_tolerance=self.config.position_tolerance,
                orientation_tolerance=self.config.orientation_tolerance,
                orientation_weight=self.config.orientation_weight,
                urdf_path=self.config.urdf_path,
                yaml_path=self.config.yaml_path,
            ),
            use_cuda_graph=True,
        )
        trajopt = _make_trajopt_solver(curobo, str(self._robot_yaml), self.config)
        self.stats.setup_s = time.perf_counter() - setup_t0

        first_goal = _goal_from_poses(
            curobo,
            {name: _pad_pose(curobo, pose, 0, 1, 1) for name, pose in pose_dict.items()},
        )
        joint_names = list(getattr(trajopt, "joint_names", getattr(ik, "joint_names", ARM_JOINTS)))
        previous = np.zeros(len(joint_names), dtype=np.float32)
        warmup_state, warmup_seed = _warm_start(
            curobo, previous, joint_names, self.config.num_seeds, noise_std=0.8
        )
        warmup_t0 = time.perf_counter()
        ik.solve_pose(
            goal_tool_poses=first_goal,
            seed_config=warmup_seed,
            current_state=warmup_state,
        )
        trajopt.solve_pose(
            goal_tool_poses=first_goal,
            current_state=_joint_state(curobo, trajopt, previous, joint_names),
        )
        if hasattr(curobo.torch, "cuda") and curobo.torch.cuda.is_available():
            curobo.torch.cuda.synchronize()
        warmup_s = time.perf_counter() - warmup_t0

        solutions: list[np.ndarray] = []
        solve_t0 = time.perf_counter()
        seed_state, seed_config = _warm_start(
            curobo, previous, joint_names, self.config.num_seeds, noise_std=0.8
        )
        first = ik.solve_pose(
            goal_tool_poses=first_goal,
            seed_config=seed_config,
            current_state=seed_state,
        )
        joint_names = _result_joint_names(ik, first)
        previous = _raw_joint_positions(first)[0]
        solutions.append(previous)
        for frame in range(1, frame_count):
            goal = _goal_from_poses(
                curobo,
                {name: _pad_pose(curobo, pose, frame, frame + 1, 1) for name, pose in pose_dict.items()},
            )
            current_state = _joint_state(curobo, trajopt, previous, joint_names)
            result = trajopt.solve_pose(goal_tool_poses=goal, current_state=current_state)
            joint_names = _result_joint_names(trajopt, result)
            previous = _trajopt_endpoint(result)
            solutions.append(previous)
        if hasattr(curobo.torch, "cuda") and curobo.torch.cuda.is_available():
            curobo.torch.cuda.synchronize()
        solve_s = time.perf_counter() - solve_t0
        return _reorder_joints(np.stack(solutions, axis=0), joint_names), solve_s, warmup_s

    def _solve_mpc(self, targets: GripperTrajectory) -> tuple[np.ndarray, float, float]:
        """Global IK on frame 0, then receding-horizon MPC for the rest."""
        curobo = self._curobo
        setup_t0 = time.perf_counter()
        cfg = curobo.MotionRetargeterCfg.create(
            robot=str(self._robot_yaml),
            tool_pose_criteria=_tool_pose_criteria(curobo, self.config.orientation_weight),
            num_envs=1,
            use_mpc=True,
            self_collision_check=False,
            load_collision_spheres=False,
            scene_model=None,
            optimization_dt=self.config.optimization_dt,
            num_seeds_global=self.config.num_seeds,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            steps_per_target=self.config.steps_per_target,
            ik_optimizer_configs=["ik/lbfgs_ik.yml"],
        )
        retargeter = curobo.MotionRetargeter(cfg)
        self.stats.setup_s = time.perf_counter() - setup_t0

        sequence = _sequence_goal(curobo, targets, list(retargeter.tool_frames))
        solve_t0 = time.perf_counter()
        result = retargeter.solve_sequence(sequence)
        if hasattr(curobo.torch, "cuda") and curobo.torch.cuda.is_available():
            curobo.torch.cuda.synchronize()
        solve_s = time.perf_counter() - solve_t0
        return _joint_positions(result, retargeter.joint_names), solve_s, 0.0

    def _to_mujoco_trajectory(self, targets: GripperTrajectory, arm_qpos: np.ndarray) -> IKTrajectory:
        frame_count = targets.position.shape[0]
        qpos_out = np.zeros((frame_count, self.model.nq), dtype=np.float64)
        ctrl_out = np.zeros((frame_count, self.model.nu), dtype=np.float64)
        position_error = np.full((frame_count, 2), np.nan, dtype=np.float64)
        orientation_error = np.full((frame_count, 2), np.nan, dtype=np.float64)
        converged = np.zeros((frame_count, 2), dtype=bool)
        if self.model.nkey:
            qpos_out[:] = self.model.key_qpos[0]
        arm_indices = self.evaluator.arm_qpos_indices
        if arm_qpos.shape[1] != len(arm_indices):
            raise ValueError(
                f"cuRobo returned {arm_qpos.shape[1]} arm joints, expected {len(arm_indices)}"
            )

        for frame in range(frame_count):
            qpos = qpos_out[frame].copy()
            qpos[arm_indices] = np.clip(
                arm_qpos[frame],
                self.evaluator._arm_lower,
                self.evaluator._arm_upper,
            )
            self.evaluator._write_grippers(qpos, targets.width[frame], targets.valid[frame])
            self.evaluator.data.qpos[:] = qpos
            mujoco.mj_forward(self.evaluator.model, self.evaluator.data)
            frame_position_error, frame_orientation_error = self.evaluator._task_errors(targets, frame)
            position_error[frame] = frame_position_error
            orientation_error[frame] = frame_orientation_error
            valid = targets.valid[frame]
            converged[frame, valid] = (
                (frame_position_error[valid] <= self.ik_config.position_tolerance)
                & (frame_orientation_error[valid] <= self.ik_config.orientation_tolerance)
            )
            qpos_out[frame] = qpos
            ctrl_out[frame] = qpos[self.evaluator._ctrl_qpos_indices]
        return IKTrajectory(qpos_out, ctrl_out, position_error, orientation_error, converged)


def _import_curobo() -> Any:
    try:
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.motion_retargeter import (
            MotionRetargeter,
            MotionRetargeterCfg,
            SequenceGoalToolPose,
            ToolPoseCriteria,
        )
        from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
        from curobo.types import GoalToolPose, JointState, Pose
    except ImportError as exc:
        raise ImportError(
            "cuRobo is not installed in this Python environment. Install the NVIDIA cuRobo v2 "
            "package (import path `curobo.inverse_kinematics`) before running this solver."
        ) from exc

    namespace = type("CuroboAPI", (), {})()
    namespace.torch = torch
    namespace.InverseKinematics = InverseKinematics
    namespace.InverseKinematicsCfg = InverseKinematicsCfg
    namespace.MotionRetargeter = MotionRetargeter
    namespace.MotionRetargeterCfg = MotionRetargeterCfg
    namespace.SequenceGoalToolPose = SequenceGoalToolPose
    namespace.GoalToolPose = GoalToolPose
    namespace.Pose = Pose
    namespace.JointState = JointState
    namespace.ToolPoseCriteria = ToolPoseCriteria
    namespace.TrajectoryOptimizer = TrajectoryOptimizer
    namespace.TrajectoryOptimizerCfg = TrajectoryOptimizerCfg
    return namespace


def _materialize_robot_yaml(yaml_path: Path, urdf_path: Path) -> Path:
    yaml_path = yaml_path.expanduser().resolve()
    urdf_path = urdf_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"cuRobo URDF does not exist: {urdf_path}")
    resolved = Path("/tmp") / "arx_acone.resolved.yml"
    joints = "\n".join(f"        - {name}" for name in ARM_JOINTS)
    zeros = ", ".join(["0.0"] * len(ARM_JOINTS))
    ones = ", ".join(["1.0"] * len(ARM_JOINTS))
    resolved.write_text(
        "\n".join((
            "robot_cfg:",
            "  kinematics:",
            f'    urdf_path: "{urdf_path}"',
            f'    asset_root_path: "{urdf_path.parent}"',
            '    base_link: "base_link"',
            '    tool_frames: ["left_tcp", "right_tcp"]',
            "    extra_links: null",
            "    collision_link_names: null",
            "    collision_spheres: null",
            "    self_collision_ignore: null",
            "    cspace:",
            "      joint_names:",
            joints,
            f"      default_joint_position: [{zeros}]",
            f"      null_space_weight: [{ones}]",
            f"      cspace_distance_weight: [{ones}]",
            "      max_acceleration: 15.0",
            "      max_jerk: 500.0",
            "",
        )),
        encoding="utf-8",
    )
    return resolved


def run_batch_ik(curobo: Any, ik: Any, targets: GripperTrajectory, batch_size: int) -> tuple[np.ndarray, list[str]]:
    """Solve every frame independently with batched cuRobo IK."""
    pose_dict = _batch_pose_dict(curobo, targets)
    frame_count = targets.position.shape[0]
    chunks: list[np.ndarray] = []
    joint_names: list[str] | None = None
    for start in range(0, frame_count, batch_size):
        end = min(start + batch_size, frame_count)
        goal = _goal_from_poses(
            curobo,
            {name: _pad_pose(curobo, pose, start, end, batch_size) for name, pose in pose_dict.items()},
        )
        result = ik.solve_pose(goal_tool_poses=goal)
        if joint_names is None:
            joint_names = _result_joint_names(ik, result)
        chunks.append(_batch_joint_positions(result, joint_names)[: end - start])
    return np.concatenate(chunks, axis=0), joint_names or list(ARM_JOINTS)


def _make_batch_ik_solver(
    curobo: Any,
    robot_yaml: str,
    config: CuroboIKConfig,
    use_cuda_graph: bool = False,
):
    kwargs = {
        "robot": robot_yaml,
        "num_seeds": config.num_seeds,
        "self_collision_check": False,
        "use_cuda_graph": use_cuda_graph,
        "load_collision_spheres": False,
        "max_batch_size": config.batch_size,
        "position_tolerance": config.position_tolerance,
        "orientation_tolerance": config.orientation_tolerance,
        # Particle MPPI has empty cost tensors for dual-TCP + no collision; L-BFGS matches seq mode.
        "optimizer_configs": ["ik/lbfgs_ik.yml"],
    }
    try:
        ik_cfg = curobo.InverseKinematicsCfg.create(**kwargs)
    except TypeError:
        kwargs.pop("use_cuda_graph", None)
        try:
            ik_cfg = curobo.InverseKinematicsCfg.create(**kwargs)
        except TypeError:
            ik_cfg = curobo.InverseKinematicsCfg.create(robot=robot_yaml)
    ik = curobo.InverseKinematics(ik_cfg)
    if hasattr(ik, "update_tool_pose_criteria"):
        ik.update_tool_pose_criteria(_tool_pose_criteria(curobo, config.orientation_weight))
    return ik


def _make_trajopt_solver(curobo: Any, robot_yaml: str, config: CuroboIKConfig):
    kwargs = {
        "robot": robot_yaml,
        "num_seeds": config.trajopt_seeds,
        "self_collision_check": False,
        "use_cuda_graph": True,
        "load_collision_spheres": False,
        "max_batch_size": 1,
        "position_tolerance": config.position_tolerance,
        "orientation_tolerance": config.orientation_tolerance,
    }
    try:
        trajopt_cfg = curobo.TrajectoryOptimizerCfg.create(**kwargs)
    except TypeError:
        kwargs.pop("use_cuda_graph", None)
        trajopt_cfg = curobo.TrajectoryOptimizerCfg.create(**kwargs)
    trajopt = curobo.TrajectoryOptimizer(trajopt_cfg)
    if hasattr(trajopt, "update_tool_pose_criteria"):
        trajopt.update_tool_pose_criteria(_tool_pose_criteria(curobo, config.orientation_weight))
    return trajopt


def _tool_pose_criteria(curobo: Any, orientation_weight: float) -> dict[str, Any]:
    return {
        name: curobo.ToolPoseCriteria.track_position_and_orientation(
            xyz=[1.0, 1.0, 1.0],
            rpy=[orientation_weight] * 3,
        )
        for name in TCP_FRAMES
    }


def _rotation_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_quat().reshape(rotation.shape[:-2] + (4,))
    return xyzw[..., [3, 0, 1, 2]]


def _held_targets(targets: GripperTrajectory) -> tuple[np.ndarray, np.ndarray]:
    """Replace invalid frames with the last valid pose so cuRobo always has a goal."""
    position = targets.position.copy()
    rotation = targets.rotation.copy()
    for side in range(2):
        last_pos = None
        last_rot = None
        for frame in range(position.shape[0]):
            if targets.valid[frame, side]:
                last_pos = position[frame, side]
                last_rot = rotation[frame, side]
                continue
            if last_pos is None:
                continue
            position[frame, side] = last_pos
            rotation[frame, side] = last_rot
        if last_pos is None:
            continue
        for frame in range(position.shape[0]):
            if not np.isfinite(position[frame, side]).all():
                position[frame, side] = last_pos
                rotation[frame, side] = last_rot
    return position, rotation


def _as_torch(curobo: Any, array: np.ndarray):
    return curobo.torch.tensor(
        np.ascontiguousarray(array),
        dtype=curobo.torch.float32,
        device="cuda",
    ).contiguous()


def _batch_pose_dict(curobo: Any, targets: GripperTrajectory) -> dict[str, Any]:
    position, rotation = _held_targets(targets)
    quats = _rotation_to_wxyz(rotation)
    poses = {}
    for side, name in enumerate(TCP_FRAMES):
        poses[name] = curobo.Pose(
            position=_as_torch(curobo, position[:, side]),
            quaternion=_as_torch(curobo, quats[:, side]),
            name=name,
            normalize_rotation=True,
        )
    return poses


def _goal_from_poses(curobo: Any, pose_dict: dict[str, Any]):
    return curobo.GoalToolPose.from_poses(pose_dict, ordered_tool_frames=list(TCP_FRAMES))


def _sequence_goal(curobo: Any, targets: GripperTrajectory, tool_frames: list[str]):
    position, rotation = _held_targets(targets)
    quats = _rotation_to_wxyz(rotation)
    ordered = np.stack([position[:, TCP_FRAMES.index(name)] for name in tool_frames], axis=1)
    ordered_q = np.stack([quats[:, TCP_FRAMES.index(name)] for name in tool_frames], axis=1)
    # SequenceGoalToolPose: (num_frames, num_envs, num_links, num_goalset, 3/4)
    return curobo.SequenceGoalToolPose(
        tool_frames=tool_frames,
        position=_as_torch(curobo, ordered[:, None, :, None, :]),
        quaternion=_as_torch(curobo, ordered_q[:, None, :, None, :]),
    )


def _pad_pose(curobo: Any, pose: Any, start: int, end: int, width: int):
    position = pose.position[start:end]
    quaternion = pose.quaternion[start:end]
    pad = width - int(position.shape[0])
    if pad > 0:
        position = curobo.torch.cat((position, position[-1:].expand(pad, -1)), dim=0).contiguous()
        quaternion = curobo.torch.cat((quaternion, quaternion[-1:].expand(pad, -1)), dim=0).contiguous()
    return curobo.Pose(
        position=position.contiguous(),
        quaternion=quaternion.contiguous(),
        name=pose.name,
        normalize_rotation=True,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _reorder_joints(values: np.ndarray, source_names: list[str]) -> np.ndarray:
    missing = [name for name in ARM_JOINTS if name not in source_names]
    if missing:
        raise ValueError(f"cuRobo solution is missing ARX arm joints: {missing}")
    index = [source_names.index(name) for name in ARM_JOINTS]
    return np.asarray(values)[..., index]


def _raw_joint_positions(result: Any) -> np.ndarray:
    joint_state = _result_joint_state(result)
    if joint_state is not None:
        values = _to_numpy(joint_state.position)
    else:
        values = _to_numpy(result.solution if hasattr(result, "solution") else result.q)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        values = values[:, 0] if values.shape[1] == 1 else values.reshape(values.shape[0], -1)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"Expected cuRobo joints of rank 2, got {values.shape}")
    return values


def _joint_state(curobo: Any, solver: Any, qpos: np.ndarray, joint_names: list[str]):
    names = list(joint_names) or list(getattr(solver, "joint_names", ARM_JOINTS))
    tensor = _as_torch(curobo, np.asarray(qpos, dtype=np.float32).reshape(1, -1))
    return curobo.JointState.from_position(tensor, joint_names=names)


def _trajopt_endpoint(result: Any) -> np.ndarray:
    """Last waypoint of the best TrajOpt seed, used as the next start state."""
    joint_state = _result_joint_state(result)
    if joint_state is not None:
        values = _to_numpy(joint_state.position)
    else:
        values = _to_numpy(result.solution)
    values = np.asarray(values, dtype=np.float64)
    while values.ndim > 2:
        values = values[0]
    if values.ndim == 1:
        return values
    return values[-1]


def _warm_start(
    curobo: Any,
    qpos: np.ndarray,
    joint_names: list[str],
    num_seeds: int,
    noise_std: float = 0.05,
):
    """Build current_state plus (1, num_seeds, dof) seeds around the previous solution."""
    tensor = _as_torch(curobo, np.asarray(qpos, dtype=np.float32).reshape(1, -1))
    current_state = curobo.JointState.from_position(tensor, joint_names=list(joint_names))
    seed_config = tensor.view(1, 1, -1).expand(1, num_seeds, -1).contiguous()
    if num_seeds > 1 and noise_std > 0.0:
        noise = noise_std * curobo.torch.randn_like(seed_config)
        noise[:, 0] = 0.0
        seed_config = seed_config + noise
    return current_state, seed_config


def _result_joint_state(result: Any) -> Any:
    for candidate in (
        getattr(result, "joint_state", None),
        getattr(result, "js_solution", None),
    ):
        if candidate is not None:
            return candidate
    return None


def _result_joint_names(solver: Any, result: Any) -> list[str]:
    joint_state = _result_joint_state(result)
    for candidate in (
        getattr(joint_state, "joint_names", None),
        getattr(solver, "joint_names", None),
        getattr(getattr(solver, "kinematics", None), "joint_names", None),
    ):
        if candidate:
            return list(candidate)
    raise RuntimeError("Could not read joint names from the cuRobo IK result")


def _joint_positions(result: Any, joint_names: list[str]) -> np.ndarray:
    joint_state = _result_joint_state(result)
    if joint_state is not None:
        values = _to_numpy(joint_state.position)
        names = list(getattr(joint_state, "joint_names", joint_names) or joint_names)
    else:
        values = _to_numpy(result.solution if hasattr(result, "solution") else result.q)
        names = list(joint_names)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected cuRobo joint trajectory of rank 2, got {values.shape}")
    return _reorder_joints(values, names)


def _batch_joint_positions(result: Any, joint_names: list[str]) -> np.ndarray:
    joint_state = _result_joint_state(result)
    if joint_state is not None:
        values = _to_numpy(joint_state.position)
        names = list(getattr(joint_state, "joint_names", joint_names) or joint_names)
    else:
        values = _to_numpy(result.solution if hasattr(result, "solution") else result.q)
        names = list(joint_names)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        values = values[:, 0] if values.shape[1] == 1 else values.reshape(values.shape[0], -1)
    if values.ndim != 2:
        raise ValueError(f"Expected batched cuRobo joints of rank 2, got {values.shape}")
    return _reorder_joints(values, names)

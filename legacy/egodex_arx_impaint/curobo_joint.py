"""Joint base SE2 + full-trajectory IK, seeded by cuRobo batch IK.

Decision variables are the ARX base offset ``(dx, dy, yaw)`` and the full
``(T, 12)`` arm trajectory.  Pose tracking uses differentiable cuRobo FK;
temporal continuity, joint-space velocity, and jerk are extra quadratic costs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egodex_arx_replay.data import load_episode
from egodex_arx_replay.defaults import DEFAULT_EPISODE, DEFAULT_SCENE, EGODEX_FPS
from egodex_arx_replay.geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from egodex_arx_replay.gripper import GripperTrajectory, convert_episode_to_grippers
from egodex_arx_replay.smoothing import SmoothingConfig, smooth_gripper_trajectory

from .base_search import _apply_base_offset, _sample_offsets
from .curobo_ik import (
    TCP_FRAMES,
    ARXDualArmCuroboIKSolver,
    CuroboIKConfig,
    JointOptConfig,
    _as_torch,
    _batch_pose_dict,
    _goal_from_poses,
    _held_targets,
    _make_batch_ik_solver,
    _pad_pose,
    _reorder_joints,
    _rotation_to_wxyz,
    run_batch_ik,
)


def solve_joint_episode(solver: ARXDualArmCuroboIKSolver, targets: GripperTrajectory) -> tuple[np.ndarray, float, float]:
    """Seed with batched IK, then jointly refine base offset and the full trajectory."""
    curobo = solver._curobo
    joint_cfg = solver.config.joint
    setup_t0 = time.perf_counter()
    ik = _make_batch_ik_solver(curobo, str(solver._robot_yaml), solver.config)
    solver.stats.setup_s = time.perf_counter() - setup_t0

    pose_dict = _batch_pose_dict(curobo, targets)
    frame_count = targets.position.shape[0]
    chunk = solver.config.batch_size
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
    base, arm_qpos, losses = _optimize_joint(curobo, ik, targets, solver.config, joint_cfg)
    if hasattr(curobo.torch, "cuda") and curobo.torch.cuda.is_available():
        curobo.torch.cuda.synchronize()
    solve_s = time.perf_counter() - solve_t0

    if not np.isfinite(base).all() or not np.isfinite(arm_qpos).all():
        raise RuntimeError("Joint optimization produced non-finite base or joints")
    _apply_base_offset(solver.model, base)
    solver.stats.base_offset = base
    solver.stats.joint_losses = losses
    return arm_qpos, solve_s, warmup_s


def _optimize_joint(
    curobo: Any,
    ik: Any,
    targets: GripperTrajectory,
    ik_config: CuroboIKConfig,
    cfg: JointOptConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    torch = curobo.torch
    rng = np.random.default_rng(cfg.seed)
    bounds = np.array(
        [
            [-cfg.base_dx_bound, cfg.base_dx_bound],
            [-cfg.base_dy_bound, cfg.base_dy_bound],
            [-cfg.base_yaw_bound, cfg.base_yaw_bound],
        ]
    )
    offsets = _sample_offsets(
        rng,
        np.zeros(3, dtype=np.float64),
        np.array([0.12, 0.12, np.deg2rad(15.0)]),
        bounds,
        max(1, cfg.base_inits),
    )

    problem: _JointProblem | None = None
    init_base = offsets[0]
    init_q = None
    init_cost = np.inf
    for index, offset in enumerate(offsets):
        robot_targets = _transform_targets_to_base(targets, offset)
        q_arm, joint_names = run_batch_ik(curobo, ik, robot_targets, ik_config.batch_size)
        q_kin = _to_kinematics_order(q_arm, joint_names, list(ik.joint_names))
        if problem is None:
            problem = _JointProblem(curobo, ik, targets, cfg, ik_config.optimization_dt, q_kin, offset)
        else:
            problem.set_state(q_kin, offset)
        cost_terms = problem.eval_losses()
        cost = cost_terms["position"] + 0.02 * cost_terms["orientation"]
        print(
            f"  joint init {index + 1}/{len(offsets)}: cost={cost:.4f} "
            f"pos={cost_terms['position'] * 1000.0:.1f}mm "
            f"ori={np.degrees(cost_terms['orientation']):.1f}deg "
            f"base=({offset[0]:+.3f}, {offset[1]:+.3f}, {np.degrees(offset[2]):+.1f}deg) "
            f"|dq|={_mean_joint_step(q_kin):.4f}"
        )
        if cost < init_cost:
            init_cost = cost
            init_base = offset.copy()
            init_q = q_kin.copy()
    if problem is None or init_q is None:
        raise RuntimeError("cuRobo batch IK produced no joint seed for joint optimization")

    problem.set_state(init_q, init_base)
    problem.run()
    base = problem.base_numpy()
    q_kin = problem.q_numpy()
    losses = problem.eval_losses()
    print(
        "  joint opt: "
        f"loss={losses['total']:.4f} pos={losses['position'] * 1000.0:.1f}mm "
        f"ori={np.degrees(losses['orientation']):.2f}deg "
        f"cont={losses['continuity']:.4f} vel={losses['velocity']:.4f} jerk={losses['jerk']:.4f} "
        f"base=({base[0]:+.3f}, {base[1]:+.3f}, {np.degrees(base[2]):+.1f}deg) "
        f"|dq|={_mean_joint_step(q_kin):.4f}"
    )
    return base, _reorder_joints(q_kin, list(ik.joint_names)), losses


class _JointProblem:
    def __init__(
        self,
        curobo: Any,
        ik: Any,
        targets: GripperTrajectory,
        cfg: JointOptConfig,
        dt: float,
        q_init: np.ndarray,
        base_init: np.ndarray,
    ) -> None:
        torch = curobo.torch
        self.curobo = curobo
        self.torch = torch
        self.ik = ik
        self.cfg = cfg
        self.dt = float(dt)
        kinematics = ik.kinematics
        self.joint_names = list(ik.joint_names)
        self.fk_joint_names = list(ik.kinematics.joint_names)
        limits = kinematics.get_joint_limits()
        self.lower = limits.position[0].to(dtype=torch.float32)
        self.upper = limits.position[1].to(dtype=torch.float32)

        position, rotation = _held_targets(targets)
        goal_pos = _as_torch(curobo, position)
        goal_quat = _as_torch(curobo, _rotation_to_wxyz(rotation))
        frame_order = [TCP_FRAMES.index(name) for name in list(kinematics.tool_frames)]
        self.goal_pos = goal_pos[:, frame_order].contiguous()
        self.goal_quat = goal_quat[:, frame_order].contiguous()
        valid = np.asarray(targets.valid, dtype=np.float32)[:, frame_order]
        self.valid = _as_torch(curobo, valid)
        self.valid_scale = self.valid.sum().clamp_min(1.0)

        self.q = torch.nn.Parameter(_as_torch(curobo, q_init.astype(np.float32)))
        self.base = torch.nn.Parameter(_as_torch(curobo, base_init.astype(np.float32)))

    def set_state(self, q_init: np.ndarray, base_init: np.ndarray) -> None:
        with self.torch.no_grad():
            self.q.copy_(_as_torch(self.curobo, q_init.astype(np.float32)))
            self.base.copy_(_as_torch(self.curobo, base_init.astype(np.float32)))

    def run(self) -> None:
        if self.cfg.lbfgs_steps > 0:
            try:
                self._run_lbfgs()
                if self._is_finite():
                    return
            except RuntimeError as exc:
                print(f"  L-BFGS failed ({exc}); switching to Adam")
        self._run_adam()

    def _is_finite(self) -> bool:
        return bool(self.torch.isfinite(self.q).all().item() and self.torch.isfinite(self.base).all().item())

    def _snapshot(self) -> tuple[Any, Any]:
        return self.q.detach().clone(), self.base.detach().clone()

    def _restore(self, q_saved: Any, base_saved: Any) -> None:
        with self.torch.no_grad():
            self.q.copy_(q_saved)
            self.base.copy_(base_saved)

    def _run_lbfgs(self) -> None:
        torch = self.torch
        optimizer = torch.optim.LBFGS(
            [self.base, self.q],
            lr=0.8,
            max_iter=self.cfg.lbfgs_max_iter,
            history_size=self.cfg.history_size,
            line_search_fn="strong_wolfe",
        )
        for step in range(self.cfg.lbfgs_steps):
            saved = self._snapshot()

            def closure():
                optimizer.zero_grad(set_to_none=True)
                terms = self._loss_terms()
                total = terms["total"]
                if not torch.isfinite(total):
                    return total
                total.backward()
                return total

            loss = optimizer.step(closure)
            if not self._is_finite():
                self._restore(*saved)
                print(f"  joint L-BFGS {step + 1}: restored after non-finite step")
                return
            with torch.no_grad():
                self.q.clamp_(self.lower, self.upper)
            terms = self.eval_losses()
            loss_value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
            print(
                f"  joint L-BFGS {step + 1}/{self.cfg.lbfgs_steps}: loss={loss_value:.4f} "
                f"pos={terms['position'] * 1000.0:.1f}mm ori={np.degrees(terms['orientation']):.2f}deg "
                f"|dq|={_mean_joint_step(self.q_numpy()):.4f}"
            )

    def _run_adam(self) -> None:
        torch = self.torch
        optimizer = torch.optim.Adam(
            [
                {"params": [self.base], "lr": 0.002},
                {"params": [self.q], "lr": 0.005},
            ]
        )
        log_every = max(1, self.cfg.adam_steps // 5)
        for step in range(self.cfg.adam_steps):
            saved = self._snapshot()
            optimizer.zero_grad(set_to_none=True)
            terms = self._loss_terms()
            total = terms["total"]
            if not torch.isfinite(total):
                self._restore(*saved)
                print(f"  joint Adam {step + 1}: non-finite loss, stop")
                return
            total.backward()
            torch.nn.utils.clip_grad_norm_([self.base, self.q], 5.0)
            optimizer.step()
            if not self._is_finite():
                self._restore(*saved)
                print(f"  joint Adam {step + 1}: restored after non-finite step")
                return
            with torch.no_grad():
                self.q.clamp_(self.lower, self.upper)
                self.base[0].clamp_(-self.cfg.base_dx_bound, self.cfg.base_dx_bound)
                self.base[1].clamp_(-self.cfg.base_dy_bound, self.cfg.base_dy_bound)
                self.base[2].clamp_(-self.cfg.base_yaw_bound, self.cfg.base_yaw_bound)
            if (step + 1) % log_every == 0 or step == 0:
                logged = self.eval_losses()
                print(
                    f"  joint Adam {step + 1}/{self.cfg.adam_steps}: loss={logged['total']:.4f} "
                    f"pos={logged['position'] * 1000.0:.1f}mm ori={np.degrees(logged['orientation']):.2f}deg "
                    f"|dq|={_mean_joint_step(self.q_numpy()):.4f}"
                )

    def _loss_terms(self) -> dict[str, Any]:
        torch = self.torch
        cfg = self.cfg
        dt = self.dt
        q = self.q
        fk_pos, fk_quat = self._fk_scene()
        pos_err = torch.linalg.norm(fk_pos - self.goal_pos, dim=-1)
        quat_dot = (fk_quat * self.goal_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
        ori_chord = 1.0 - quat_dot.square()
        position = (pos_err * self.valid).sum() / self.valid_scale
        orientation_loss = (ori_chord * self.valid).sum() / self.valid_scale
        with torch.no_grad():
            orientation = (
                2.0 * torch.acos(quat_dot.clamp(max=1.0 - 1e-6)) * self.valid
            ).sum() / self.valid_scale

        delta = q[1:] - q[:-1]
        continuity = delta.square().sum(dim=-1).mean()
        velocity = (delta / dt).square().mean()
        if q.shape[0] >= 4:
            jerk = (q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]).square().mean()
        else:
            jerk = q.new_zeros(())

        below = torch.relu(self.lower - q)
        above = torch.relu(q - self.upper)
        joint_limit = below.square().mean() + above.square().mean()

        dx, dy, yaw = self.base[0], self.base[1], self.base[2]
        base_reg = cfg.base_reg_weight * (
            (dx / cfg.base_dx_bound) ** 2
            + (dy / cfg.base_dy_bound) ** 2
            + 0.5 * (yaw / cfg.base_yaw_bound) ** 2
        )
        base_bound = (
            torch.relu(dx.abs() - cfg.base_dx_bound).square()
            + torch.relu(dy.abs() - cfg.base_dy_bound).square()
            + torch.relu(yaw.abs() - cfg.base_yaw_bound).square()
        )

        total = (
            cfg.position_weight * position
            + cfg.orientation_weight * orientation_loss
            + cfg.continuity_weight * continuity
            + cfg.velocity_weight * velocity
            + cfg.jerk_weight * jerk
            + cfg.joint_limit_weight * joint_limit
            + base_reg
            + 10.0 * base_bound
        )
        return {
            "total": total,
            "position": position,
            "orientation": orientation,
            "continuity": continuity,
            "velocity": velocity,
            "jerk": jerk,
            "joint_limit": joint_limit,
            "base_reg": base_reg,
        }

    def _fk_scene(self) -> tuple[Any, Any]:
        kin_state = self.ik.kinematics.compute_kinematics(
            self.curobo.JointState.from_position(
                self.q.unsqueeze(0),
                joint_names=self.ik.kinematics.joint_names,
            )
        )
        tool = kin_state.tool_poses
        fk_quat = tool.quaternion[0]
        fk_quat = fk_quat / fk_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return _apply_base_se2(self.torch, tool.position[0], fk_quat, self.base)

    def eval_losses(self) -> dict[str, float]:
        with self.torch.no_grad():
            terms = self._loss_terms()
        return {key: float(value.detach().item()) for key, value in terms.items()}

    def q_numpy(self) -> np.ndarray:
        return self.q.detach().float().cpu().numpy().astype(np.float64)

    def base_numpy(self) -> np.ndarray:
        return self.base.detach().float().cpu().numpy().astype(np.float64)


def _apply_base_se2(torch: Any, position: Any, quaternion: Any, base: Any) -> tuple[Any, Any]:
    """Map robot-frame TCP poses into the scene with a yaw-about-z base offset."""
    dx, dy, yaw = base[0], base[1], base[2]
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    x, y, z = position.unbind(dim=-1)
    scene_pos = torch.stack((cos_y * x - sin_y * y + dx, sin_y * x + cos_y * y + dy, z), dim=-1)
    half = 0.5 * yaw
    ones = torch.ones_like(position[..., :1])
    zeros = torch.zeros_like(position[..., :1])
    yaw_quat = torch.cat((torch.cos(half).expand_as(ones), zeros, zeros, torch.sin(half).expand_as(ones)), dim=-1)
    return scene_pos, _quat_mul_wxyz(yaw_quat, quaternion)


def _quat_mul_wxyz(a: Any, b: Any):
    import torch as torch_mod

    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch_mod.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _transform_targets_to_base(targets: GripperTrajectory, offset: np.ndarray) -> GripperTrajectory:
    """Express scene-frame gripper targets in the robot base frame."""
    dx, dy, yaw = (float(offset[0]), float(offset[1]), float(offset[2]))
    rotation_z = Rotation.from_euler("z", yaw).as_matrix()
    translation = np.array([dx, dy, 0.0], dtype=np.float64)
    position = np.einsum("ji,...j->...i", rotation_z, targets.position - translation)
    rotation = np.einsum("ji,...jk->...ik", rotation_z, targets.rotation)
    return GripperTrajectory(
        position=position.astype(np.float64),
        rotation=rotation.astype(np.float64),
        width=targets.width.copy(),
        valid=targets.valid.copy(),
    )


def _to_kinematics_order(q_arm: np.ndarray, source_names: list[str], kin_names: list[str]) -> np.ndarray:
    missing = [name for name in kin_names if name not in source_names]
    if missing:
        raise ValueError(f"Batch IK is missing kinematics joints: {missing}")
    index = [source_names.index(name) for name in kin_names]
    return np.asarray(q_arm, dtype=np.float64)[..., index]


def _mean_joint_step(qpos: np.ndarray) -> float:
    if qpos.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(qpos, axis=0), axis=-1).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--base-inits", type=int, default=8)
    parser.add_argument("--lbfgs-steps", type=int, default=4)
    parser.add_argument("--continuity-weight", type=float, default=0.05)
    parser.add_argument("--velocity-weight", type=float, default=2.0e-4)
    parser.add_argument("--jerk-weight", type=float, default=0.20)
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
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
    episode = load_episode(args.episode.expanduser().resolve())
    scene_t_egodex = make_scene_T_egodex(episode.world_T_joint, np.asarray(args.scene_anchor))
    targets = convert_episode_to_grippers(episode, scene_t_egodex)
    if not args.no_smoothing:
        targets = smooth_gripper_trajectory(targets, SmoothingConfig(window=args.smoothing_window))
    scene_path = args.scene.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    joint_cfg = JointOptConfig(
        base_inits=args.base_inits,
        lbfgs_steps=args.lbfgs_steps,
        continuity_weight=args.continuity_weight,
        velocity_weight=args.velocity_weight,
        jerk_weight=args.jerk_weight,
    )
    solver = ARXDualArmCuroboIKSolver(
        model,
        CuroboIKConfig(mode="joint", num_seeds=args.num_seeds, joint=joint_cfg),
    )
    ik = solver.solve_episode(targets)
    offset = solver.stats.base_offset
    valid = targets.valid
    print(
        f"solved {ik.qpos.shape[0]} frames in {solver.stats.solve_s:.3f}s | "
        f"mean_pos={float(np.nanmean(ik.position_error[valid])) * 1000.0:.1f}mm | "
        f"max_pos={float(np.nanmax(ik.position_error[valid])) * 1000.0:.1f}mm | "
        f"mean |dq|={_mean_joint_step(ik.qpos):.4f}"
    )
    if offset is not None:
        print(
            f"base=({offset[0]:+.3f}, {offset[1]:+.3f}, {np.degrees(offset[2]):+.1f}deg)"
        )
    if args.output_json is not None:
        path = args.output_json.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "episode": str(args.episode),
                    "offset": None
                    if offset is None
                    else {
                        "dx": float(offset[0]),
                        "dy": float(offset[1]),
                        "yaw_rad": float(offset[2]),
                        "yaw_deg": float(np.degrees(offset[2])),
                    },
                    "setup_s": solver.stats.setup_s,
                    "warmup_s": solver.stats.warmup_s,
                    "solve_s": solver.stats.solve_s,
                    "mean_pos_error_m": float(np.nanmean(ik.position_error[valid])),
                    "max_pos_error_m": float(np.nanmax(ik.position_error[valid])),
                    "mean_ori_error_rad": float(np.nanmean(ik.orientation_error[valid])),
                    "mean_joint_step": _mean_joint_step(ik.qpos),
                    "losses": solver.stats.joint_losses,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

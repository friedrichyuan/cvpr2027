"""Compare existing MuJoCo Jacobian IK against cuRobo on one EgoDex episode."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from egodex_arx_replay.data import load_episode
from egodex_arx_replay.defaults import DEFAULT_EPISODE, DEFAULT_SCENE
from egodex_arx_replay.geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from egodex_arx_replay.gripper import GripperTrajectory, convert_episode_to_grippers
from egodex_arx_replay.ik import ARXDualArmIKSolver, IKConfig, IKTrajectory
from egodex_arx_replay.smoothing import SmoothingConfig, smooth_gripper_trajectory

from .base_search import _apply_base_offset, _count_contacts
from .curobo_ik import ARXDualArmCuroboIKSolver, CuroboIKConfig, CuroboMode

CUROBO_SOLVERS: dict[str, CuroboMode] = {
    "curobo": "seq",
    "curobo_seq": "seq",
    "curobo_batch": "batch",
    "curobo_trajopt": "trajopt",
    "curobo_mpc": "mpc",
    "curobo_joint": "joint",
}


@dataclass(frozen=True)
class SolverReport:
    name: str
    frames: int
    setup_s: float
    solve_s: float
    warmup_s: float
    mean_pos_mm: float
    max_pos_mm: float
    median_pos_mm: float
    mean_ori_deg: float
    max_ori_deg: float
    median_ori_deg: float
    converged_pct: float
    mean_joint_step: float
    max_joint_step: float
    contact_count: int
    base_dx: float = 0.0
    base_dy: float = 0.0
    base_yaw_deg: float = 0.0


def compare_ik(
    episode_path: Path,
    scene_path: Path,
    scene_anchor: np.ndarray,
    solvers: list[str],
    smoothing_window: int,
    no_smoothing: bool,
    max_frames: int | None,
    num_seeds: int,
    base_offset: np.ndarray,
    output_json: Path | None,
) -> list[SolverReport]:
    episode_path = resolve_episode_path(episode_path)
    episode = load_episode(episode_path)
    scene_t_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    targets = convert_episode_to_grippers(episode, scene_t_egodex)
    if not no_smoothing:
        targets = smooth_gripper_trajectory(targets, SmoothingConfig(window=smoothing_window))
    if max_frames is not None:
        targets = _slice_frames(targets, max_frames)

    print(f"Episode: {episode_path}")
    print(f"Frames: {targets.position.shape[0]} | valid TCP pairs: {int(targets.valid.all(axis=1).sum())}")
    reports: list[SolverReport] = []
    for name in solvers:
        print(f"\n=== {name} ===")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        _apply_base_offset(model, base_offset)
        report = _run_solver(name, model, targets, num_seeds)
        reports.append(report)
        _print_report(report)

    _print_table(reports)
    if output_json is not None:
        _write_json(output_json, episode_path, scene_path, reports)
        print(f"Wrote {output_json}")
    return reports


def resolve_episode_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = [path.with_suffix(".hdf5"), path.with_suffix(".h5df"), path.with_suffix(".h5")]
    if path.suffix == ".h5df":
        candidates.insert(0, path.with_suffix(".hdf5"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"EgoDex episode does not exist: {path}")


def _run_solver(name: str, model: mujoco.MjModel, targets: GripperTrajectory, num_seeds: int) -> SolverReport:
    if name == "mujoco":
        solver = ARXDualArmIKSolver(model, IKConfig())
        setup_s = 0.0
        t0 = time.perf_counter()
        ik = solver.solve_episode(targets)
        solve_s = time.perf_counter() - t0
        warmup_s = 0.0
        return _summarize(name, ik, model, targets, setup_s, solve_s, warmup_s)
    if name not in CUROBO_SOLVERS:
        known = "mujoco, " + ", ".join(label for label in CUROBO_SOLVERS if label != "curobo")
        raise ValueError(f"Unknown solver '{name}'. Use {known}.")
    mode = CUROBO_SOLVERS[name]
    solver = ARXDualArmCuroboIKSolver(model, CuroboIKConfig(mode=mode, num_seeds=num_seeds))
    ik = solver.solve_episode(targets)
    report = _summarize(name, ik, model, targets, solver.stats.setup_s, solver.stats.solve_s, solver.stats.warmup_s)
    offset = solver.stats.base_offset
    if offset is None:
        return report
    report = replace(
        report,
        base_dx=float(offset[0]),
        base_dy=float(offset[1]),
        base_yaw_deg=float(np.degrees(offset[2])),
    )
    print(
        f"base: dx={report.base_dx:+.3f} dy={report.base_dy:+.3f} "
        f"yaw={report.base_yaw_deg:+.1f}deg"
    )
    if solver.stats.joint_losses:
        losses = solver.stats.joint_losses
        print(
            "joint loss: "
            f"total={losses['total']:.4f} cont={losses['continuity']:.4f} "
            f"vel={losses['velocity']:.4f} jerk={losses['jerk']:.4f}"
        )
    return report


def _summarize(
    name: str,
    ik: IKTrajectory,
    model: mujoco.MjModel,
    targets: GripperTrajectory,
    setup_s: float,
    solve_s: float,
    warmup_s: float,
) -> SolverReport:
    valid = targets.valid
    pos = ik.position_error[valid]
    ori = ik.orientation_error[valid]
    joint_step = np.linalg.norm(np.diff(ik.qpos, axis=0), axis=-1)
    return SolverReport(
        name=name,
        frames=ik.qpos.shape[0],
        setup_s=setup_s,
        solve_s=solve_s,
        warmup_s=warmup_s,
        mean_pos_mm=float(np.nanmean(pos) * 1000.0),
        max_pos_mm=float(np.nanmax(pos) * 1000.0),
        median_pos_mm=float(np.nanmedian(pos) * 1000.0),
        mean_ori_deg=float(np.degrees(np.nanmean(ori))),
        max_ori_deg=float(np.degrees(np.nanmax(ori))),
        median_ori_deg=float(np.degrees(np.nanmedian(ori))),
        converged_pct=float(ik.converged[valid].mean() * 100.0),
        mean_joint_step=float(np.mean(joint_step)) if joint_step.size else 0.0,
        max_joint_step=float(np.max(joint_step)) if joint_step.size else 0.0,
        contact_count=int(_count_contacts(model, ik.qpos)),
    )


def _print_report(report: SolverReport) -> None:
    fps = report.frames / max(report.solve_s, 1.0e-9)
    print(
        f"time: setup {report.setup_s:.3f}s | warmup {report.warmup_s:.3f}s | "
        f"solve {report.solve_s:.3f}s | {fps:.1f} fps"
    )
    print(
        f"pos: mean {report.mean_pos_mm:.2f} mm | median {report.median_pos_mm:.2f} mm | "
        f"max {report.max_pos_mm:.2f} mm"
    )
    print(
        f"ori: mean {report.mean_ori_deg:.2f} deg | median {report.median_ori_deg:.2f} deg | "
        f"max {report.max_ori_deg:.2f} deg"
    )
    print(
        f"converged {report.converged_pct:.1f}% | mean |dq| {report.mean_joint_step:.4f} | "
        f"contacts {report.contact_count}"
    )


def _print_table(reports: list[SolverReport]) -> None:
    print("\n=== comparison ===")
    header = (
        f"{'solver':<16} {'setup_s':>8} {'solve_s':>8} {'fps':>8} "
        f"{'mean_mm':>9} {'max_mm':>9} {'mean_deg':>9} {'conv%':>7} {'|dq|':>8}"
    )
    print(header)
    for report in reports:
        fps = report.frames / max(report.solve_s, 1.0e-9)
        print(
            f"{report.name:<16} {report.setup_s:8.3f} {report.solve_s:8.3f} {fps:8.1f} "
            f"{report.mean_pos_mm:9.2f} {report.max_pos_mm:9.2f} {report.mean_ori_deg:9.2f} "
            f"{report.converged_pct:7.1f} {report.mean_joint_step:8.4f}"
        )


def _write_json(path: Path, episode_path: Path, scene_path: Path, reports: list[SolverReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode": str(episode_path),
        "scene": str(scene_path),
        "solvers": [
            {
                "name": report.name,
                "frames": report.frames,
                "setup_s": report.setup_s,
                "solve_s": report.solve_s,
                "warmup_s": report.warmup_s,
                "fps": report.frames / max(report.solve_s, 1.0e-9),
                "mean_pos_mm": report.mean_pos_mm,
                "max_pos_mm": report.max_pos_mm,
                "median_pos_mm": report.median_pos_mm,
                "mean_ori_deg": report.mean_ori_deg,
                "max_ori_deg": report.max_ori_deg,
                "median_ori_deg": report.median_ori_deg,
                "converged_pct": report.converged_pct,
                "mean_joint_step": report.mean_joint_step,
                "max_joint_step": report.max_joint_step,
                "contact_count": report.contact_count,
                "base_dx": report.base_dx,
                "base_dy": report.base_dy,
                "base_yaw_deg": report.base_yaw_deg,
            }
            for report in reports
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _slice_frames(targets: GripperTrajectory, count: int) -> GripperTrajectory:
    count = min(count, targets.position.shape[0])
    return GripperTrajectory(
        position=targets.position[:count].copy(),
        rotation=targets.rotation[:count].copy(),
        width=targets.width[:count].copy(),
        valid=targets.valid[:count].copy(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--solvers",
        default="mujoco,curobo_seq,curobo_batch,curobo_trajopt,curobo_mpc",
        help="Comma-separated: mujoco,curobo_seq,curobo_batch,curobo_trajopt,curobo_mpc,curobo_joint",
    )
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--base-dx", type=float, default=0.0)
    parser.add_argument("--base-dy", type=float, default=0.0)
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
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
    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {scene_path}")
    compare_ik(
        args.episode,
        scene_path,
        np.asarray(args.scene_anchor, dtype=np.float64),
        [name.strip() for name in args.solvers.split(",") if name.strip()],
        args.smoothing_window,
        args.no_smoothing,
        args.max_frames,
        args.num_seeds,
        np.array([args.base_dx, args.base_dy, np.deg2rad(args.base_yaw_deg)], dtype=np.float64),
        args.output_json.expanduser().resolve() if args.output_json else None,
    )


if __name__ == "__main__":
    main()

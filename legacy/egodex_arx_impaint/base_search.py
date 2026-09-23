"""Search ARX base (x, y, yaw) offsets for EgoDex IK feasibility."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egodex_arx_replay.data import load_episode
from egodex_arx_replay.defaults import DEFAULT_EPISODE, DEFAULT_SCENE
from egodex_arx_replay.geometry import DEFAULT_SCENE_ANCHOR, make_scene_T_egodex
from egodex_arx_replay.gripper import GripperTrajectory, convert_episode_to_grippers
from egodex_arx_replay.ik import ARXDualArmIKSolver, IKConfig
from egodex_arx_replay.smoothing import SmoothingConfig, smooth_gripper_trajectory


@dataclass(frozen=True)
class BaseSearchResult:
    offset: np.ndarray
    cost: float
    mean_pos_error: float
    max_pos_error: float
    mean_ori_error: float
    contact_count: int
    keyframes: np.ndarray


def search_base(
    episode_path: Path,
    scene_path: Path,
    scene_anchor: np.ndarray,
    keyframe_count: int,
    samples: int,
    iterations: int,
    elite_fraction: float,
    seed: int,
    output_json: Path | None,
) -> BaseSearchResult:
    episode = load_episode(episode_path)
    scene_t_egodex = make_scene_T_egodex(episode.world_T_joint, scene_anchor)
    targets = smooth_gripper_trajectory(
        convert_episode_to_grippers(episode, scene_t_egodex),
        SmoothingConfig(),
    )
    keyframes = _select_keyframes(targets, keyframe_count)
    key_targets = _slice_targets(targets, keyframes)
    rng = np.random.default_rng(seed)
    bounds = np.array([[-0.30, 0.30], [-0.30, 0.30], [np.deg2rad(-35.0), np.deg2rad(35.0)]])
    mean = np.zeros(3, dtype=np.float64)
    std = np.array([0.12, 0.12, np.deg2rad(15.0)], dtype=np.float64)
    best: BaseSearchResult | None = None

    for iteration in range(iterations):
        offsets = _sample_offsets(rng, mean, std, bounds, samples)
        results = [score_base_offset(scene_path, key_targets, keyframes, offset) for offset in offsets]
        results.sort(key=lambda result: result.cost)
        best = results[0] if best is None or results[0].cost < best.cost else best
        elite_count = max(2, int(round(samples * elite_fraction)))
        elites = np.stack([result.offset for result in results[:elite_count]])
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), np.array([0.01, 0.01, np.deg2rad(1.0)]))
        mean = np.clip(mean, bounds[:, 0], bounds[:, 1])
        print(
            f"iter {iteration + 1}/{iterations}: "
            f"best cost={results[0].cost:.4f}, "
            f"mean_pos={results[0].mean_pos_error * 1000.0:.1f}mm, "
            f"max_pos={results[0].max_pos_error * 1000.0:.1f}mm, "
            f"contacts={results[0].contact_count}, "
            f"offset=({results[0].offset[0]:+.3f}, {results[0].offset[1]:+.3f}, "
            f"{np.degrees(results[0].offset[2]):+.1f}deg)"
        )

    assert best is not None
    print(
        "best: "
        f"cost={best.cost:.4f}, "
        f"mean_pos={best.mean_pos_error * 1000.0:.1f}mm, "
        f"max_pos={best.max_pos_error * 1000.0:.1f}mm, "
        f"mean_ori={np.degrees(best.mean_ori_error):.1f}deg, "
        f"contacts={best.contact_count}, "
        f"offset=({best.offset[0]:+.3f}, {best.offset[1]:+.3f}, {np.degrees(best.offset[2]):+.1f}deg)"
    )
    if output_json is not None:
        _write_result(output_json, episode_path, scene_path, best)
    return best


def score_base_offset(
    scene_path: Path,
    targets: GripperTrajectory,
    keyframes: np.ndarray,
    offset: np.ndarray,
) -> BaseSearchResult:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _apply_base_offset(model, offset)
    solver = ARXDualArmIKSolver(
        model,
        IKConfig(step_iterations=40, independent_random_restarts=8, independent_max_nfev=1000),
    )
    ik = solver.solve_episode(targets)
    valid = targets.valid
    pos_errors = ik.position_error[valid]
    ori_errors = ik.orientation_error[valid]
    contact_count = _count_contacts(model, ik.qpos)
    mean_pos = float(np.nanmean(pos_errors))
    max_pos = float(np.nanmax(pos_errors))
    mean_ori = float(np.nanmean(ori_errors))
    regularization = 0.02 * float((offset[0] / 0.30) ** 2 + (offset[1] / 0.30) ** 2)
    regularization += 0.01 * float((offset[2] / np.deg2rad(35.0)) ** 2)
    collision_cost = 0.002 * contact_count
    cost = mean_pos + 0.5 * max_pos + 0.02 * mean_ori + collision_cost + regularization
    return BaseSearchResult(
        offset=offset,
        cost=cost,
        mean_pos_error=mean_pos,
        max_pos_error=max_pos,
        mean_ori_error=mean_ori,
        contact_count=contact_count,
        keyframes=keyframes,
    )


def _apply_base_offset(model: mujoco.MjModel, offset: np.ndarray) -> None:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        raise ValueError("Scene is missing body 'base_link'")
    model.body_pos[base_id, 0] += offset[0]
    model.body_pos[base_id, 1] += offset[1]
    yaw = Rotation.from_euler("z", float(offset[2]))
    base_rot = Rotation.from_quat(model.body_quat[base_id, [1, 2, 3, 0]])
    quat_xyzw = (yaw * base_rot).as_quat()
    model.body_quat[base_id] = quat_xyzw[[3, 0, 1, 2]]


def _count_contacts(model: mujoco.MjModel, qpos: np.ndarray) -> int:
    data = mujoco.MjData(model)
    total = 0
    for frame_qpos in qpos:
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        total += int(data.ncon)
    return total


def _select_keyframes(targets: GripperTrajectory, count: int) -> np.ndarray:
    frames = targets.position.shape[0]
    if count >= frames:
        return np.arange(frames, dtype=np.int32)
    uniform = np.linspace(0, frames - 1, count, dtype=np.int32)
    speed = np.linalg.norm(np.diff(targets.position, axis=0), axis=-1).mean(axis=-1)
    high_motion = np.argsort(speed)[-max(1, count // 4):] + 1
    selected = np.unique(np.concatenate(([0, frames - 1], uniform, high_motion))).astype(np.int32)
    if len(selected) > count:
        selected = selected[np.linspace(0, len(selected) - 1, count, dtype=np.int32)]
    return selected


def _slice_targets(targets: GripperTrajectory, frames: np.ndarray) -> GripperTrajectory:
    return GripperTrajectory(
        position=targets.position[frames].copy(),
        rotation=targets.rotation[frames].copy(),
        width=targets.width[frames].copy(),
        valid=targets.valid[frames].copy(),
    )


def _sample_offsets(
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    bounds: np.ndarray,
    samples: int,
) -> np.ndarray:
    offsets = rng.normal(mean[None, :], std[None, :], size=(samples, 3))
    offsets[0] = mean
    return np.clip(offsets, bounds[:, 0], bounds[:, 1])


def _write_result(path: Path, episode_path: Path, scene_path: Path, result: BaseSearchResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "episode": str(episode_path),
        "scene": str(scene_path),
        "offset": {
            "dx": float(result.offset[0]),
            "dy": float(result.offset[1]),
            "yaw_rad": float(result.offset[2]),
            "yaw_deg": float(np.degrees(result.offset[2])),
        },
        "cost": result.cost,
        "mean_pos_error_m": result.mean_pos_error,
        "max_pos_error_m": result.max_pos_error,
        "mean_ori_error_rad": result.mean_ori_error,
        "contact_count": result.contact_count,
        "keyframes": result.keyframes.tolist(),
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--keyframes", type=int, default=12)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
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
    search_base(
        args.episode.expanduser().resolve(),
        args.scene.expanduser().resolve(),
        np.asarray(args.scene_anchor, dtype=np.float64),
        args.keyframes,
        args.samples,
        args.iterations,
        args.elite_fraction,
        args.seed,
        args.output_json.expanduser().resolve() if args.output_json else None,
    )


if __name__ == "__main__":
    main()

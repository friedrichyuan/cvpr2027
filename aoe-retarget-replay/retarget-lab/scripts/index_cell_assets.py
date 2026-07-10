#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def cell_key(trajectory_6dof: str, hand_source: str, retargeting: str) -> str:
    return f"traj_{trajectory_6dof}__hand_{hand_source}__retarget_{retargeting}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def link_or_copy(src: Path | None, dst: Path, mode: str, root: Path) -> str | None:
    if src is None or not src.exists():
        return None
    if dst.exists() or dst.is_symlink():
        try:
            if src.resolve() == dst.resolve():
                return rel(dst, root)
        except FileNotFoundError:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    return rel(dst, root)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def object_id_from_task(task: str) -> str:
    for marker in ["_bimanual", "_right", "_left"]:
        if marker in task:
            return task.split(marker, 1)[0]
    return task


def write_do_as_i_do_object_6dof(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists():
        return None
    data = np.load(src, allow_pickle=False)
    payload = {}
    for key in ["qpos_obj_right", "qpos_obj_left", "qpos_obj"]:
        if key in data:
            payload[key] = data[key]
    if not payload:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **payload, source=str(src))
    return str(dst)


def do_as_i_do_roots(exp: Path, task: str, hand_type: str, retargeting: str, key: str) -> dict[str, Path | None]:
    shared = exp / "intermediates" / "do_as_i_do" / "retargeting_outputs"
    cell_root = exp / "intermediates" / "retargeting" / retargeting / key / "retargeting_outputs"
    root = cell_root if cell_root.exists() else shared
    mano = root / "mano" / hand_type / task
    robot_type = "sharpa"
    robot = root / robot_type / hand_type / task / "0"
    obj = root / "assets" / "objects" / task
    return {
        "root": root,
        "mano": mano,
        "robot": robot,
        "object": obj,
        "trajectory_keypoints": mano / "0" / "trajectory_keypoints.npz",
        "task_info": mano / "task_info.json",
        "object_visual": obj / "visual.obj",
        "object_convex": obj / "convex",
        "robot_type": Path(robot_type),
    }


def egoinfinity_roots(exp: Path, robot: str = "g1") -> dict[str, Path]:
    clip = exp / "intermediates" / "egoinfinity" / "clip"
    return {
        "clip": clip,
        "retarget": clip / "retarget" / robot,
        "retarget_legacy": clip / f"retarget_{robot}",
        "samples": clip / "retarget_samples",
    }


def spider_roots(exp: Path, key: str) -> dict[str, Path]:
    root = exp / "intermediates" / "retargeting" / "spider" / key
    return {
        "root": root,
        "processed": root / "processed",
        "robot": root / "robot",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create canonical experiment asset indexes for one 12-way demo cell.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trajectory-6dof", required=True, choices=["egoinfinity", "do_as_i_do"])
    parser.add_argument("--hand-source", required=True, choices=["aoe", "estimated"])
    parser.add_argument("--retargeting", required=True, choices=["egoinfinity", "do_as_i_do", "spider"])
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--hand-type", default="bimanual")
    parser.add_argument("--robot", default=None)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    exp = REPO_ROOT / "experiments" / args.run_name
    key = cell_key(args.trajectory_6dof, args.hand_source, args.retargeting)
    traj_assets = ensure_dir(exp / "assets" / "trajectory_6dof" / args.trajectory_6dof / args.task)
    cell_assets = ensure_dir(exp / "assets" / "cells" / key)
    cell_bundle = ensure_dir(exp / "cells" / key)
    manifest = {
        "cell_key": key,
        "settings": {
            "trajectory_6dof": args.trajectory_6dof,
            "hand_source": args.hand_source,
            "retargeting": args.retargeting,
            "task": args.task,
            "hand_type": args.hand_type,
            "robot": args.robot,
        },
        "experiment_root": str(exp),
        "assets": {},
        "missing": [],
    }

    def record(name: str, value: str | None) -> None:
        manifest["assets"][name] = value
        if value is None:
            manifest["missing"].append(name)

    def record_optional(name: str, value: str | None) -> None:
        manifest["assets"][name] = value

    # Trajectory-6DoF assets: object trajectory, object mesh, overlay, depth.
    if args.trajectory_6dof == "do_as_i_do":
        roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
        object_6dof_path = traj_assets / "object_6dof.npz"
        made = write_do_as_i_do_object_6dof(roots["trajectory_keypoints"], object_6dof_path)
        if not made and object_6dof_path.exists():
            made = str(object_6dof_path)
        record("trajectory_6dof.object_6dof_npz", rel(object_6dof_path, exp) if made else None)
        record("trajectory_6dof.object_visual_mesh", link_or_copy(first_existing([roots["object_visual"], traj_assets / "object_meshes" / "visual.obj"]), traj_assets / "object_meshes" / "visual.obj", args.mode, exp))
        record("trajectory_6dof.object_convex_meshes", link_or_copy(first_existing([roots["object_convex"], traj_assets / "object_meshes" / "convex"]), traj_assets / "object_meshes" / "convex", args.mode, exp))
        record("trajectory_6dof.task_info", link_or_copy(first_existing([roots["task_info"], traj_assets / "task_info.json"]), traj_assets / "task_info.json", args.mode, exp))
        record("trajectory_6dof.source_keypoints", link_or_copy(first_existing([roots["trajectory_keypoints"], traj_assets / "source_trajectory_keypoints.npz"]), traj_assets / "source_trajectory_keypoints.npz", args.mode, exp))
        object_id = object_id_from_task(args.task)
        did_clip = first_existing(
            [
                exp / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction",
                exp / "intermediates" / "do_as_i_do" / "reconstruction",
            ]
        ) or (exp / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction")
        raw_source = first_existing([did_clip / "raw_dir"]) or did_clip
        source_roots = [raw_source, did_clip]
        record("trajectory_6dof.raw_dir", link_or_copy(raw_source, traj_assets / "inputs" / "raw_dir", args.mode, exp))
        overlay_candidates = []
        depth_candidates = []
        for root in source_roots:
            overlay_candidates += [
                root / "mesh_overlay.mp4",
                root / "overlay.mp4",
                root / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
                root / f"output_tapir_{object_id}.mp4",
                root / "raw.mp4",
            ]
            depth_candidates += [
                root / "depth.mp4",
                root / "moge_depth.mp4",
                root / "pointmap_depth.mp4",
                root / "raw.mp4",
            ]
        record("trajectory_6dof.overlay_video", link_or_copy(first_existing(overlay_candidates), traj_assets / "overlay.mp4", args.mode, exp))
        record("trajectory_6dof.depth_video", link_or_copy(first_existing(depth_candidates), traj_assets / "depth.mp4", args.mode, exp))
    else:
        roots = egoinfinity_roots(exp)
        record("trajectory_6dof.raw_clip", link_or_copy(roots["clip"], traj_assets / "inputs" / "clip", args.mode, exp))
        record("trajectory_6dof.object_6dof_native", link_or_copy(first_existing([roots["clip"] / "pipeline_result.pkl.gz", roots["clip"] / "object_poses.npz"]), traj_assets / "object_6dof_native", args.mode, exp))
        record(
            "trajectory_6dof.object_visual_meshes",
            link_or_copy(
                first_existing([roots["clip"] / "sam3_meshes", roots["clip"] / "sam3d_objects", roots["clip"] / "objects"]),
                traj_assets / "object_meshes",
                args.mode,
                exp,
            ),
        )
        record("trajectory_6dof.overlay_video", link_or_copy(first_existing([
            roots["clip"] / "rgb_mesh_overlay.mp4",
        ]), traj_assets / "overlay.mp4", args.mode, exp))
        record("trajectory_6dof.depth_video", link_or_copy(first_existing([roots["samples"] / "depth.mp4", roots["clip"] / "depth.mp4"]), traj_assets / "depth.mp4", args.mode, exp))

    # Hand source assets.
    hand_dir = ensure_dir(cell_assets / "hand_source")
    if args.hand_source == "aoe":
        record("hand_source.aoe_hands", link_or_copy(first_existing([exp / "intermediates" / "egoinfinity" / "clip" / "aoe_hands.npz"]), hand_dir / "aoe_hands.npz", args.mode, exp))
    else:
        if args.trajectory_6dof == "do_as_i_do":
            roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
            record("hand_source.estimated_hands", link_or_copy(first_existing([roots["trajectory_keypoints"], traj_assets / "source_trajectory_keypoints.npz"]), hand_dir / "estimated_hands_and_keypoints.npz", args.mode, exp))
        else:
            roots = egoinfinity_roots(exp)
            record("hand_source.estimated_hands", link_or_copy(first_existing([roots["samples"] / "hand_joints.bin", roots["clip"] / "hand_joints.bin"]), hand_dir / "estimated_hand_joints", args.mode, exp))

    # Retargeting assets.
    ret_dir = ensure_dir(cell_assets / "retargeting" / args.retargeting)
    if args.retargeting == "do_as_i_do":
        roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
        for name in ["scene.xml", "scene_ik.xml", "scene_eq.xml"]:
            record(f"retargeting.scene.{name}", link_or_copy(roots["robot"] / name, ret_dir / "scenes" / name, args.mode, exp))
        for name in ["trajectory_kinematic.npz", "trajectory_ikrollout.npz", "trajectory_mjwp.npz", "config.yaml"]:
            record(f"retargeting.trajectory.{name}", link_or_copy(roots["robot"] / name, ret_dir / "trajectories" / name, args.mode, exp))
        record("retargeting.trajectory.trajectory_keypoints.npz", link_or_copy(roots["trajectory_keypoints"], ret_dir / "trajectories" / "trajectory_keypoints.npz", args.mode, exp))
        record_optional("retargeting.video.visualization_ik.mp4", link_or_copy(roots["robot"] / "visualization_ik.mp4", ret_dir / "videos" / "visualization_ik.mp4", args.mode, exp))
        record("retargeting.video.visualization_mjwp.mp4", link_or_copy(roots["robot"] / "visualization_mjwp.mp4", ret_dir / "videos" / "visualization_mjwp.mp4", args.mode, exp))
        robot_video = ret_dir / "videos" / "visualization_mjwp.mp4"
    elif args.retargeting == "egoinfinity":
        robot = args.robot or "g1"
        roots = egoinfinity_roots(exp, robot)
        rr = roots["retarget"] if roots["retarget"].exists() else roots["retarget_legacy"]
        record("retargeting.trajectory.trajectory_npz", link_or_copy(rr / "trajectory.npz", ret_dir / "trajectories" / "trajectory.npz", args.mode, exp))
        record("retargeting.video.input_viz", link_or_copy(rr / "input_viz.mp4", ret_dir / "videos" / "input_viz.mp4", args.mode, exp))
        record("retargeting.video.robot_sim", link_or_copy(rr / "robot_sim.mp4", ret_dir / "videos" / "robot_sim.mp4", args.mode, exp))
        robot_video = ret_dir / "videos" / "robot_sim.mp4"
    else:
        roots = spider_roots(exp, key)
        record("retargeting.trajectory.trajectory_mjwp", link_or_copy(first_existing([roots["root"] / "trajectory_mjwp.npz", roots["robot"] / "trajectory_mjwp.npz"]), ret_dir / "trajectories" / "trajectory_mjwp.npz", args.mode, exp))
        record("retargeting.scene.scene_xml", link_or_copy(first_existing([roots["root"] / "scene.xml", roots["robot"] / "scene.xml"]), ret_dir / "scenes" / "scene.xml", args.mode, exp))
        record("retargeting.video.visualization_mjwp", link_or_copy(first_existing([roots["root"] / "visualization_mjwp.mp4", roots["robot"] / "visualization_mjwp.mp4"]), ret_dir / "videos" / "visualization_mjwp.mp4", args.mode, exp))
        robot_video = ret_dir / "videos" / "visualization_mjwp.mp4"

    # Per-cell convenience bundle used by compose scripts.
    overlay = traj_assets / "overlay.mp4"
    depth = traj_assets / "depth.mp4"
    record("cell.overlay", link_or_copy(overlay if overlay.exists() else None, cell_bundle / "overlay.mp4", args.mode, exp))
    record("cell.depth", link_or_copy(depth if depth.exists() else None, cell_bundle / "depth.mp4", args.mode, exp))
    record("cell.robot", link_or_copy(robot_video if robot_video.exists() else None, cell_bundle / "robot.mp4", args.mode, exp))

    manifest_path = cell_assets / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (cell_bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cell_key": key, "manifest": str(manifest_path), "missing": manifest["missing"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe_retarget_lab.matrix import MatrixCell


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = ("egoinfinity", "do_as_i_do")
HAND_SOURCES = ("aoe", "estimated")
RETARGETERS = ("egoinfinity", "do_as_i_do", "spider")


def cell_key(trajectory: str, hand_source: str, retargeting: str) -> str:
    return MatrixCell(trajectory, hand_source, retargeting).key


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def materialize(src: Path | None, dst: Path, mode: str) -> Path | None:
    if src is None or not src.exists():
        return None
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
    elif mode == "hardlink" and src.is_file():
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    return dst


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def source_paths(source: Path, task: str, hand_type: str) -> dict[str, Path | None]:
    dai_key = cell_key("do_as_i_do", "estimated", "do_as_i_do")
    dai_root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / dai_key
        / "retargeting_outputs"
    )
    dai_robot = dai_root / "sharpa" / hand_type / task / "0"
    dai_mano = dai_root / "mano" / hand_type / task
    dai_obj = dai_root / "assets" / "objects" / task
    ego_clip = source / "intermediates" / "egoinfinity" / "clip"
    return {
        "ego_overlay": first_existing([
            ego_clip / "rgb_mesh_overlay.mp4",
        ]),
        "ego_depth": first_existing([
            ego_clip / "retarget_samples" / "depth.mp4",
            ego_clip / "retarget" / "g1" / "input_viz.mp4",
            ego_clip / "retarget_g1" / "input_viz.mp4",
        ]),
        "ego_robot": first_existing([ego_clip / "retarget" / "g1" / "robot_sim.mp4", ego_clip / "retarget_g1" / "robot_sim.mp4"]),
        "ego_pipeline": ego_clip / "pipeline_result.pkl.gz",
        "ego_meshes": first_existing([ego_clip / "sam3_meshes", ego_clip / "sam3d_objects", ego_clip / "objects"]),
        "ego_hands": first_existing([ego_clip / "retarget_samples" / "hand_joints.bin", ego_clip / "hand_joints.bin"]),
        "dai_overlay": first_existing([
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "mesh_overlay.mp4",
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "overlay.mp4",
        ]),
        "dai_depth": first_existing([
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "depth.mp4",
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "moge_depth.mp4",
        ]),
        "dai_robot": dai_robot / "visualization_mjwp.mp4",
        "dai_scene": dai_robot / "scene.xml",
        "dai_traj": dai_robot / "trajectory_mjwp.npz",
        "dai_keypoints": dai_mano / "0" / "trajectory_keypoints.npz",
        "dai_task_info": dai_mano / "task_info.json",
        "dai_object_visual": dai_obj / "visual.obj",
        "dai_object_convex": dai_obj / "convex",
    }


def prepare_trajectory_assets(dst: Path, srcs: dict[str, Path | None], task: str, mode: str) -> dict:
    report: dict[str, dict[str, str | None]] = {}
    for trajectory in TRAJECTORIES:
        root = ensure_dir(dst / "assets" / "trajectory_6dof" / trajectory / task)
        if trajectory == "egoinfinity":
            entries = {
                "overlay": materialize(srcs["ego_overlay"], root / "overlay.mp4", mode),
                "depth": materialize(srcs["ego_depth"], root / "depth.mp4", mode),
                "pipeline_result": materialize(srcs["ego_pipeline"], root / "object_6dof_native.pkl.gz", mode),
                "object_meshes": materialize(srcs["ego_meshes"], root / "object_meshes", mode),
                "estimated_hands": materialize(srcs["ego_hands"], root / "estimated_hands", mode),
            }
        else:
            entries = {
                "overlay": materialize(srcs["dai_overlay"], root / "overlay.mp4", mode),
                "depth": materialize(srcs["dai_depth"], root / "depth.mp4", mode),
                "source_keypoints": materialize(srcs["dai_keypoints"], root / "source_trajectory_keypoints.npz", mode),
                "task_info": materialize(srcs["dai_task_info"], root / "task_info.json", mode),
                "object_visual": materialize(srcs["dai_object_visual"], root / "object_meshes" / "visual.obj", mode),
                "object_convex": materialize(srcs["dai_object_convex"], root / "object_meshes" / "convex", mode),
            }
        write_json(root / "reuse_source_manifest.json", {k: rel(v, dst) for k, v in entries.items()})
        report[trajectory] = {k: rel(v, dst) for k, v in entries.items()}
    return report


def run_spider_cells(args: argparse.Namespace, dst: Path) -> list[dict]:
    if args.run_spider == "none":
        return []
    cells: list[tuple[str, str]] = []
    if args.run_spider in {"do_as_i_do", "all"}:
        cells += [("do_as_i_do", hand) for hand in HAND_SOURCES]
    if args.run_spider == "all":
        cells += [("egoinfinity", hand) for hand in HAND_SOURCES]

    results = []
    env = os.environ.copy()
    if args.spider_python:
        env["SPIDER_PYTHON"] = args.spider_python
    if args.spider_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.spider_cuda_visible_devices
    env["SPIDER_DEVICE"] = args.spider_device
    env["SPIDER_MAX_SIM_STEPS"] = str(args.spider_max_sim_steps)
    env["SPIDER_NUM_SAMPLES"] = str(args.spider_num_samples)
    env["SPIDER_MAX_NUM_ITERATIONS"] = str(args.spider_max_num_iterations)
    ensure_dir(dst / "logs")

    for trajectory, hand_source in cells:
        key = cell_key(trajectory, hand_source, "spider")
        robot = dst / "intermediates" / "retargeting" / "spider" / key / "visualization_mjwp.mp4"
        if args.skip_existing_spider and robot.exists():
            results.append({"cell": key, "status": "skipped_existing", "robot": rel(robot, dst)})
            continue
        cmd = [
            str(REPO_ROOT / "scripts" / "run_spider_retarget.sh"),
            "--run-name",
            args.run_name,
            "--trajectory-6dof",
            trajectory,
            "--hand-source",
            hand_source,
            "--task",
            args.task,
            "--hand-type",
            args.hand_type,
            "--robot-type",
            args.spider_robot_type,
            "--dataset-name",
            args.spider_dataset_name,
            "--data-id",
            str(args.spider_data_id),
            "--device",
            args.spider_device,
        ]
        if args.spider_skip_mjwp:
            cmd.append("--skip-mjwp")
        log_path = dst / "logs" / f"reuse12_spider_{key}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        results.append(
            {
                "cell": key,
                "status": "ok" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "log": rel(log_path, dst),
                "robot": rel(robot, dst) if robot.exists() else None,
            }
        )
        if proc.returncode != 0 and not args.keep_going:
            raise SystemExit(f"SPIDER failed for {key}; see {log_path}")
    return results


def find_spider_robot(dst: Path, key: str) -> Path | None:
    root = dst / "intermediates" / "retargeting" / "spider" / key
    return first_existing([root / "visualization_mjwp.mp4", root / "robot" / "visualization_mjwp.mp4"])


def compose_cell(dst: Path, key: str, duration: float) -> Path:
    out = dst / "videos" / f"{key}__triptych.mp4"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compose_triptych.py"),
        "--overlay",
        str(dst / "cells" / key / "overlay.mp4"),
        "--depth",
        str(dst / "cells" / key / "depth.mp4"),
        "--robot",
        str(dst / "cells" / key / "robot.mp4"),
        "--output",
        str(out),
        "--layout",
        "vertical",
        "--labels",
        "reconstruction overlay|depth / input geometry|retargeting robot",
    ]
    if duration > 0:
        cmd += ["--duration", str(duration)]
    subprocess.run(cmd, check=True)
    return out


def build_cells(dst: Path, srcs: dict[str, Path | None], args: argparse.Namespace) -> list[dict]:
    reports = []
    fallback_spider = None
    if args.allow_spider_fallback:
        fallback_spider = first_existing(
            [
                find_spider_robot(dst, cell_key("do_as_i_do", "estimated", "spider")) or Path("__missing__"),
                find_spider_robot(dst, cell_key("do_as_i_do", "aoe", "spider")) or Path("__missing__"),
            ]
        )

    for trajectory in TRAJECTORIES:
        overlay = dst / "assets" / "trajectory_6dof" / trajectory / args.task / "overlay.mp4"
        depth = dst / "assets" / "trajectory_6dof" / trajectory / args.task / "depth.mp4"
        for hand_source in HAND_SOURCES:
            for retargeting in RETARGETERS:
                key = cell_key(trajectory, hand_source, retargeting)
                cell_dir = ensure_dir(dst / "cells" / key)
                assets_dir = ensure_dir(dst / "assets" / "cells" / key)
                if retargeting == "egoinfinity":
                    robot_src = srcs["ego_robot"]
                    robot_note = "reused from source EgoInfinity/G1 full run"
                elif retargeting == "do_as_i_do":
                    robot_src = srcs["dai_robot"]
                    robot_note = "reused from source Do-as-I-Do/Sharpa full run"
                else:
                    exact = find_spider_robot(dst, key)
                    robot_src = exact or fallback_spider
                    robot_note = "exact SPIDER cell" if exact else "fallback SPIDER robot reused from Do-as-I-Do trajectory cell"

                overlay_dst = materialize(overlay, cell_dir / "overlay.mp4", args.mode)
                depth_dst = materialize(depth, cell_dir / "depth.mp4", args.mode)
                robot_dst = materialize(robot_src, cell_dir / "robot.mp4", args.mode)
                missing = [
                    name
                    for name, value in {
                        "overlay": overlay_dst,
                        "depth": depth_dst,
                        "robot": robot_dst,
                    }.items()
                    if value is None
                ]
                video = None
                if args.compose and not missing:
                    video = compose_cell(dst, key, args.duration)
                manifest = {
                    "cell_key": key,
                    "settings": {
                        "trajectory_6dof": trajectory,
                        "hand_source": hand_source,
                        "retargeting": retargeting,
                        "task": args.task,
                        "hand_type": args.hand_type,
                    },
                    "reuse_policy": {
                        "source_run": args.source_run,
                        "overlay_depth": f"from {trajectory} trajectory_6dof full run",
                        "robot": robot_note,
                    },
                    "assets": {
                        "cell.overlay": rel(overlay_dst, dst),
                        "cell.depth": rel(depth_dst, dst),
                        "cell.robot": rel(robot_dst, dst),
                        "source.overlay": rel(overlay, dst),
                        "source.depth": rel(depth, dst),
                        "source.robot": rel(robot_src, dst) if robot_src else None,
                        "video.triptych": rel(video, dst) if video else None,
                    },
                    "missing": missing,
                }
                write_json(cell_dir / "manifest.json", manifest)
                write_json(assets_dir / "asset_manifest.json", manifest)
                reports.append(manifest)
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse one EgoInfinity full run and one Do-as-I-Do full run to expand a 12-cell AoE retargeting demo matrix."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--hand-type", default="bimanual")
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--run-spider", choices=["none", "do_as_i_do", "all"], default="do_as_i_do")
    parser.add_argument("--allow-spider-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-spider", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spider-python")
    parser.add_argument("--spider-cuda-visible-devices", default="")
    parser.add_argument("--spider-device", default="cuda:0")
    parser.add_argument("--spider-robot-type", default="xhand")
    parser.add_argument("--spider-dataset-name", default="do_as_i_do")
    parser.add_argument("--spider-data-id", type=int, default=0)
    parser.add_argument("--spider-max-sim-steps", default="-1")
    parser.add_argument("--spider-num-samples", default="1024")
    parser.add_argument("--spider-max-num-iterations", default="16")
    parser.add_argument("--spider-skip-mjwp", action="store_true")
    args = parser.parse_args()
    if not args.run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.source_run}__reuse12_{stamp}"
    return args


def main() -> int:
    args = parse_args()
    source = REPO_ROOT / "experiments" / args.source_run
    if not source.exists():
        raise SystemExit(f"source run does not exist: {source}")
    dst = REPO_ROOT / "experiments" / args.run_name
    ensure_dir(dst / "logs")
    ensure_dir(dst / "videos")
    srcs = source_paths(source, args.task, args.hand_type)
    materialize(source, dst / "reuse" / "source_run", args.mode)
    traj_report = prepare_trajectory_assets(dst, srcs, args.task, args.mode)
    spider_report = run_spider_cells(args, dst)
    cell_reports = build_cells(dst, srcs, args)
    summary = {
        "run_name": args.run_name,
        "source_run": args.source_run,
        "experiment_root": str(dst),
        "mode": args.mode,
        "task": args.task,
        "hand_type": args.hand_type,
        "trajectory_assets": traj_report,
        "spider": spider_report,
        "cells": len(cell_reports),
        "composed": sum(1 for item in cell_reports if item["assets"].get("video.triptych")),
        "missing_cells": {item["cell_key"]: item["missing"] for item in cell_reports if item["missing"]},
        "fallback_cells": [
            item["cell_key"]
            for item in cell_reports
            if item["reuse_policy"]["robot"].startswith("fallback")
        ],
    }
    write_json(dst / "reuse_12_demo_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

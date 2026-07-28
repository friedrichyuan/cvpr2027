#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoe_retarget_lab.io_utils import read_json_object as load_json_optional  # noqa: E402
from aoe_retarget_lab.path_utils import ensure_dir, relative_path as rel  # noqa: E402
from aoe_retarget_lab.task_utils import object_id_from_task  # noqa: E402
from aoe_retarget_lab.video_utils import probe_video  # noqa: E402


def cell_key(trajectory_6dof: str, hand_source: str, retargeting: str) -> str:
    return f"traj_{trajectory_6dof}__hand_{hand_source}__retarget_{retargeting}"


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


def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            if path.is_file() and path.suffix.lower() == ".mp4" and path.stat().st_size < 1024:
                continue
            return path
    return None


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
    raw_dir = root.parent / "raw_dir"
    return {
        "root": root,
        "raw_dir": raw_dir,
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


def dai_indexed_keypoint_paths(
    retargeting_dir: Path,
    *,
    hand_type: str,
    task: str,
    data_id: int | str = 0,
) -> tuple[Path, Path]:
    """Return the cell-specific keypoint path and legacy generic alias.

    The canonical path carries the MANO hand/task/data identity even in copy
    mode. The flat alias remains available for existing asset consumers.
    """

    identity = tuple(str(value).strip() for value in (hand_type, task, data_id))
    if any(not value or "/" in value for value in identity):
        raise ValueError("hand_type, task, and data_id must be non-empty path components")
    hand_label, task_label, data_label = identity
    trajectories = retargeting_dir / "trajectories"
    canonical = (
        trajectories
        / "mano"
        / hand_label
        / task_label
        / data_label
        / "trajectory_keypoints.npz"
    )
    return canonical, trajectories / "trajectory_keypoints.npz"


def dai_indexed_visual_mesh_path(retargeting_dir: Path, *, task: str) -> Path:
    task = str(task).strip()
    if not task or "/" in task:
        raise ValueError("task must be a non-empty path component")
    return retargeting_dir / "assets" / "objects" / task / "visual.obj"


def aoe_hand_candidates(
    exp: Path,
    task: str,
    hand_type: str,
    retargeting: str,
    key: str,
) -> list[Path]:
    candidates = [
        exp / "intermediates" / "egoinfinity" / "clip" / "aoe_hands.npz",
        exp
        / "intermediates"
        / "trajectory_6dof"
        / "do_as_i_do"
        / "reconstruction"
        / "clip"
        / "raw"
        / "all_hand_meshes.npz",
        exp
        / "intermediates"
        / "trajectory_6dof"
        / "egoinfinity"
        / "do_as_i_do_raw_dir_hand_aoe"
        / "raw"
        / "all_hand_meshes.npz",
        exp
        / "assets"
        / "trajectory_6dof"
        / "do_as_i_do"
        / task
        / "hand_meshes"
        / "all_hand_meshes.npz",
    ]
    if retargeting == "do_as_i_do":
        roots = do_as_i_do_roots(exp, task, hand_type, retargeting, key)
        candidates.insert(0, roots["raw_dir"] / "raw" / "all_hand_meshes.npz")
        candidates.insert(1, roots["raw_dir"] / "all_hand_meshes.npz")
    return candidates


def minimum_raw_dir_preflight(raw_dir: Path) -> dict[str, object]:
    """Check only the files required to start the official DAI backend."""

    required = [
        raw_dir / "config.json",
        raw_dir / "raw.mp4",
        raw_dir / "all_frames",
        raw_dir / "video_segmentation",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return {
        "status": "invalid" if missing else "ok",
        "policy": "minimum_backend_input",
        "raw_dir": str(raw_dir),
        "missing": missing,
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
    parser.add_argument(
        "--preflight-raw-dir",
        type=Path,
        default=None,
        help="Validate only this exact DAI raw-dir adapter before official launch.py.",
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=None,
        help="Optional JSON path for --preflight-raw-dir results.",
    )
    args = parser.parse_args()

    if args.preflight_raw_dir is not None:
        if args.retargeting != "do_as_i_do":
            parser.error("--preflight-raw-dir is only valid with --retargeting do_as_i_do")
        preflight = minimum_raw_dir_preflight(args.preflight_raw_dir)
        if args.preflight_output is not None:
            args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_output.write_text(
                json.dumps(preflight, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0 if preflight.get("status") == "ok" else 6

    exp = REPO_ROOT / "experiments" / args.run_name
    key = cell_key(args.trajectory_6dof, args.hand_source, args.retargeting)
    hand_selection = None
    if args.retargeting == "do_as_i_do":
        hand_selection = load_json_optional(
            exp
            / "logs"
            / f"do_as_i_do_hand_selection__{args.trajectory_6dof}__{args.hand_source}.json"
        )
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
        "provenance": {
            "cell_key": key,
            "trajectory_6dof": args.trajectory_6dof,
            "hand_source": args.hand_source,
            "retargeting": args.retargeting,
            "success_standard": "backend_success_and_manual_video_review",
            "recommended_route": args.trajectory_6dof == "egoinfinity",
            "baseline_only": args.trajectory_6dof == "do_as_i_do",
            "retarget_hand_type": args.hand_type,
            "retarget_hand_selection_source": (
                hand_selection.get("selection_source") if hand_selection else None
            ),
            "retarget_hand_selection_reason": (
                hand_selection.get("selection_reason") if hand_selection else None
            ),
        },
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
        # Hand-bearing keypoints are exact-cell assets. Never write them to a
        # trajectory-level last-writer-wins alias that can mix aoe/estimated.
        object_id = object_id_from_task(args.task)
        did_clip = first_existing(
            [
                exp / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction",
                exp / "intermediates" / "do_as_i_do" / "reconstruction",
            ]
        ) or (exp / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction")
        raw_source = first_existing([did_clip / "raw_dir"]) or did_clip
        # Prefer lab-rendered reconstruction mesh overlays over raw Do-as-I-Do
        # TAPIR/debug videos.  raw_dir normally points back to the official clip
        # directory, where output_tapir_*.mp4 exists but does not contain the
        # aligned hand+object mesh overlay expected by the 12-demo triptychs.
        record("trajectory_6dof.raw_dir", link_or_copy(raw_source, traj_assets / "inputs" / "raw_dir", args.mode, exp))
        overlay_candidates = [
            did_clip / "mesh_overlay.mp4",
            did_clip / "overlay.mp4",
            raw_source / "mesh_overlay.mp4",
            raw_source / "overlay.mp4",
            did_clip / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
            raw_source / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
            did_clip / "obj_tracking_out" / object_id / "combined_visualization" / "projected" / "video.mp4",
            raw_source / "obj_tracking_out" / object_id / "combined_visualization" / "projected" / "video.mp4",
            did_clip / f"output_tapir_{object_id}_overlay.mp4",
            raw_source / f"output_tapir_{object_id}_overlay.mp4",
            did_clip / f"output_tapir_{object_id}.mp4",
            raw_source / f"output_tapir_{object_id}.mp4",
            did_clip / "raw.mp4",
            raw_source / "raw.mp4",
        ]
        depth_candidates = [
            did_clip / "depth.mp4",
            raw_source / "depth.mp4",
            did_clip / "moge_depth.mp4",
            raw_source / "moge_depth.mp4",
            did_clip / "pointmap_depth.mp4",
            raw_source / "pointmap_depth.mp4",
            did_clip / "raw.mp4",
            raw_source / "raw.mp4",
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
        record(
            "hand_source.aoe_hands",
            link_or_copy(
                first_existing(
                    aoe_hand_candidates(
                        exp,
                        args.task,
                        args.hand_type,
                        args.retargeting,
                        key,
                    )
                ),
                hand_dir / "aoe_hands.npz",
                args.mode,
                exp,
            ),
        )
    else:
        if args.trajectory_6dof == "do_as_i_do":
            roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
            record("hand_source.estimated_hands", link_or_copy(roots["trajectory_keypoints"], hand_dir / "estimated_hands_and_keypoints.npz", args.mode, exp))
        else:
            roots = egoinfinity_roots(exp)
            record("hand_source.estimated_hands", link_or_copy(first_existing([roots["samples"] / "hand_joints.bin", roots["clip"] / "hand_joints.bin"]), hand_dir / "estimated_hand_joints", args.mode, exp))

    # Retargeting assets.
    ret_dir = ensure_dir(cell_assets / "retargeting" / args.retargeting)
    if args.retargeting == "do_as_i_do":
        roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
        exact_adapter_manifest = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "raw_dir"
            / "adapter_manifest.json"
        )
        exact_processed_keypoints = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "retargeting_outputs"
            / "mano"
            / args.hand_type
            / args.task
            / "0"
            / "trajectory_keypoints.npz"
        )
        preprocess_invariance_path = roots["trajectory_keypoints"].with_name(
            "hoi_preprocess_invariance.json"
        )
        preprocess_invariance = load_json_optional(preprocess_invariance_path)
        raw_to_processed_hoi_path = exact_processed_keypoints.with_name(
            "raw_to_processed_hoi_invariance.json"
        )
        raw_to_processed_hoi = load_json_optional(raw_to_processed_hoi_path)
        alignment_manifest = roots["robot"] / "mjwp_alignment_manifest.json"
        tracking_quality_path = roots["robot"] / "mjwp_object_tracking_quality.json"
        tracking_quality = load_json_optional(tracking_quality_path)
        hand_provenance_path = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "input_hand_provenance.json"
        )
        hand_provenance = load_json_optional(hand_provenance_path)
        manifest["diagnostics"] = {
            "input_hand_provenance": hand_provenance,
            "dai_preprocess_invariance": preprocess_invariance,
            "raw_to_processed_hoi_invariance": raw_to_processed_hoi,
            "mjwp_object_tracking": tracking_quality,
            "advisory_only": True,
        }
        record(
            "retargeting.input_hand_provenance",
            link_or_copy(
                hand_provenance_path if hand_provenance_path.exists() else None,
                ret_dir / "metadata" / "input_hand_provenance.json",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.quality.dai_preprocess_invariance",
            link_or_copy(
                preprocess_invariance_path
                if preprocess_invariance_path.exists()
                else None,
                ret_dir / "metadata" / "dai_preprocess_invariance.json",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.quality.raw_to_processed_hoi_invariance",
            link_or_copy(
                raw_to_processed_hoi_path
                if raw_to_processed_hoi_path.exists()
                else None,
                ret_dir / "metadata" / "raw_to_processed_hoi_invariance.json",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.alignment.mjwp_manifest",
            link_or_copy(
                alignment_manifest if alignment_manifest.exists() else None,
                ret_dir / "metadata" / "mjwp_alignment_manifest.json",
                args.mode,
                exp,
            ),
        )
        record_optional(
            "retargeting.quality.mjwp_object_tracking",
            link_or_copy(
                tracking_quality_path if tracking_quality_path.exists() else None,
                ret_dir / "metadata" / "mjwp_object_tracking_quality.json",
                args.mode,
                exp,
            ),
        )
        for name in ["scene.xml", "scene_ik.xml", "scene_eq.xml"]:
            record(f"retargeting.scene.{name}", link_or_copy(roots["robot"] / name, ret_dir / "scenes" / name, args.mode, exp))
        for name in ["trajectory_kinematic.npz", "trajectory_ikrollout.npz"]:
            record(f"retargeting.trajectory.{name}", link_or_copy(roots["robot"] / name, ret_dir / "trajectories" / name, args.mode, exp))
        record(
            "retargeting.trajectory.config.yaml",
            link_or_copy(
                first_existing([
                    roots["robot"] / "config.yaml",
                    roots["robot"] / "config_act.yaml",
                ]),
                ret_dir / "trajectories" / "config.yaml",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.trajectory.trajectory_mjwp.npz",
            link_or_copy(
            first_existing([
                roots["robot"] / "trajectory_mjwp_act_aligned.npz",
                roots["robot"] / "trajectory_mjwp_aligned.npz",
                ]),
                ret_dir / "trajectories" / "trajectory_mjwp.npz",
                args.mode,
                exp,
            ),
        )
        processed_keypoints, generic_keypoints_alias = dai_indexed_keypoint_paths(
            ret_dir,
            hand_type=args.hand_type,
            task=args.task,
            data_id=0,
        )
        canonical_keypoints_asset = link_or_copy(
            roots["trajectory_keypoints"], processed_keypoints, args.mode, exp
        )
        record_optional(
            "retargeting.trajectory.trajectory_keypoints.route_canonical",
            canonical_keypoints_asset,
        )
        record(
            "retargeting.trajectory.trajectory_keypoints.npz",
            link_or_copy(
                processed_keypoints if processed_keypoints.exists() else None,
                generic_keypoints_alias,
                args.mode,
                exp,
            ),
        )
        processed_object_mesh = dai_indexed_visual_mesh_path(
            ret_dir, task=args.task
        )
        record(
            "retargeting.object_mesh.visual.route_canonical",
            link_or_copy(
                roots["object_visual"] if roots["object_visual"].is_file() else None,
                processed_object_mesh,
                args.mode,
                exp,
            ),
        )
        record_optional("retargeting.video.visualization_ik.mp4", link_or_copy(roots["robot"] / "visualization_ik.mp4", ret_dir / "videos" / "visualization_ik.mp4", args.mode, exp))
        mjwp_plain = first_existing([
            roots["robot"] / "visualization_mjwp_act_aligned.mp4",
            roots["robot"] / "visualization_mjwp_aligned.mp4",
            roots["robot"] / "visualization_mjwp_act.mp4",
            roots["robot"] / "visualization_mjwp.mp4",
        ])
        record("retargeting.video.visualization_mjwp.mp4", link_or_copy(mjwp_plain, ret_dir / "videos" / "visualization_mjwp.mp4", args.mode, exp))
        record_optional("retargeting.video.visualization_mjwp_front.mp4", link_or_copy(mjwp_plain, ret_dir / "videos" / "visualization_mjwp_front.mp4", args.mode, exp))
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
        spider_trajectory = first_existing([
            roots["root"] / "trajectory_mjwp_act_aligned.npz",
            roots["root"] / "trajectory_mjwp_aligned.npz",
            roots["root"] / "trajectory_mjwp.npz",
            roots["robot"] / "trajectory_mjwp.npz",
        ])
        spider_video = first_existing([
            roots["root"] / "visualization_mjwp_act_aligned.mp4",
            roots["root"] / "visualization_mjwp_aligned.mp4",
            roots["root"] / "visualization_mjwp.mp4",
            roots["robot"] / "visualization_mjwp.mp4",
        ])
        spider_scene = first_existing([
            roots["root"] / "scene_act.xml",
            roots["root"] / "scene.xml",
            roots["robot"] / "scene.xml",
        ])
        record(
            "retargeting.trajectory.trajectory_mjwp",
            link_or_copy(
                spider_trajectory,
                ret_dir / "trajectories" / "trajectory_mjwp.npz",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.scene.scene_xml",
            link_or_copy(
                spider_scene,
                ret_dir / "scenes" / "scene.xml",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.video.visualization_mjwp",
            link_or_copy(
                spider_video,
                ret_dir / "videos" / "visualization_mjwp.mp4",
                args.mode,
                exp,
            ),
        )
        robot_video = ret_dir / "videos" / "visualization_mjwp.mp4"

    # Per-cell convenience bundle used by compose scripts.
    overlay = traj_assets / "overlay.mp4"
    depth = traj_assets / "depth.mp4"
    record("cell.overlay", link_or_copy(overlay if overlay.exists() else None, cell_bundle / "overlay.mp4", args.mode, exp))
    record("cell.depth", link_or_copy(depth if depth.exists() else None, cell_bundle / "depth.mp4", args.mode, exp))
    record("cell.robot", link_or_copy(robot_video if robot_video.exists() else None, cell_bundle / "robot.mp4", args.mode, exp))
    robot_probe = probe_video(cell_bundle / "robot.mp4")
    success_errors = [] if robot_probe is not None else ["missing_or_undecodable_robot_video"]
    manifest["backend_result"] = {
        "status": "ok" if not success_errors else "invalid",
        "errors": success_errors,
        "review_video": rel(cell_bundle / "robot.mp4", exp) if robot_probe else None,
        "video_probe": robot_probe,
        "policy": "backend_success_and_manual_video_review",
        "manual_review_required": True,
    }

    manifest_path = cell_assets / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (cell_bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "cell_key": key,
        "manifest": str(manifest_path),
        "missing": manifest["missing"],
        "backend_result": manifest["backend_result"],
    }, indent=2))
    return 0 if not success_errors else 5


if __name__ == "__main__":
    raise SystemExit(main())

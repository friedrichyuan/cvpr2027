from __future__ import annotations

from pathlib import Path


def first_existing(paths: list[str | Path | None]) -> Path | None:
    for item in paths:
        if not item:
            continue
        path = Path(item)
        if path.exists():
            return path
    return None


def cell_artifacts(config: dict, cell_key: str, cell) -> dict:
    art = config.get("artifacts", {})
    method_root = Path(config["method_root"])
    run_dir = method_root / "runs" / config["run_name"] / "cells" / cell_key
    object_id = config.get("scene", {}).get("object_id", "foundation_jar")

    raw_rgb = Path(art["raw_rgb"])
    depth_candidates = []
    robot_candidates = []
    overlay_candidates = []

    if cell.object_pipeline == "egoinfinity":
        ei_run = Path(art.get("egoinfinity_run", ""))
        depth_candidates += [
            ei_run / "retarget_samples" / "depth.mp4",
            ei_run / "retarget_g1" / "input_viz.mp4",
        ]
        robot_candidates += [
            ei_run / "retarget_g1" / "robot_sim.mp4",
        ]
        overlay_candidates += [
            ei_run / "baked_fp_object_overlay.mp4",
            ei_run / "object_overlay.mp4",
        ]
    elif cell.object_pipeline == "do_as_i_do":
        clip = Path(art.get("do_as_i_do_clip", ""))
        retarget_root = Path(
            art.get(
                "do_as_i_do_retargeting_output_root",
                method_root / "data" / "do_as_i_do_retargeting_outputs",
            )
        )
        depth_candidates += [
            clip / "moge_depth.mp4",
            clip / "pointmap_depth.mp4",
            clip / "raw.mp4",
        ]
        overlay_candidates += [
            clip / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
            clip / "obj_tracking_out" / object_id / "combined_visualization" / "projected" / "video.mp4",
        ]
        if cell.retargeter == "do_as_i_do_sharpa":
            task = art.get("do_as_i_do_task", "foundation_jar_bimanual_leftscale")
            robot_candidates += [
                clip / "retarget_render" / "visualization_mjwp.mp4",
                retarget_root / "sharpa" / "bimanual" / task / "0" / "visualization_mjwp.mp4",
            ]

    if cell.retargeter == "egoinfinity_g1":
        ei_run = Path(art.get("egoinfinity_run", ""))
        robot_candidates += [
            ei_run / "retarget_g1" / "robot_sim.mp4",
        ]
    elif cell.retargeter == "spider_mjwp":
        robot_candidates += [
            run_dir / "spider" / "visualization_mjwp.mp4",
            run_dir / "spider" / "visualization_mjwp_rendered.mp4",
        ]

    return {
        "raw_rgb": raw_rgb,
        "overlay": first_existing(overlay_candidates) or raw_rgb,
        "depth": first_existing(depth_candidates),
        "robot": first_existing(robot_candidates),
    }

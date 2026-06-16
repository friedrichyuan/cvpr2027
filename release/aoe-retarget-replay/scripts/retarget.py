#!/usr/bin/env python3
"""AoE-Retarget-Replay: Retarget AoE egocentric data to robot actions.

Usage:
    # Single episode
    python scripts/retarget.py --episode_dir /path/to/segment --robot g1_inspire

    # Batch (all episodes under data root)
    python scripts/retarget.py --data_root /path/to/poc_deliver --robot g1_inspire

    # With visualization
    python scripts/retarget.py --episode_dir /path/to/segment --robot g1_inspire --visualize
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AoE → G1 retargeting pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode_dir", type=Path, help="Single episode directory")
    group.add_argument("--data_root", type=Path, help="Root dir containing episodes")

    parser.add_argument("--robot", choices=["g1_inspire", "g1_dex3"],
                        default="g1_inspire", help="Robot variant")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory (default: ./output/<robot>)")
    parser.add_argument("--no_refine_shoulder", action="store_true",
                        help="Disable per-episode shoulder refinement")
    parser.add_argument("--frame", choices=["cam", "world"], default="cam",
                        help="Coordinate frame for retargeting (default: cam)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to process (batch mode)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip episodes with existing output")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate 2x3 visualization video")
    parser.add_argument("--stage3", action="store_true",
                        help="Enable Stage 3 (SAM2 mask + E2FGVI inpaint + compose)")
    parser.add_argument("--write_parquet", action="store_true",
                        help="Write LeRobot v2.1 parquet output")
    args = parser.parse_args()

    from aoe_retarget_replay.pipeline import (
        RetargetSession,
        discover_episodes,
    )
    from aoe_retarget_replay.robots import get_spec

    spec = get_spec(args.robot)
    output_dir = args.output_dir or Path(f"./output/{args.robot}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.episode_dir:
        episodes = [args.episode_dir]
    else:
        episodes = discover_episodes(args.data_root)
        if args.max_episodes:
            episodes = episodes[:args.max_episodes]

    session = RetargetSession(spec)
    episodes_info = []

    for ep_idx, ep_dir in enumerate(episodes):
        seg_name = ep_dir.name
        npy_path = output_dir / f"{seg_name}_actions.npy"

        if args.skip_existing and npy_path.exists():
            logger.info("Skipping existing: %s", seg_name)
            continue

        logger.info("[%d/%d] Processing: %s", ep_idx + 1, len(episodes), seg_name)

        try:
            result = session.retarget(
                ep_dir,
                refine_shoulder=not args.no_refine_shoulder,
                frame=args.frame,
            )
        except Exception as e:
            logger.error("Failed on %s: %s", seg_name, e)
            continue

        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, result.actions)
        logger.info("  → %s (T=%d, scale=%.3f)",
                    npy_path, result.actions.shape[0], result.scale)

        if args.write_parquet:
            from aoe_retarget_replay.io import write_episode_parquet
            ep_info = write_episode_parquet(
                episode_index=ep_idx,
                actions=result.actions,
                output_dir=output_dir,
                task_description=result.task_description,
            )
            episodes_info.append(ep_info)

        if args.visualize:
            from aoe_retarget_replay.retarget.visualize import render_2x3_video
            ego_video = _find_ego_video(ep_dir)
            if ego_video:
                viz_path = output_dir / f"{seg_name}_2x3.mp4"
                render_2x3_video(
                    result.actions,
                    ego_video,
                    viz_path,
                    spec=spec,
                    use_stage3=args.stage3,
                )
            else:
                from aoe_retarget_replay.retarget.visualize import render_trajectory_streaming
                viz_path = output_dir / f"{seg_name}_robot.mp4"
                render_trajectory_streaming(result.actions, viz_path, spec=spec)

    if args.write_parquet and episodes_info:
        from aoe_retarget_replay.io import write_metadata
        write_metadata(
            output_dir=output_dir,
            dataset_name=f"aoe_{args.robot}",
            episodes_info=episodes_info,
            canonical_description="AoE egocentric manipulation",
            robot_type=spec.robot_type,
            action_dim=spec.action_dim,
            action_joint_names=spec.action_joint_names,
            modality=spec.modality,
        )

    logger.info("Done. Output in: %s", output_dir)


def _find_ego_video(episode_dir: Path) -> Path | None:
    """Find the undistorted ego video in a poc_deliver episode."""
    for subdir in ("ego_process/ego_undistorted_video",
                   "ego_undistorted_video"):
        vdir = episode_dir / subdir
        if vdir.exists():
            for f in vdir.iterdir():
                if f.suffix == ".mp4":
                    return f
    raw = episode_dir / "raw_video.mp4"
    if raw.exists():
        return raw
    return None


if __name__ == "__main__":
    import numpy as np
    main()

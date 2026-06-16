#!/usr/bin/env python3
"""AoE-Retarget-Replay: Visualize retargeted robot trajectories.

Generates a 2x3 video layout:
  Row 1: [ego video | MuJoCo ext | MuJoCo front]
  Row 2: [SAM2 mask | E2FGVI inpaint | composed robot]

Usage:
    MUJOCO_GL=egl python scripts/visualize.py \
        --episode_dir /path/to/segment \
        --robot g1_inspire \
        --output viz_2x3.mp4

    # With Stage 3 (SAM2 + E2FGVI + compose):
    MUJOCO_GL=egl python scripts/visualize.py \
        --episode_dir /path/to/segment \
        --robot g1_inspire \
        --stage3 \
        --output viz_2x3_full.mp4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize retargeted trajectory")
    parser.add_argument("--episode_dir", type=Path, required=True,
                        help="Episode directory (for ego video)")
    parser.add_argument("--actions", type=Path, default=None,
                        help="Actions .npy file (if not provided, runs retarget)")
    parser.add_argument("--robot", choices=["g1_inspire", "g1_dex3"],
                        default="g1_inspire")
    parser.add_argument("--output", type=Path, default=Path("viz_2x3.mp4"))
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit frames for quick preview")
    parser.add_argument("--stage3", action="store_true",
                        help="Enable Stage 3 (SAM2 mask + E2FGVI inpaint + compose)")
    args = parser.parse_args()

    from aoe_retarget_replay.robots import get_spec

    spec = get_spec(args.robot)

    if args.actions and args.actions.exists():
        actions = np.load(args.actions)
    else:
        logger.info("No actions file provided, running retarget...")
        from aoe_retarget_replay.pipeline import RetargetSession
        session = RetargetSession(spec)
        result = session.retarget(args.episode_dir)
        actions = result.actions

    if args.max_frames:
        actions = actions[:args.max_frames]

    ego_video = _find_ego_video(args.episode_dir)
    if ego_video is None:
        logger.warning("No ego video found, rendering robot-only")
        from aoe_retarget_replay.retarget.visualize import render_trajectory_streaming
        render_trajectory_streaming(actions, args.output, spec=spec)
    else:
        from aoe_retarget_replay.retarget.visualize import render_2x3_video
        render_2x3_video(
            actions, ego_video, args.output, spec=spec,
            use_stage3=args.stage3,
        )

    logger.info("Output: %s", args.output)


def _find_ego_video(episode_dir: Path) -> Path | None:
    for subdir in ("ego_process/ego_undistorted_video", "ego_undistorted_video"):
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
    main()

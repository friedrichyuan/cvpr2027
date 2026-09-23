from __future__ import annotations

import json

import numpy as np

from egodex_arx_replay.data import load_episode
from egodex_arx_replay.defaults import EGODEX_FPS
from egodex_arx_replay.smoothing import SmoothingConfig, smooth_gripper_trajectory

from ..geom.camera_frame import convert_episode_to_camera_grippers
from ..processor import CPU, Artifacts, EpisodeContext, ResourceSpec


class ActionAlignProcessor:
    name = "action_align"
    requires: frozenset[str] = frozenset()
    produces = frozenset({Artifacts.SOURCE, Artifacts.ACTION})
    optional_requires: frozenset[str] = frozenset()
    resources: ResourceSpec = CPU

    def run(self, ctx: EpisodeContext) -> None:
        episode = load_episode(ctx.hdf5_path)
        raw = convert_episode_to_camera_grippers(episode)
        smoothed = smooth_gripper_trajectory(raw, SmoothingConfig())
        source_path = ctx.out_dir / Artifacts.SOURCE
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            json.dumps(
                {
                    "hdf5": str(episode.path),
                    "video": str(ctx.video_path) if ctx.video_path else None,
                    "fps": EGODEX_FPS,
                    "frames": episode.frame_count,
                    "task": episode.metadata.get("task", ctx.task),
                    "instruction": episode.metadata.get("llm_description", ""),
                    "intrinsic": episode.intrinsic.tolist(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        action_path = ctx.out_dir / Artifacts.ACTION
        action_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            action_path,
            position=smoothed.position.astype(np.float32),
            rotation=smoothed.rotation.astype(np.float32),
            width=smoothed.width.astype(np.float32),
            valid=smoothed.valid,
            raw_position=raw.position.astype(np.float32),
            raw_rotation=raw.rotation.astype(np.float32),
            raw_width=raw.width.astype(np.float32),
            intrinsic=episode.intrinsic.astype(np.float32),
            world_T_camera=episode.world_T_camera.astype(np.float32),
        )

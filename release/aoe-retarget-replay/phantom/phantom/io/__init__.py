"""LeRobot v2.1 dataset writer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from phantom.constants import (
    CHUNK_SIZE,
    FPS,
    VIDEO_CODEC,
    VIDEO_HEIGHT,
    VIDEO_KEY,
    VIDEO_PIX_FMT,
    VIDEO_WIDTH,
)

logger = logging.getLogger(__name__)


def write_episode_parquet(
    episode_index: int,
    actions: np.ndarray,
    output_dir: Path,
    task_description: str = "",
) -> dict:
    """Write a (T, action_dim) action trajectory as a LeRobot parquet file."""
    n_frames = actions.shape[0]
    chunk_idx = episode_index // CHUNK_SIZE
    chunk_dir = f"chunk-{chunk_idx:03d}"

    states = actions.copy()
    timestamps = np.arange(n_frames, dtype=np.float32) / FPS
    frame_indices = np.arange(n_frames, dtype=np.int64)
    episode_indices = np.full(n_frames, episode_index, dtype=np.int64)
    task_indices = np.zeros(n_frames, dtype=np.int64)
    annotation_indices = np.zeros(n_frames, dtype=np.int64)

    data = {
        "observation.state": [row.tolist() for row in states],
        "action": [row.tolist() for row in actions],
        "timestamp": timestamps.tolist(),
        "annotation.human.action.task_description": annotation_indices.tolist(),
        "frame_index": frame_indices.tolist(),
        "episode_index": episode_indices.tolist(),
        "index": frame_indices.tolist(),
        "task_index": task_indices.tolist(),
    }

    df = pd.DataFrame(data)
    parquet_dir = output_dir / "data" / chunk_dir
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"episode_{episode_index:06d}.parquet"
    df.to_parquet(parquet_path, index=False)

    stats = {
        "observation.state": {
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
        },
        "action": {
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
        },
    }

    return {
        "episode_index": episode_index,
        "length": n_frames,
        "task_description": task_description,
        "stats": stats,
    }


def write_metadata(
    output_dir: Path,
    dataset_name: str,
    episodes_info: list[dict],
    canonical_description: str,
    *,
    robot_type: str,
    action_dim: int,
    action_joint_names: list[str],
    modality: dict,
    fps: int = FPS,
) -> None:
    """Write the LeRobot v2.1 meta/ directory."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = len(episodes_info)
    total_frames = sum(ep["length"] for ep in episodes_info)
    n_chunks = max(1, (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE)

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes,
        "total_chunks": n_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [VIDEO_HEIGHT, VIDEO_WIDTH, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": VIDEO_HEIGHT,
                    "video.width": VIDEO_WIDTH,
                    "video.codec": VIDEO_CODEC,
                    "video.pix_fmt": VIDEO_PIX_FMT,
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": {"axes": action_joint_names},
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": {"axes": action_joint_names},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": canonical_description}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_info:
            f.write(json.dumps({
                "episode_index": ep["episode_index"],
                "tasks": [ep["task_description"]],
                "length": ep["length"],
            }) + "\n")

    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    logger.info("Wrote metadata for '%s': %d episodes, %d frames",
                dataset_name, total_episodes, total_frames)

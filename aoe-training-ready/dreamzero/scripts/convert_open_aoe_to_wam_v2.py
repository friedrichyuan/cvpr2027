#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path("/PATH_TO/general_datasets/Open-AoE/poc_deliver")
DEFAULT_OUTPUT_ROOT = Path("/PATH_TO/dataset/Open_AoE/open_aoe_mano_nextstate_224_110d_wam_v2")
FALLBACK_TASK = "egocentric hand manipulation"
HAND_POSE_DIMS = 45
HAND_VECTOR_DIMS = 1 + 3 + 3 + 3 + HAND_POSE_DIMS
STATE_DIMS = 2 * HAND_VECTOR_DIMS
ACTION_DIMS = 2 * HAND_VECTOR_DIMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-frames-per-episode", type=int, default=360)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def resize_letterbox_rgb(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return np.ascontiguousarray(canvas)


def format_action_task(scene: str, action: dict) -> str:
    verb = str(action.get("verb") or "manipulate").strip().replace("_", " ")
    obj = str(action.get("object") or "object").strip().replace("_", " ")
    hand = str(action.get("hand") or "hand").strip().replace("_", " ")
    scene = str(scene or "indoor").strip().replace("_", " ")
    return f"In {scene}, use {hand} hand to {verb} {obj}."


def load_annotation_tasks(sample_dir: Path, num_frames: int) -> list[tuple[int, int, str]]:
    ann_path = sample_dir / "ego_annotation" / "ego_action_annotation.json"
    if not ann_path.exists():
        return []
    try:
        raw_segments = read_json(ann_path)
    except Exception:
        return []
    segments = []
    for seg in raw_segments if isinstance(raw_segments, list) else []:
        if not isinstance(seg, dict):
            continue
        start = safe_int(seg.get("start_frame"), 0)
        end = safe_int(seg.get("end_frame"), start)
        if end <= start:
            continue
        start = max(0, min(start, num_frames))
        end = max(0, min(end, num_frames))
        if end <= start:
            continue
        actions = seg.get("atomic_action") or []
        if isinstance(actions, list) and actions and isinstance(actions[0], dict):
            task = format_action_task(str(seg.get("scene") or "indoor"), actions[0])
        else:
            task = FALLBACK_TASK
        segments.append((start, end, task))
    return sorted(segments, key=lambda item: item[0])


def task_for_frame(segments: list[tuple[int, int, str]], frame_index: int, cursor: int) -> tuple[str, int]:
    while cursor + 1 < len(segments) and frame_index >= segments[cursor][1]:
        cursor += 1
    if segments and segments[cursor][0] <= frame_index < segments[cursor][1]:
        return segments[cursor][2], cursor
    return FALLBACK_TASK, cursor


def source_fps(sample_dir: Path) -> float:
    info_path = sample_dir / "ego_process" / "ego_undistorted_video" / "undistorted_video_info.json"
    if not info_path.exists():
        return 30.0
    return safe_float(read_json(info_path).get("fps"), 30.0)


def hand_vectors(hands: np.lib.npyio.NpzFile, fps: float) -> np.ndarray:
    valid = np.asarray(hands["pred_valid"], dtype=np.float32) > 0.5
    trans = np.asarray(hands["pred_trans_cam"], dtype=np.float32)
    rot = np.asarray(hands["pred_rot_cam"], dtype=np.float32)
    pose = np.asarray(hands["pred_hand_pose"], dtype=np.float32)[..., :HAND_POSE_DIMS]
    velocity = np.zeros_like(trans, dtype=np.float32)
    velocity[:, 1:] = (trans[:, 1:] - trans[:, :-1]) * float(fps)
    velocity[:, 1:] *= (valid[:, 1:] & valid[:, :-1])[..., None]

    out = np.zeros((2, valid.shape[1], HAND_VECTOR_DIMS), dtype=np.float32)
    for hand_idx in range(2):
        mask = valid[hand_idx]
        out[hand_idx, :, 0] = mask.astype(np.float32)
        out[hand_idx, mask, 1:4] = trans[hand_idx, mask]
        out[hand_idx, mask, 4:7] = rot[hand_idx, mask]
        out[hand_idx, mask, 7:10] = velocity[hand_idx, mask]
        out[hand_idx, mask, 10:] = pose[hand_idx, mask]
    return out


def state_from_vectors(vectors: np.ndarray, frame_index: int) -> np.ndarray:
    state = np.zeros((STATE_DIMS,), dtype=np.float32)
    state[:HAND_VECTOR_DIMS] = vectors[0, frame_index]
    state[HAND_VECTOR_DIMS : 2 * HAND_VECTOR_DIMS] = vectors[1, frame_index]
    return state


def action_from_vectors(vectors: np.ndarray, frame_index: int) -> np.ndarray:
    action = np.zeros((ACTION_DIMS,), dtype=np.float32)
    action[:HAND_VECTOR_DIMS] = vectors[0, frame_index]
    action[HAND_VECTOR_DIMS:] = vectors[1, frame_index]
    return action


def feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def episode_stats(episode_index: int, rows: list[dict]) -> dict:
    return {
        "episode_index": episode_index,
        "stats": {
            "observation.state": feature_stats(
                np.stack([row["observation.state"] for row in rows], axis=0)
            ),
            "action": feature_stats(np.stack([row["action"] for row in rows], axis=0)),
            "timestamp": feature_stats(np.asarray([row["timestamp"] for row in rows], dtype=np.float32)),
        },
    }


def convert_episode(
    sample_dir: Path,
    output_root: Path,
    episode_index: int,
    global_index_start: int,
    fps: int,
    image_size: int,
    max_frames: int,
) -> tuple[int, str]:
    hands_path = sample_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    video_path = sample_dir / "ego_process" / "ego_undistorted_video" / "raw_video_undistorted.mp4"
    with np.load(hands_path, mmap_mode="r") as hands:
        num_frames = int(hands["pred_valid"].shape[1])
        vectors = hand_vectors(hands, fps=source_fps(sample_dir))

    frames_to_write = min(num_frames, max_frames)
    segments = load_annotation_tasks(sample_dir, frames_to_write)
    task_cursor = 0
    rows = []

    chunk_dir = output_root / "videos" / "observation.images.front" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_video = chunk_dir / f"episode_{episode_index:06d}.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (image_size, image_size),
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    try:
        for local_idx in range(frames_to_write):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = resize_letterbox_rgb(bgr, image_size)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            target_idx = min(local_idx + 1, num_frames - 1)
            task, task_cursor = task_for_frame(segments, local_idx, task_cursor)
            rows.append(
                {
                    "observation.state": state_from_vectors(vectors, local_idx),
                    "action": action_from_vectors(vectors, target_idx),
                    "timestamp": float(local_idx) / float(fps),
                    "frame_index": local_idx,
                    "episode_index": episode_index,
                    "index": global_index_start + local_idx,
                    "task": task,
                }
            )
    finally:
        cap.release()
        writer.release()

    data_dir = output_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_dir / f"episode_{episode_index:06d}.parquet", index=False)
    episode_task = rows[0]["task"] if rows else FALLBACK_TASK
    return len(rows), episode_task, episode_stats(episode_index, rows)


def write_info(output_root: Path, fps: int, total_episodes: int, total_frames: int, image_size: int) -> None:
    meta = output_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "open_aoe_egocentric_mano",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 0,
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.front": {
                "dtype": "video",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIMS],
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIMS],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task": {"dtype": "string", "shape": [1]},
        },
    }
    with (meta / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main() -> int:
    args = parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if args.limit is not None:
        samples = samples[: args.limit]
    total_frames = 0
    episodes = []
    stats = []
    for episode_index, sample_dir in enumerate(samples):
        n_frames, task, ep_stats = convert_episode(
            sample_dir=sample_dir,
            output_root=args.output_root,
            episode_index=episode_index,
            global_index_start=total_frames,
            fps=args.fps,
            image_size=args.image_size,
            max_frames=args.max_frames_per_episode,
        )
        total_frames += n_frames
        episodes.append({"episode_index": episode_index, "tasks": [task], "length": n_frames})
        stats.append(ep_stats)
        print(f"[{episode_index + 1}/{len(samples)}] {sample_dir.name} frames={n_frames}")

    write_info(args.output_root, args.fps, len(episodes), total_frames, args.image_size)
    meta = args.output_root / "meta"
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in episodes:
            ep["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": int(ep["length"]),
                    "action_text": ep["tasks"][0] if ep["tasks"] else FALLBACK_TASK,
                }
            ]
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    with (meta / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for ep_stats in stats:
            f.write(json.dumps(ep_stats, ensure_ascii=False) + "\n")
    unique_tasks = {}
    for ep in episodes:
        for task in ep["tasks"]:
            unique_tasks.setdefault(task, len(unique_tasks))
    with (meta / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task, idx in unique_tasks.items():
            f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")
    print(f"[done] wrote {args.output_root} episodes={len(episodes)} frames={total_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

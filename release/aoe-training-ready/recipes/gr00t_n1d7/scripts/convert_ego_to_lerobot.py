# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Convert AoE Egocentric dataset (poc_deliver) → LeRobot V2 format for GR00T N1.7.

Two hand representation modes (EEF-first layout, matching NVIDIA GR00T convention):
  sharpa:  62D = [left_wrist_eef(9), right_wrist_eef(9), left_hand_joints(22), right_hand_joints(22)]
  gripper: 20D = [left_wrist_eef(9), right_wrist_eef(9), left_gripper(1), right_gripper(1)]

Input: poc_deliver/ directory with episode folders containing:
    - ego_process/ego_hands_reconstruction/hands.npz
    - ego_process/ego_undistorted_video/raw_video_undistorted.mp4
    - ego_annotation/ego_action_annotation.json

Output: LeRobot V2 dataset directory
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mano_to_eef import (
    process_hands_npz_gripper,
    process_hands_npz_sharpa,
    split_by_validity,
)
from utils import discover_episodes

# Import canonical mode definitions from the single source of truth
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
from human_ego_config import MODE_CONFIGS as _CANONICAL_MODE_CONFIGS


def _build_modality_groups(mode: str) -> dict:
    """Derive modality_groups with dim/slice from the canonical MODE_CONFIGS."""
    cfg = _CANONICAL_MODE_CONFIGS[mode]
    keys = cfg["modality_keys"]
    # Dimension of each modality key: EEF is 9, sharpa joints are 22, gripper is 1
    key_dims = {
        "left_wrist_eef": 9, "right_wrist_eef": 9,
        "left_gripper": 1, "right_gripper": 1,
        "left_sharpa_joints": 22, "right_sharpa_joints": 22,
        "left_hand_joints": 22, "right_hand_joints": 22,
    }
    offset = 0
    groups = {}
    for key in keys:
        dim = key_dims[key]
        groups[key] = {"dim": dim, "slice": (offset, offset + dim)}
        offset += dim
    return groups


MODE_CONFIGS = {}
for _mode, _cfg in _CANONICAL_MODE_CONFIGS.items():
    MODE_CONFIGS[_mode] = {
        "state_dim": _cfg["state_dim"],
        "robot_type": _cfg["embodiment_tag"],
        "modality_keys": _cfg["modality_keys"],
        "modality_groups": _build_modality_groups(_mode),
    }


def load_annotations(annotation_path: str | None) -> list[dict]:
    if annotation_path is None or not os.path.exists(annotation_path):
        return []
    with open(annotation_path, "r") as f:
        return json.load(f)


def get_task_description(annotations: list[dict], frame_idx: int, fps: int) -> str:
    timestamp = frame_idx / fps

    for ann in annotations:
        start_ts = float(ann.get("start_ts", 0))
        end_ts = float(ann.get("end_ts", 0))
        if start_ts <= timestamp <= end_ts:
            actions = ann.get("atomic_action", [])
            if actions:
                desc = actions[0].get("description", "")
                if desc:
                    return desc
                verb = actions[0].get("verb", "manipulate")
                obj = actions[0].get("object", "object")
                return f"{verb} {obj}"

    return "perform bimanual hand manipulation"


def _build_state(result: dict, modality_keys: list) -> np.ndarray:
    """Build full state by concatenating modality arrays in order."""
    return np.concatenate([result[k] for k in modality_keys], axis=1)


def convert_episode(
    episode: dict,
    mode: str = "gripper",
    action_horizon: int = 16,
    target_fps: int = 15,
    source_fps: int = 30,
    min_segment_len: int = 30,
    retarget_dir: str | None = None,
) -> list[dict]:
    """Convert a single episode. Returns list of sub-episode metadata."""
    cfg = MODE_CONFIGS[mode]
    state_dim = cfg["state_dim"]

    if mode == "sharpa":
        if retarget_dir:
            retarget_path = str(Path(retarget_dir) / episode["name"] / "hands_retargeted_sharpa.npz")
            if not os.path.exists(retarget_path):
                retarget_path = str(Path(episode["hands_npz"]).parent / "hands_retargeted_sharpa.npz")
        else:
            retarget_path = str(Path(episode["hands_npz"]).parent / "hands_retargeted_sharpa.npz")
        if not os.path.exists(retarget_path):
            return []
        result = process_hands_npz_sharpa(
            episode["hands_npz"], retarget_path,
            target_fps=target_fps, source_fps=source_fps,
        )
        full_state = _build_state(result, cfg["modality_keys"])
    else:  # gripper
        result = process_hands_npz_gripper(
            episode["hands_npz"], target_fps=target_fps, source_fps=source_fps,
        )
        full_state = _build_state(result, cfg["modality_keys"])

    segments = split_by_validity(
        result["valid_mask"], min_segment_len=min_segment_len, state=full_state,
    )

    if not segments:
        return []

    annotations = load_annotations(episode["annotation"])
    actual_fps = result["fps"]
    step = max(1, source_fps // target_fps)

    sub_episodes = []
    for seg_idx, seg in enumerate(segments):
        state = seg["state"]
        N = len(state)

        if N < action_horizon + 1:
            continue

        # Build action chunks
        num_valid_steps = N - action_horizon
        actions = np.zeros((N, action_horizon * state_dim), dtype=np.float32)

        for t in range(num_valid_steps):
            actions[t] = state[t + 1: t + 1 + action_horizon].flatten()

        for t in range(num_valid_steps, N):
            remaining = min(action_horizon, N - t - 1)
            if remaining > 0:
                chunk = state[t + 1: t + 1 + remaining]
                pad = np.tile(state[N - 1], (action_horizon - remaining, 1))
                full_chunk = np.concatenate([chunk, pad], axis=0)
            else:
                full_chunk = np.tile(state[N - 1], (action_horizon, 1))
            actions[t] = full_chunk.flatten()

        orig_start_frame = seg["start_idx"] * step
        task_desc = get_task_description(annotations, orig_start_frame, source_fps)
        timestamps = np.arange(N, dtype=np.float64) / actual_fps

        sub_episodes.append({
            "state": state.astype(np.float32),
            "action": actions,
            "timestamps": timestamps,
            "task_description": task_desc,
            "source_episode": episode["name"],
            "segment_idx": seg_idx,
            "num_frames": N,
            "video_path": episode["video_path"],
            "orig_start_frame": orig_start_frame,
            "orig_end_frame": seg["end_idx"] * step,
        })

    return sub_episodes


def write_lerobot_dataset(
    all_episodes: list[dict],
    output_dir: str,
    mode: str = "gripper",
    action_horizon: int = 16,
    target_fps: int = 15,
    link_videos: bool = True,
):
    cfg = MODE_CONFIGS[mode]
    state_dim = cfg["state_dim"]

    output_dir = Path(output_dir)
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000" / "observation.images.ego_view"
    meta_dir = output_dir / "meta"

    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    task_to_idx = {}
    episodes_meta = []
    global_index = 0

    for ep_idx, ep_data in enumerate(all_episodes):
        N = ep_data["num_frames"]
        task_desc = ep_data["task_description"]

        if task_desc not in task_to_idx:
            task_to_idx[task_desc] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": task_desc})

        task_idx = task_to_idx[task_desc]

        rows = []
        # modality group slices for per-key columns (state.{key}, action.{key})
        groups = cfg["modality_groups"]
        for t in range(N):
            row = {
                "observation.state": ep_data["state"][t].tolist(),
                "action": ep_data["action"][t].tolist(),
                "timestamp": ep_data["timestamps"][t],
                "episode_index": ep_idx,
                "index": global_index + t,
                "task_index": task_idx,
            }
            # Per-key sub-columns required by GR00T stats/loader
            state_vec = ep_data["state"][t]
            action_vec = ep_data["action"][t]
            for name, info in groups.items():
                s = info["slice"]
                row[f"state.{name}"] = state_vec[s[0]:s[1]].tolist()
                row[f"action.{name}"] = action_vec[s[0]:s[1]].tolist()
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        video_link = video_dir / f"episode_{ep_idx:06d}.mp4"
        if link_videos:
            src = Path(ep_data["video_path"]).resolve()
            if video_link.exists() or video_link.is_symlink():
                video_link.unlink()
            video_link.symlink_to(src)
        else:
            shutil.copy2(ep_data["video_path"], video_link)

        episodes_meta.append({
            "episode_index": ep_idx,
            "task_index": task_idx,
            "length": N,
            "source": ep_data["source_episode"],
            "segment_idx": ep_data["segment_idx"],
            "orig_start_frame": ep_data["orig_start_frame"],
            "orig_end_frame": ep_data["orig_end_frame"],
        })

        global_index += N

    action_dim = action_horizon * state_dim

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": cfg["robot_type"],
        "mode": mode,
        "total_episodes": len(all_episodes),
        "total_frames": global_index,
        "total_videos": len(all_episodes),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": target_fps,
        "action_horizon": action_horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "bimanual": True,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
            "observation.images.ego_view": {"dtype": "video", "shape": [720, 1280, 3]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    modality = build_modality_json(all_episodes, cfg)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    return {"total_episodes": len(all_episodes), "total_frames": global_index, "num_tasks": len(tasks)}


def build_modality_json(all_episodes: list[dict], cfg: dict) -> dict:
    groups = cfg["modality_groups"]

    group_data = {name: [] for name in groups}
    for ep in all_episodes:
        state = ep["state"]
        for name, info in groups.items():
            s = info["slice"]
            group_data[name].append(state[:, s[0]:s[1]])

    def stats(arr):
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }

    modality = {"state": {}, "action": {}, "annotation": {"human.task_description": {"original_key": "task_index"}}}
    for name, info in groups.items():
        arr = np.concatenate(group_data[name], axis=0)
        s = info["slice"]
        entry = {
            "dim": info["dim"],
            "start": s[0],
            "end": s[1],
            "original_key": "observation.state",
            **stats(arr),
        }
        modality["state"][name] = entry
        # action shares the same slice/keys
        entry_action = dict(entry)
        entry_action["original_key"] = "action"
        modality["action"][name] = entry_action

    return modality


def main():
    parser = argparse.ArgumentParser(
        description="Convert AoE ego dataset to LeRobot V2 format for GR00T N1.7"
    )
    parser.add_argument(
        "--mode", type=str, default="sharpa", choices=["gripper", "sharpa"],
        help="sharpa: 62D (EEF + 22-DoF Sharpa); gripper: 20D (open/close)",
    )
    parser.add_argument(
        "--data-root", type=str,
        default="/media/hdd4tb/sankuai/code/07_具身/dataset/07_Ego/poc_deliver",
    )
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Base output dir. Dataset written to <output-dir>/<episode>/ego_<mode>_lerobotv21/")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--min-segment-len", type=int, default=30)
    parser.add_argument("--copy-videos", action="store_true")
    parser.add_argument(
        "--retarget-dir", type=str, default=None,
        help="Directory containing hands_retargeted.npz from Step 1. "
             "If set, reads from <retarget-dir>/<episode>/hands_retargeted.npz first, "
             "falls back to source directory. If not set, reads from source directory.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)

    args = parser.parse_args()
    cfg = MODE_CONFIGS[args.mode]

    print(f"=== AoE Ego → LeRobot V2 ({args.mode.upper()}) ===")
    print(f"  Mode: {args.mode}")
    print(f"  State dim: {cfg['state_dim']}D")
    print(f"  Data root: {args.data_root}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Action horizon: {args.action_horizon}")
    print(f"  Target FPS: {args.target_fps}")
    if args.retarget_dir:
        print(f"  Retarget dir: {args.retarget_dir}")
    print()

    episodes = discover_episodes(args.data_root)
    print(f"Found {len(episodes)} episodes")

    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
        print(f"  (limited to {args.max_episodes})")

    total_sub = 0
    total_frames = 0
    failed = []

    for i, ep in enumerate(episodes):
        try:
            subs = convert_episode(
                ep, mode=args.mode, action_horizon=args.action_horizon,
                target_fps=args.target_fps, source_fps=args.source_fps,
                min_segment_len=args.min_segment_len,
                retarget_dir=args.retarget_dir,
            )
            if not subs:
                print(f"  [{i+1}/{len(episodes)}] {ep['name']}: SKIPPED")
                continue

            ep_output_dir = str(Path(args.output_dir) / ep["name"] / f"ego_{args.mode}_lerobotv21")
            stats = write_lerobot_dataset(
                subs, ep_output_dir, mode=args.mode,
                action_horizon=args.action_horizon, target_fps=args.target_fps,
                link_videos=not args.copy_videos,
            )
            n_frames = sum(s['num_frames'] for s in subs)
            total_sub += len(subs)
            total_frames += n_frames
            print(f"  [{i+1}/{len(episodes)}] {ep['name']}: "
                  f"{len(subs)} seg, {n_frames} frames → {ep_output_dir}")
        except Exception as e:
            failed.append((ep["name"], str(e)))
            print(f"  [{i+1}/{len(episodes)}] {ep['name']}: FAILED ({e})")

    print(f"\n=== Summary ===")
    print(f"  Mode: {args.mode}")
    print(f"  Episodes: {total_sub} segments from {len(episodes)} sources")
    print(f"  Total frames: {total_frames}")
    print(f"  State: {cfg['state_dim']}D | Action: {args.action_horizon}×{cfg['state_dim']}D = {args.action_horizon * cfg['state_dim']}D")
    if failed:
        print(f"  Failed: {len(failed)}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Merge multiple per-episode LeRobot V2.1 datasets into a single training dataset.

Each episode dir has structure:
    <ep>/ego_<mode>_lerobotv21/
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.ego_view/episode_000000.mp4
        meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}

Output (merged):
    <output>/
        data/chunk-000/episode_000000.parquet ... episode_0000XX.parquet
        videos/chunk-000/observation.images.ego_view/episode_*.mp4
        meta/{info, episodes, tasks, modality}.json(l)

Usage:
    python merge_lerobot_datasets.py \
        --input-dir ../output \
        --mode sharpa \
        --output-dir ../output/ego_sharpa_merged
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def merge(input_dir: Path, mode: str, output_dir: Path):
    suffix = f"ego_{mode}_lerobotv21"
    episode_dirs = sorted([d for d in input_dir.iterdir()
                           if d.is_dir() and (d / suffix).is_dir()])

    if not episode_dirs:
        print(f"ERROR: No '{suffix}' dirs found in {input_dir}")
        return

    print(f"=== Merge LeRobot datasets ({mode}) ===")
    print(f"  Found {len(episode_dirs)} episode datasets")

    out_data = output_dir / "data" / "chunk-000"
    out_video = output_dir / "videos" / "chunk-000" / "observation.images.ego_view"
    out_meta = output_dir / "meta"
    for d in (out_data, out_video, out_meta):
        d.mkdir(parents=True, exist_ok=True)

    global_ep_idx = 0
    global_frame_idx = 0
    tasks = []           # list of {task_index, task}
    task_to_idx = {}     # task_string -> global task_index
    episodes_meta = []
    group_arrays = {}    # modality_key -> list of (T, dim) arrays
    info_template = None

    for ep_dir in episode_dirs:
        ds_dir = ep_dir / suffix
        pq_path = ds_dir / "data" / "chunk-000" / "episode_000000.parquet"
        if not pq_path.exists():
            print(f"  SKIP {ep_dir.name}: no parquet")
            continue

        df = pd.read_parquet(pq_path)
        N = len(df)

        # Remap task_index
        old_task = int(df["task_index"].iloc[0])
        with open(ds_dir / "meta" / "tasks.jsonl") as f:
            ep_tasks = [json.loads(line) for line in f]
        task_str = ep_tasks[old_task]["task"] if old_task < len(ep_tasks) else "unknown"
        if task_str not in task_to_idx:
            task_to_idx[task_str] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": task_str})
        new_task_idx = task_to_idx[task_str]

        # Remap episode_index and global index
        df["episode_index"] = global_ep_idx
        df["index"] = np.arange(global_frame_idx, global_frame_idx + N)
        df["task_index"] = new_task_idx

        df.to_parquet(out_data / f"episode_{global_ep_idx:06d}.parquet", index=False)

        # Copy/symlink video
        src_video = ds_dir / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
        dst_video = out_video / f"episode_{global_ep_idx:06d}.mp4"
        if src_video.exists():
            if dst_video.exists() or dst_video.is_symlink():
                dst_video.unlink()
            # Resolve symlink target then create new symlink to absolute source
            src_resolved = src_video.resolve()
            dst_video.symlink_to(src_resolved)
        else:
            shutil.copy2(src_video, dst_video) if src_video.exists() else None

        # Read episode meta for length info
        with open(ds_dir / "meta" / "episodes.jsonl") as f:
            ep_meta = json.loads(f.readline())
        episodes_meta.append({
            "episode_index": global_ep_idx,
            "task_index": new_task_idx,
            "length": N,
            "source": ep_meta.get("source", ep_dir.name),
            "segment_idx": ep_meta.get("segment_idx", 0),
            "orig_start_frame": ep_meta.get("orig_start_frame", 0),
            "orig_end_frame": ep_meta.get("orig_end_frame", 0),
        })

        # Collect modality stats
        with open(ds_dir / "meta" / "modality.json") as f:
            ep_modality = json.load(f)
        # modality.json structure: {"state": {key: {dim, min, max, mean, std}}, "action": {...}}
        for modality, keys_dict in ep_modality.items():
            if modality not in group_arrays:
                group_arrays[modality] = {}
            for key, stats in keys_dict.items():
                # Skip non-numeric modalities (annotation, video) — they lack "dim"
                if "dim" not in stats:
                    continue
                # Reconstruct per-episode array from min/max/mean isn't possible (no raw per-frame)
                # Instead we stored mean/std; accumulate sums weighted by N
                arr_shape = (N, stats["dim"])
                # We can't reconstruct raw, so approximate stats later via recompute from parquet state
                # Store N and stats for weighted merge
                if key not in group_arrays[modality]:
                    group_arrays[modality][key] = {"dim": stats["dim"], "N": 0,
                                                    "min": np.full(stats["dim"], np.inf),
                                                    "max": np.full(stats["dim"], -np.inf),
                                                    "sum": np.zeros(stats["dim"]),
                                                    "sumsq": np.zeros(stats["dim"])}
                g = group_arrays[modality][key]
                g["N"] += N
                g["min"] = np.minimum(g["min"], np.array(stats["min"]))
                g["max"] = np.maximum(g["max"], np.array(stats["max"]))
                g["sum"] += np.array(stats["mean"]) * N
                g["sumsq"] += (np.array(stats["std"]) ** 2 + np.array(stats["mean"]) ** 2) * N

        if info_template is None:
            info_template = json.load(open(ds_dir / "meta" / "info.json"))

        global_ep_idx += 1
        global_frame_idx += N
        print(f"  [{global_ep_idx:2d}] {ep_dir.name}: {N} frames, task='{task_str[:40]}'")

    # Write merged meta
    with open(out_meta / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with open(out_meta / "episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")

    # Merged modality.json — preserve key order from MODE_CONFIGS, add start/end/original_key
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
    from human_ego_config import MODE_CONFIGS  # noqa: E402
    modality_keys = MODE_CONFIGS[mode]["modality_keys"]
    key_dims = {k: v["dim"] for k, v in (info_template.get("features", {}).items()
                if False else iter([]))}  # placeholder
    # Compute start index per key in canonical order
    off = 0
    key_slice = {}
    for k in modality_keys:
        dim = group_arrays["state"][k]["dim"]
        key_slice[k] = (off, off + dim)
        off += dim

    merged_modality = {}
    for modality, keys_dict in group_arrays.items():
        # Only process numeric modalities (state, action); skip empty ones (annotation, video)
        if not keys_dict:
            continue
        merged_modality[modality] = {}
        orig_key = "observation.state" if modality == "state" else "action"
        for key in modality_keys:  # canonical order
            g = keys_dict[key]
            n = g["N"]
            mean = g["sum"] / n
            var = g["sumsq"] / n - mean ** 2
            std = np.sqrt(np.maximum(var, 0))
            s = key_slice[key]
            merged_modality[modality][key] = {
                "dim": g["dim"],
                "start": s[0],
                "end": s[1],
                "original_key": orig_key,
                "min": g["min"].tolist(),
                "max": g["max"].tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
    # Add video + annotation modality meta (required by GR00T loader)
    merged_modality["video"] = {"ego_view": {"original_key": "observation.images.ego_view"}}
    merged_modality["annotation"] = {"human.task_description": {"original_key": "task_index"}}
    with open(out_meta / "modality.json", "w") as f:
        json.dump(merged_modality, f, indent=2)

    # Merged info.json
    info_template["total_episodes"] = global_ep_idx
    info_template["total_frames"] = global_frame_idx
    with open(out_meta / "info.json", "w") as f:
        json.dump(info_template, f, indent=2)

    print(f"\n=== Merged ===")
    print(f"  Episodes: {global_ep_idx}")
    print(f"  Total frames: {global_frame_idx}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge per-episode LeRobot datasets into one")
    parser.add_argument("--input-dir", type=str, default="../output")
    parser.add_argument("--mode", type=str, required=True, choices=["sharpa", "gripper"])
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    merge(Path(args.input_dir), args.mode, Path(args.output_dir))


if __name__ == "__main__":
    main()

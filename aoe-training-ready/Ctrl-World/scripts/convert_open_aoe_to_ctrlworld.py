#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.models import AutoencoderKLTemporalDecoder


DEFAULT_INPUT_ROOT = Path("/PATH_TO/general_datasets/Open-AoE/poc_deliver")
DEFAULT_OUTPUT_ROOT = Path("/PATH_TO/dataset/Open_AoE/open_aoe_mano_nextstate_224_110d_ctrlworld")
FALLBACK_TASK = "egocentric hand manipulation"
HAND_POSE_DIMS = 45
HAND_VECTOR_DIMS = 1 + 3 + 3 + 3 + HAND_POSE_DIMS
STATE_DIMS = 2 * HAND_VECTOR_DIMS
ACTION_DIMS = 2 * HAND_VECTOR_DIMS


@dataclass
class EpisodeSummary:
    sample: str
    episode_id: int
    split: str
    source_frames: int
    video_frames: int
    samples: int
    task: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Open-AoE into CtrlWorld latent dataset format.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--dataset-name", default="open_aoe_ctrlworld")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=192)
    parser.add_argument("--rgb-skip", type=int, default=3)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
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


def source_fps(sample_dir: Path) -> float:
    info_path = sample_dir / "ego_process" / "ego_undistorted_video" / "undistorted_video_info.json"
    if not info_path.exists():
        return 30.0
    return safe_float(read_json(info_path).get("fps"), 30.0)


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
    segments: list[tuple[int, int, str]] = []
    for seg in raw_segments if isinstance(raw_segments, list) else []:
        if not isinstance(seg, dict):
            continue
        start = safe_int(seg.get("start_frame"), 0)
        end = safe_int(seg.get("end_frame"), start)
        if end <= start:
            continue
        start = max(0, min(start, num_frames))
        end = max(0, min(end, num_frames))
        actions = seg.get("atomic_action") or []
        if isinstance(actions, list) and actions and isinstance(actions[0], dict):
            task = format_action_task(str(seg.get("scene") or "indoor"), actions[0])
        else:
            task = FALLBACK_TASK
        if end > start:
            segments.append((start, end, task))
    return sorted(segments, key=lambda item: item[0])


def task_for_frame(segments: list[tuple[int, int, str]], frame_index: int, cursor: int) -> tuple[str, int]:
    while cursor + 1 < len(segments) and frame_index >= segments[cursor][1]:
        cursor += 1
    if segments and segments[cursor][0] <= frame_index < segments[cursor][1]:
        return segments[cursor][2], cursor
    return FALLBACK_TASK, cursor


def resize_letterbox_bgr(bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(height / h, width / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return np.ascontiguousarray(canvas)


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


def vector_at(vectors: np.ndarray, frame_index: int) -> np.ndarray:
    out = np.zeros((STATE_DIMS,), dtype=np.float32)
    out[:HAND_VECTOR_DIMS] = vectors[0, frame_index]
    out[HAND_VECTOR_DIMS:] = vectors[1, frame_index]
    return out


def encode_latents(vae, frames_bgr: list[np.ndarray], batch_size: int, device: str) -> torch.Tensor:
    latents = []
    for start in range(0, len(frames_bgr), batch_size):
        batch_bgr = frames_bgr[start : start + batch_size]
        rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch_bgr], axis=0)
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.mul(2.0).sub(1.0).to(device)
        with torch.no_grad():
            latent = vae.encode(tensor).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
        latents.append(latent)
    return torch.cat(latents, dim=0)


def write_video(path: Path, frames_bgr: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


def convert_episode(
    sample_dir: Path,
    output_root: Path,
    dataset_name: str,
    episode_id: int,
    split: str,
    vae,
    args: argparse.Namespace,
) -> tuple[EpisodeSummary, np.ndarray]:
    hands_path = sample_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    video_path = sample_dir / "ego_process" / "ego_undistorted_video" / "raw_video_undistorted.mp4"
    if not hands_path.exists():
        raise FileNotFoundError(hands_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    fps = source_fps(sample_dir)
    with np.load(hands_path, mmap_mode="r") as hands:
        source_frames = int(hands["pred_valid"].shape[1])
        vectors = hand_vectors(hands, fps=fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    max_source = min(source_frames, args.max_frames_per_episode * args.rgb_skip)
    frames_bgr: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    segments = load_annotation_tasks(sample_dir, source_frames)
    task_cursor = 0
    first_task = FALLBACK_TASK
    try:
        for frame_index in range(max_source):
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_index % args.rgb_skip != 0:
                continue
            task, task_cursor = task_for_frame(segments, frame_index, task_cursor)
            if not frames_bgr:
                first_task = task
            frames_bgr.append(resize_letterbox_bgr(bgr, args.height, args.width))
            target = min(frame_index + args.rgb_skip, source_frames - 1)
            state_rows.append(vector_at(vectors, frame_index))
            action_rows.append(vector_at(vectors, target))
    finally:
        cap.release()

    if not frames_bgr:
        raise RuntimeError(f"No frames extracted from {sample_dir}")

    dataset_root = output_root / dataset_name
    for cam_id in range(3):
        write_video(dataset_root / "videos" / split / str(episode_id) / f"{cam_id}.mp4", frames_bgr, args.fps)
    latent = encode_latents(vae, frames_bgr, args.batch_size, args.device)
    for cam_id in range(3):
        latent_path = dataset_root / "latent_videos" / split / str(episode_id) / f"{cam_id}.pt"
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent, latent_path)

    states = np.stack(state_rows, axis=0).astype(np.float32)
    actions = np.stack(action_rows, axis=0).astype(np.float32)
    annotation = {
        "texts": [first_task],
        "episode_id": episode_id,
        "success": 1,
        "video_length": len(frames_bgr),
        "state_length": len(states),
        "raw_length": source_frames,
        "videos": [{"video_path": f"videos/{split}/{episode_id}/{cam_id}.mp4"} for cam_id in range(3)],
        "latent_videos": [
            {"latent_video_path": f"latent_videos/{split}/{episode_id}/{cam_id}.pt"} for cam_id in range(3)
        ],
        "observation.state.open_aoe": states.tolist(),
        "action.open_aoe": actions.tolist(),
    }
    ann_path = dataset_root / "annotation" / split / f"{episode_id}.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)

    sample_items = [{"episode_id": episode_id, "frame_ids": [idx]} for idx in range(len(frames_bgr))]
    summary = EpisodeSummary(
        sample=sample_dir.name,
        episode_id=episode_id,
        split=split,
        source_frames=source_frames,
        video_frames=len(frames_bgr),
        samples=len(sample_items),
        task=first_task,
    )
    return summary, actions


def write_meta(output_root: Path, dataset_name: str, samples_by_split: dict[str, list[dict]], actions: np.ndarray) -> None:
    meta_root = output_root / "dataset_meta_info" / dataset_name
    meta_root.mkdir(parents=True, exist_ok=True)
    for split, samples in samples_by_split.items():
        with (meta_root / f"{split}_sample.json").open("w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
    q01 = np.quantile(actions, 0.01, axis=0).astype(float).tolist()
    q99 = np.quantile(actions, 0.99, axis=0).astype(float).tolist()
    with (meta_root / "stat.json").open("w", encoding="utf-8") as f:
        json.dump({"state_01": q01, "state_99": q99, "action_dim": ACTION_DIMS}, f, indent=2)


def main() -> int:
    args = parse_args()
    if args.rgb_skip <= 0:
        raise ValueError("--rgb-skip must be positive")
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    candidates = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise FileNotFoundError(f"No Open-AoE samples found under {args.input_root}")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    args.device = device
    vae = AutoencoderKLTemporalDecoder.from_pretrained(str(args.svd_model_path), subfolder="vae").to(device)
    vae.eval()

    samples_by_split: dict[str, list[dict]] = {"train": [], "val": []}
    summaries: list[EpisodeSummary] = []
    all_actions = []
    for episode_id, sample_dir in enumerate(candidates):
        split = "val" if episode_id % 8 == 7 else "train"
        summary, actions = convert_episode(
            sample_dir=sample_dir,
            output_root=args.output_root,
            dataset_name=args.dataset_name,
            episode_id=episode_id,
            split=split,
            vae=vae,
            args=args,
        )
        summaries.append(summary)
        all_actions.append(actions)
        samples_by_split[split].extend({"episode_id": episode_id, "frame_ids": [idx]} for idx in range(summary.video_frames))
        print(f"[{episode_id + 1}/{len(candidates)}] {sample_dir.name} split={split} frames={summary.video_frames}")

    actions_cat = np.concatenate(all_actions, axis=0)
    write_meta(args.output_root, args.dataset_name, samples_by_split, actions_cat)
    with (args.output_root / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_root": str(args.input_root),
                "output_root": str(args.output_root),
                "dataset_name": args.dataset_name,
                "state_dim": STATE_DIMS,
                "action_dim": ACTION_DIMS,
                "rgb_skip": args.rgb_skip,
                "episodes": [asdict(item) for item in summaries],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] wrote CtrlWorld Open-AoE dataset to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-sequence-length", type=int, default=226)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16 if name == "float16" else torch.float32


def read_frames(path: Path, num_frames: int, height: int, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames, axis=0)


@torch.no_grad()
def encode_text(tokenizer, text_encoder, text: str, max_len: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    emb = text_encoder(tokens.input_ids.to(device), tokens.attention_mask.to(device)).last_hidden_state
    return emb[0].to(dtype=dtype).cpu()


@torch.no_grad()
def encode_video(vae, frames: np.ndarray, device: str, dtype: torch.dtype) -> torch.Tensor:
    video = torch.from_numpy(frames).float().permute(3, 0, 1, 2).unsqueeze(0)
    video = video / 255.0 * 2.0 - 1.0
    mu = vae.encode(video.to(device=device, dtype=dtype)).latent_dist.mean
    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    return ((mu - latents_mean) / latents_std).to(torch.bfloat16).cpu()


def action_norm(dataset_root: Path, action_dim: int = 110) -> dict:
    vals = []
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(path, columns=["action"])
        vals.append(np.stack(df["action"].values).astype(np.float32))
    data = np.concatenate(vals, axis=0) if vals else np.zeros((1, action_dim), dtype=np.float32)
    return {
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def patch_episode_action_config(dataset_root: Path, frames: int) -> None:
    path = dataset_root / "meta" / "episodes.jsonl"
    patched = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("tasks", ["egocentric hand manipulation"])[0] or "egocentric hand manipulation"
            item["action_config"] = [{"start_frame": 0, "end_frame": min(int(item["length"]), frames), "action_text": text}]
            patched.append(item)
    with path.open("w", encoding="utf-8") as f:
        for item in patched:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_episodes(dataset_root: Path) -> list[dict]:
    with (dataset_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepared_artifacts_exist(dataset_root: Path, episodes: list[dict]) -> bool:
    if not (dataset_root / "empty_emb.pt").is_file():
        return False
    if not (dataset_root / "meta" / "open_aoe_action_norm.json").is_file():
        return False
    latent_dir = dataset_root / "latents" / "chunk-000" / "observation.images.front"
    for episode in episodes:
        ep = int(episode["episode_index"])
        cfg = episode["action_config"][0]
        end = int(cfg["end_frame"])
        if not (latent_dir / f"episode_{ep:06d}_0_{end}.pth").is_file():
            return False
    return True


def main() -> int:
    args = parse_args()
    patch_episode_action_config(args.dataset_root, args.frames)
    episodes = load_episodes(args.dataset_root)
    if prepared_artifacts_exist(args.dataset_root, episodes):
        print(f"[skip] LingBot latent artifacts already exist under {args.dataset_root}")
        return 0

    dtype = dtype_from_name(args.dtype)
    vae = load_vae(args.model_root / "vae", torch_dtype=dtype, torch_device=args.device).eval()
    text_encoder = load_text_encoder(args.model_root / "text_encoder", torch_dtype=dtype, torch_device=args.device).eval()
    tokenizer = load_tokenizer(args.model_root / "tokenizer")

    latent_dir = args.dataset_root / "latents" / "chunk-000" / "observation.images.front"
    latent_dir.mkdir(parents=True, exist_ok=True)

    empty_emb = None
    for episode in episodes:
        ep = int(episode["episode_index"])
        cfg = episode["action_config"][0]
        end = int(cfg["end_frame"])
        video_path = args.dataset_root / "videos" / "observation.images.front" / "chunk-000" / f"episode_{ep:06d}.mp4"
        frames = read_frames(video_path, end, args.height, args.width)
        text = str(cfg.get("action_text") or episode.get("tasks", ["egocentric hand manipulation"])[0])
        text_emb = encode_text(tokenizer, text_encoder, text, args.max_sequence_length, args.device, dtype)
        if empty_emb is None:
            empty_emb = torch.zeros_like(text_emb)
        latent = encode_video(vae, frames, args.device, dtype)
        _, channels, latent_frames, latent_h, latent_w = latent.shape
        flat = latent[0].permute(1, 2, 3, 0).reshape(-1, channels).contiguous()
        payload = {
            "latent": flat,
            "latent_num_frames": int(latent_frames),
            "latent_height": int(latent_h),
            "latent_width": int(latent_w),
            "video_num_frames": int(end),
            "video_height": int(args.height),
            "video_width": int(args.width),
            "text_emb": text_emb,
            "text": text,
            "frame_ids": list(range(end)),
            "start_frame": 0,
            "end_frame": int(end),
            "fps": 10,
            "ori_fps": 30,
        }
        out = latent_dir / f"episode_{ep:06d}_0_{end}.pth"
        torch.save(payload, out)
        print(f"[latent] {out} latent={tuple(latent.shape)}")

    torch.save(empty_emb if empty_emb is not None else torch.zeros((args.max_sequence_length, 4096)), args.dataset_root / "empty_emb.pt")
    with (args.dataset_root / "meta" / "open_aoe_action_norm.json").open("w", encoding="utf-8") as f:
        json.dump(action_norm(args.dataset_root), f, indent=2)
    print(f"[done] prepared LingBot latents under {args.dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

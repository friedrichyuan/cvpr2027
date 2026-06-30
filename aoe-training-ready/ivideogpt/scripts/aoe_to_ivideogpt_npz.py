#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert AoE egocentric video segments -> iVideoGPT .npz (action-free).

iVideoGPT's SimpleRoboticDatasetv2 / NPZParser expect, per episode, an .npz with
a single image key (default 'image' for unknown dataset names) holding frames of
shape (T, H, W, 3) uint8. Files are discovered via  <parent_dir>/<dataset_name>/*.npz
and named  image_eps_{i:08d}.npz  (the OXE converter convention).

This writes 64x64 (configurable) RGB frames decoded from each AoE segment's video.
Action-free: no 'action' key. (Action-conditioned would add 'action' (T, D) later.)

Usage:
  python aoe_to_ivideogpt_npz.py \
      --data-root /path/to/hololens_data --limit 3 \
      --out-dir /path/to/aoe_npz --dataset-name aoe \
      --resolution 64 --max-frames 128 --video raw   # or 'undistorted'
"""
import argparse
import os
import sys
import glob
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    import imageio
from PIL import Image


def find_segment_video(seg_dir, which):
    """Return path to the chosen mp4 inside an AoE segment dir, or None."""
    if which == "undistorted":
        cands = [os.path.join(seg_dir, "ego_process", "ego_undistorted_video",
                              "raw_video_undistorted.mp4")]
    else:  # raw (smaller)
        cands = [os.path.join(seg_dir, "raw_video.mp4")]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def decode_video(path, resolution, max_frames):
    """Decode mp4 -> (T, res, res, 3) uint8, evenly subsampled to <= max_frames."""
    reader = imageio.get_reader(path)
    try:
        n = reader.count_frames()
    except Exception:
        n = None
    # Choose frame indices
    if n and max_frames and n > max_frames:
        idxs = set(np.linspace(0, n - 1, max_frames).round().astype(int).tolist())
    else:
        idxs = None
    frames = []
    for i, frame in enumerate(reader):
        if idxs is not None and i not in idxs:
            continue
        img = Image.fromarray(frame).convert("RGB").resize(
            (resolution, resolution), Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.uint8))
        if max_frames and idxs is None and len(frames) >= max_frames:
            break
    reader.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0)  # (T, res, res, 3) uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", help="dir containing poc_raw_video_* segments")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="explicit list of mp4 paths (overrides --data-root)")
    ap.add_argument("--out-dir", required=True, help="parent_dir; npz go to <out-dir>/<dataset-name>/")
    ap.add_argument("--dataset-name", default="aoe")
    ap.add_argument("--key", default="image", help="npz key (must match get_display_key)")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=128, help="cap frames per episode (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="max segments (0 = all)")
    ap.add_argument("--video", choices=["raw", "undistorted"], default="raw")
    args = ap.parse_args()

    # Build the list of (label, video_path)
    items = []
    if args.videos:
        for p in args.videos:
            items.append((os.path.splitext(os.path.basename(p))[0], p))
    else:
        if not args.data_root:
            ap.error("need --data-root or --videos")
        segs = sorted(d for d in glob.glob(os.path.join(args.data_root, "poc_raw_video_*"))
                      if os.path.isdir(d))
        for seg in segs:
            v = find_segment_video(seg, args.video)
            if v:
                items.append((os.path.basename(seg), v))
            else:
                print(f"[skip] no {args.video} video in {seg}", file=sys.stderr)
    if args.limit:
        items = items[:args.limit]
    if not items:
        ap.error("no input videos found")

    out_dir = os.path.join(args.out_dir, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    for i, (label, path) in enumerate(items):
        arr = decode_video(path, args.resolution, args.max_frames)
        out = os.path.join(out_dir, f"{args.key}_eps_{i:08d}.npz")
        np.savez_compressed(out, **{args.key: arr})
        print(f"[{i+1}/{len(items)}] {label}: {arr.shape} {arr.dtype} -> {out}")

    print(f"DONE: {len(items)} episodes -> {out_dir}")


if __name__ == "__main__":
    main()

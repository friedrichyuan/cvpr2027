#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""aoe_to_dreamdojo.py — 把 AoE 视频适配成 DreamDojo (NVIDIA, Cosmos-Predict2.5) 的输入。

适配范围（与用户确认）：**数据管道 + LAM 条件推理**，不做 post-train（需 8×H100）。
DreamDojo 的 `MultiVideoActionDataset` 对非 gr1/g1/yam/agibot 路径走通用 `VideoDataset`：
递归读文件夹里的 MP4，并由 LAM(`LAM_400k.ckpt`) 从视频自推 latent action 作为条件——
因此 AoE 只需转成 DreamDojo 期望的 **480×640 MP4** 即可被其 dataloader 直接消费。

注意：
- 分辨率 **必须 480(H)×640(W)**（位置编码硬编码；降分辨率会 NaN/黑屏，见上游 issue#6）。
- 视频帧数必须 **> --num-frames**（否则采样窗口为空：`empty range for randrange`）。
- 显存：2B teacher 单卡推理峰值 ~31GB → 需 ≥45GB 卡（L40 可，24GB 不行）。

用法（CPU，任意有 imageio 的机器即可）：
  python aoe_to_dreamdojo.py --data-root /PATH_TO/Open-AoE/poc_deliver \
      --out-dir /PATH_TO/aoe_dd_videos --num-clips 3 --frames 150

转出的 480×640 MP4 目录交给 DreamDojo 的通用 VideoDataset 做 zero-shot 推理（命令见 README）。
"""
import argparse
import glob
import os

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    import imageio
from PIL import Image

DD_H, DD_W = 480, 640  # DreamDojo 期望分辨率（H×W），硬编码，勿改


def find_undistorted_videos(data_root):
    out = []
    for seg in sorted(glob.glob(os.path.join(data_root, "poc_raw_video_*"))):
        v = os.path.join(seg, "ego_process", "ego_undistorted_video", "raw_video_undistorted.mp4")
        if os.path.isfile(v):
            out.append((os.path.basename(seg), v))
    return out


def convert(vp, out_path, frames, h=DD_H, w=DD_W, stride=1):
    """解码 AoE undistorted 视频 -> 480×640 MP4（前 `frames` 帧，按 stride 抽）。"""
    r = imageio.get_reader(vp)
    buf = []
    for j, fr in enumerate(r):
        if j % stride != 0:
            continue
        buf.append(np.asarray(Image.fromarray(fr).convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.uint8))
        if len(buf) >= frames:
            break
    r.close()
    if not buf:
        raise RuntimeError(f"no frames from {vp}")
    wtr = imageio.get_writer(out_path, fps=10)
    for f in buf:
        wtr.append_data(f)
    wtr.close()
    return len(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="hololens_data dir (AoE segments)")
    ap.add_argument("--out-dir", required=True, help="output folder of MP4s for DreamDojo VideoDataset")
    ap.add_argument("--num-clips", type=int, default=3, help="how many segments to convert (0 = all)")
    ap.add_argument("--frames", type=int, default=150, help="frames per clip (must exceed inference --num-frames)")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    # 目录名不可含 gr1/g1/yam/agibot，否则 DreamDojo 会当成 LeRobot 机器人数据集
    os.makedirs(args.out_dir, exist_ok=True)
    vids = find_undistorted_videos(args.data_root)
    if args.num_clips:
        vids = vids[:args.num_clips]
    if not vids:
        ap.error(f"no AoE undistorted videos under {args.data_root}")
    for i, (name, vp) in enumerate(vids):
        out = os.path.join(args.out_dir, f"aoe_{i:02d}.mp4")
        n = convert(vp, out, args.frames, stride=args.stride)
        print(f"[{i+1}/{len(vids)}] {name}: {n}f @ {DD_W}x{DD_H} -> {out}")
    print(f"DONE: {len(vids)} clips -> {args.out_dir}")


if __name__ == "__main__":
    main()

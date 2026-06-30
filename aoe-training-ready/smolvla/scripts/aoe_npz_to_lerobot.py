#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
aoe_npz_to_lerobot.py — 快速从已解码的 aoe26 npz 直接构 LeRobot 数据集。

为什么：直接 --full 解码 110 段原视频 + lerobot 默认 SVT-AV1 编码，单段 ~3-4min，全量数小时。
已有 <npz-dir>/image_eps_{i:08d}.npz（image 64×64 + action26，已解码、与 clip 帧 1:1），
其顺序与 aoe_to_lerobot.iter_episodes 严格对齐（actcond 导出器同源 iter_all_episodes）。
故：
  - 图像：取自 npz（跳过重解码）；
  - state(22) / 原始 delta 动作：重跑 iter_episodes（纯 numpy，秒级）；
  - use_videos=False：图像存 PNG，跳过 AV1 慢编码；
  - 下采样到 max_frames，动作按窗口累加保持 delta 语义；lerobot 自行算 stats 归一化。

A/B：--dim 20（手部）或 26（手部+相机），用于 SmolVLA 20D vs 26D 消融（承接 iVideoGPT 结论）。

用法:
  python scripts/aoe_npz_to_lerobot.py --data-root /PATH_TO/Open-AoE/poc_deliver \
     --npz-dir /PATH_TO/aoe_npz/aoe26 --out /PATH_TO/lerobot_aoe/aoe26 \
     --repo-id aoe26 --dim 26 --max-frames 64
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aoe_to_lerobot as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="aoe")
    ap.add_argument("--dim", type=int, choices=[20, 26], default=20)
    ap.add_argument("--max-frames", type=int, default=64, help="每 episode 下采样上限")
    ap.add_argument("--no-exclude", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    exclude = set() if args.no_exclude else set(A.DEFAULT_EXCLUDE)
    npz_dir = os.path.expanduser(args.npz_dir)
    segs = A.find_segments(os.path.expanduser(args.data_root))
    if args.limit:
        segs = segs[:args.limit]

    ds = None
    i = 0            # 与 npz 全局 episode 序号对齐
    n_written = n_frames = 0
    mism = 0
    stop = False
    for name, path in segs:
        if stop:
            break
        for ep in A.iter_episodes(name, path, exclude, "delta", append_camera=False):
            fn = os.path.join(npz_dir, f"image_eps_{i:08d}.npz")
            i += 1
            if not os.path.exists(fn):
                print(f"  [warn] missing {fn}; 停止（npz 用尽/错位）", file=sys.stderr)
                stop = True
                break
            d = np.load(fn)
            img = d["image"]                                   # (Ti,64,64,3) uint8
            state = ep["state"].astype(np.float32)             # (Ts,22) raw
            a20 = ep["action"].astype(np.float32)              # (Ts,20) raw delta
            cam6 = A.camera_delta_6d(ep["canonical"]["cam_c2w"])
            raw = np.concatenate([a20, cam6], 1) if args.dim == 26 else a20
            # 对齐健康检查：同源 clip，npz 图像帧数应≈state 帧数
            if abs(img.shape[0] - state.shape[0]) > 2:
                mism += 1
            T = min(img.shape[0], state.shape[0], raw.shape[0])
            if T < 2:
                continue
            img, state, raw = img[:T], state[:T], raw[:T]
            m = min(args.max_frames, T)
            idxs = np.unique(np.linspace(0, T - 1, m).round().astype(int))
            if ds is None:
                Hh, Ww = img.shape[1], img.shape[2]
                features = {
                    "observation.images.ego": {"dtype": "image", "shape": (Hh, Ww, 3),
                                               "names": ["height", "width", "channel"]},
                    "observation.state": {"dtype": "float32", "shape": (22,), "names": None},
                    "action": {"dtype": "float32", "shape": (args.dim,), "names": None},
                }
                ds = LeRobotDataset.create(repo_id=args.repo_id, fps=A.FPS, features=features,
                                           root=os.path.expanduser(args.out), use_videos=False)
            for k, fr in enumerate(idxs):
                nxt = idxs[k + 1] if k + 1 < len(idxs) else T
                act = raw[fr:nxt].sum(0).astype(np.float32)    # 窗口累计 delta，保持语义
                ds.add_frame({"observation.images.ego": img[fr],
                              "observation.state": state[fr],
                              "action": act,
                              "task": ep["instruction"]})
            ds.save_episode()
            n_written += 1
            n_frames += len(idxs)
            if n_written % 100 == 0:
                print(f"  [{n_written} ep] ~{n_frames} frames", flush=True)
    print(f"[OK] {n_written} episodes, ~{n_frames} frames -> {args.out} (dim={args.dim}); "
          f"长度不匹配 episode={mism}")
    if mism > 5:
        print("[!] 多个 episode 图像/状态长度不匹配，可能 npz 与 iter_episodes 错位，请核对。", file=sys.stderr)


if __name__ == "__main__":
    main()

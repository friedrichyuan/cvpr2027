#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""aoe_to_ivideogpt_actcond.py — 导出 AoE -> iVideoGPT action-conditioned npz。

每个原子动作 clip -> 一个 npz：
  image  (T, res, res, 3) uint8   —— 解码 undistorted 视频、resize
  action (T, D) float32           —— 已按 action_stats 标准化
D=20: build_episode 的相机系手部 delta（复用 aoe_to_lerobot）。
D=26: 再 append 相机 6D 增量（相对位姿 平移3 + 轴角3，从 cam_c2w 连续帧算）。

两步用法（先算 stats 再导出，保证动作预归一化——iVideoGPT 模型内不归一化）：
  # 1) 统计（无需解码视频，快）：算 26D 每维 mean/std
  python aoe_to_ivideogpt_actcond.py --data-root <segs> --compute-stats --stats action_stats.json
  # 2) 导出（解码视频 + 归一化）：
  python aoe_to_ivideogpt_actcond.py --data-root <segs> --stats action_stats.json \
      --out-dir <parent> --dataset-name aoe20 --dim 20 --resolution 64
  python aoe_to_ivideogpt_actcond.py --data-root <segs> --stats action_stats.json \
      --out-dir <parent> --dataset-name aoe26 --dim 26 --resolution 64
"""
import argparse
import json
import os
import sys

import numpy as np

# 复用 aoe_to_lerobot 的解析/解码/几何工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aoe_to_lerobot as A  # noqa: E402


def matrix_to_axis_angle(R):
    """旋转矩阵 (3,3) -> 轴角 (3,)。小角度稳定；相邻帧相机旋转很小，不触发 θ≈π。"""
    R = np.asarray(R, dtype=np.float64)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    theta = np.arccos(cos)
    if theta < 1e-7:
        return np.zeros(3, dtype=np.float64)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-7:  # θ≈π 退化，罕见；退回对角线估计
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, 1.0))
        return axis * theta
    return (v / n) * theta


def camera_delta_6d(cam_c2w):
    """cam_c2w (T,4,4) -> (T,6) 相对相机运动 [平移3, 轴角3]；末帧=0。
    无效(全零/不可逆)则返回全零。"""
    cam = np.asarray(cam_c2w, dtype=np.float64)
    T = cam.shape[0]
    out = np.zeros((T, 6), dtype=np.float32)
    if cam.shape[1:] != (4, 4) or not np.isfinite(cam).all() or np.allclose(cam, 0):
        return out
    for t in range(T - 1):
        try:
            rel = np.linalg.inv(cam[t]) @ cam[t + 1]
        except np.linalg.LinAlgError:
            continue
        out[t, :3] = rel[:3, 3]
        out[t, 3:] = matrix_to_axis_angle(rel[:3, :3])
    return out


def episode_action(ep, dim):
    """返回 (T, dim) 未归一化动作。"""
    a20 = ep["action"].astype(np.float32)            # (T,20) 手部 delta（相机系）
    if dim == 20:
        return a20
    cam6 = camera_delta_6d(ep["canonical"]["cam_c2w"])  # (T,6)
    T = min(a20.shape[0], cam6.shape[0])
    return np.concatenate([a20[:T], cam6[:T]], axis=1).astype(np.float32)  # (T,26)


def iter_all_episodes(data_root, exclude, action_mode, limit, seg_start=0, seg_end=0):
    segs = A.find_segments(data_root)
    if seg_end:                       # process a disjoint segment range [seg_start, seg_end) (parallel build workers)
        segs = segs[seg_start:seg_end]
    elif limit:
        segs = segs[:limit]
    for name, path in segs:
        for ep in A.iter_episodes(name, path, exclude, action_mode):
            yield name, ep


def compute_stats(data_root, exclude, action_mode, limit, stats_path):
    """对 26D 动作累计 per-dim mean/std（含 20D 子集），并记录 validity。"""
    n = 0
    s1 = np.zeros(26, dtype=np.float64)
    s2 = np.zeros(26, dtype=np.float64)
    vL = vR = vtot = 0
    n_ep = 0
    for name, ep in iter_all_episodes(data_root, exclude, action_mode, limit):
        a = episode_action(ep, 26)  # (T,26)
        s1 += a.sum(axis=0)
        s2 += (a.astype(np.float64) ** 2).sum(axis=0)
        n += a.shape[0]
        st = ep["state"]
        vL += float(st[:, 20].sum()); vR += float(st[:, 21].sum()); vtot += st.shape[0]
        n_ep += 1
    mean = s1 / max(n, 1)
    var = np.maximum(s2 / max(n, 1) - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-4, 1.0, std)  # 防止除零/常量维
    stats = {
        "dim": 26, "n_frames": int(n), "n_episodes": int(n_ep),
        "mean": mean.astype(float).tolist(), "std": std.astype(float).tolist(),
        "validity_left": round(vL / max(vtot, 1), 4), "validity_right": round(vR / max(vtot, 1), 4),
        "raw_abs_mean": np.abs(mean).round(5).tolist(),
        "layout": "0-19=hand delta(L[xyz3,rot6d6,grip1],R[...]); 20-25=camera delta[trans3,axisangle3]",
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] episodes={n_ep} frames={n} validity L={stats['validity_left']} R={stats['validity_right']}")
    print(f"[stats] hand |mean|: {np.abs(mean[:20]).mean():.4f}  cam |mean|: {np.abs(mean[20:]).mean():.4f}")
    print(f"[stats] std range: [{std.min():.4g}, {std.max():.4g}]   -> {stats_path}")
    assert np.isfinite(mean).all() and np.isfinite(std).all(), "stats 含 NaN/Inf"


def run_export(data_root, exclude, action_mode, limit, stats_path, out_dir, dataset_name, dim, res, max_frames, seg_start=0, seg_end=0):
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"][:dim], dtype=np.float32)
    std = np.array(stats["std"][:dim], dtype=np.float32)
    decode_all = A._get_full_decoder()
    out = os.path.join(out_dir, dataset_name)
    os.makedirs(out, exist_ok=True)
    i = 0
    n_frames_total = 0
    _cache = {"vp": None, "frames": None}
    for name, ep in iter_all_episodes(data_root, exclude, action_mode, limit, seg_start, seg_end):
        vp = ep["video"]
        if vp != _cache["vp"]:                        # decode each segment video ONCE (avoid per-clip re-decode from frame 0)
            _cache["vp"] = vp
            _cache["frames"] = decode_all(vp, (res, res))
        s, e = ep["frame_start"], ep["frame_end"]
        frames = _cache["frames"][s:e]                # list of (res,res,3) uint8
        if not frames:
            print(f"  [skip] {name} ep frames empty", file=sys.stderr); continue
        act = episode_action(ep, dim)
        T = min(len(frames), act.shape[0])
        if T < 2:
            continue
        img = np.stack(frames[:T], axis=0).astype(np.uint8)      # (T,res,res,3)
        act = ((act[:T] - mean) / std).astype(np.float32)        # 标准化 (T,dim)
        fn = os.path.join(out, f"image_eps_{i:08d}.npz")
        np.savez_compressed(fn, image=img, action=act)
        i += 1; n_frames_total += T
        if i <= 3:
            print(f"  ep{i-1}: {name} image{img.shape} action{act.shape} "
                  f"a_mean={act.mean():+.3f} a_std={act.std():.3f}")
    print(f"DONE: {i} episodes, {n_frames_total} frames -> {out}  (dim={dim})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--dataset-name", default="aoe20")
    ap.add_argument("--dim", type=int, choices=[20, 26], default=20)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seg-start", type=int, default=0)   # parallel build: this worker handles segments [seg_start, seg_end)
    ap.add_argument("--seg-end", type=int, default=0)     # 0 = all (single-process)
    ap.add_argument("--action-mode", choices=["delta", "next"], default="delta")
    ap.add_argument("--stats", required=True, help="path to action_stats.json (read for export, write for --compute-stats)")
    ap.add_argument("--compute-stats", action="store_true")
    ap.add_argument("--no-exclude", action="store_true")
    args = ap.parse_args()
    exclude = set() if args.no_exclude else set(A.DEFAULT_EXCLUDE)

    if args.compute_stats:
        compute_stats(args.data_root, exclude, args.action_mode, args.limit, args.stats)
    else:
        if not args.out_dir:
            ap.error("export 需要 --out-dir")
        run_export(args.data_root, exclude, args.action_mode, args.limit, args.stats,
                   args.out_dir, args.dataset_name, args.dim, args.resolution, args.max_frames,
                   args.seg_start, args.seg_end)


if __name__ == "__main__":
    main()

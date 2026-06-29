#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
smolvla_eval.py — 离线评估 SmolVLA 在 AoE held-out 上的动作预测（无机器人/环境，不测 success rate）。

用 lerobot 原生 processor（rename ego→camera1 + 归一化 + 语言 tokenize）保证与训练同口径：
  batch = preprocessor(raw_batch); policy.forward(batch) -> (loss, _); policy.predict_action_chunk(batch)
报告：
  1) held-out forward loss（泛化/「跑通+」）；
  2) 预测 vs GT（归一化空间）逐维 R²；dim=26 时拆 hand(0:20) vs camera(20:26)——看 VLA 是否也
     "相机比手部更可预测"（呼应 iVideoGPT/LAM）；
  3) baseline：zero / persistence(上一帧) / shuffle。

用法:
  python scripts/smolvla_eval.py --policy-path /PATH_TO/smolvla_aoe26/checkpoints/last/pretrained_model \
     --dataset-root /PATH_TO/lerobot_aoe/aoe26 --repo-id aoe26 --dim 26 --out /PATH_TO/smolvla_aoe26
"""
import argparse, json, os
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


def agg_r2(G, P):
    return float(1.0 - ((G - P) ** 2).sum() / (((G - G.mean(0)) ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--dim", type=int, default=26)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-samples", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    pp = os.path.expanduser(args.policy_path)
    policy = SmolVLAPolicy.from_pretrained(pp).to(dev).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=pp)

    # 按 policy 的动作分块加载（delta_timestamps）
    cs = int(getattr(policy.config, "chunk_size", 1))
    meta_fps = None
    ds0 = LeRobotDataset(args.repo_id, root=os.path.expanduser(args.dataset_root))
    fps = ds0.fps
    delta = {"action": [i / fps for i in range(cs)]}
    try:
        ds = LeRobotDataset(args.repo_id, root=os.path.expanduser(args.dataset_root), delta_timestamps=delta)
    except Exception as e:
        print("[warn] delta_timestamps load failed, fallback single-step:", e)
        ds = ds0; cs = 1
    print(f"[load] eps={ds.num_episodes} frames={ds.num_frames} fps={fps} chunk={cs}; policy on {dev}")

    ep_arr = np.asarray(ds.hf_dataset["episode_index"])
    uniq = np.unique(ep_arr)
    val_eps = set(uniq[int(len(uniq) * (1 - args.val_frac)):].tolist())
    val_idx = np.where(np.isin(ep_arr, list(val_eps)))[0].tolist()
    rng = np.random.RandomState(0)
    if len(val_idx) > args.max_samples:
        val_idx = sorted(rng.choice(val_idx, args.max_samples, replace=False).tolist())
    print(f"[val] {len(val_idx)} held-out frames from {len(val_eps)} episodes")

    P, G, losses = [], [], []
    B = args.batch
    for b in range(0, len(val_idx), B):
        items = [ds[i] for i in val_idx[b:b + B]]
        raw = {
            "observation.images.ego": torch.stack([it["observation.images.ego"] for it in items]),
            "observation.state": torch.stack([it["observation.state"] for it in items]),
            "action": torch.stack([it["action"] for it in items]),
            "task": [it.get("task", "") for it in items],
        }
        if "action_is_pad" in items[0]:
            raw["action_is_pad"] = torch.stack([it["action_is_pad"] for it in items])
        with torch.no_grad():
            batch = pre(raw)
            try:
                out = policy.forward(batch)
                loss = out[0] if isinstance(out, tuple) else (out["loss"] if isinstance(out, dict) else out)
                losses.append(float(loss))
            except Exception as e:
                if not losses:
                    print("[warn] forward failed:", repr(e)[:160])
            try:
                pred = policy.predict_action_chunk(batch)          # (B, cs, dim) normalized
                gt = batch["action"]
                p0 = pred[:, 0, :] if pred.dim() == 3 else pred
                g0 = gt[:, 0, :] if gt.dim() == 3 else gt
                P.append(p0[:, :args.dim].float().cpu().numpy())
                G.append(g0[:, :args.dim].float().cpu().numpy())
            except Exception as e:
                if not P:
                    print("[warn] predict failed:", repr(e)[:160])

    res = {"n_val": len(val_idx), "chunk": cs,
           "held_out_forward_loss": float(np.mean(losses)) if losses else None}
    if P:
        P = np.concatenate(P); G = np.concatenate(G)
        zero = np.zeros_like(G)
        persist = np.concatenate([G[:1], G[:-1]], 0)
        shuf = G[rng.permutation(len(G))]
        res.update(model_R2_agg=agg_r2(G, P), baseline_zero_R2=agg_r2(G, zero),
                   baseline_persistence_R2=agg_r2(G, persist), baseline_shuffle_R2=agg_r2(G, shuf))
        if args.dim == 26:
            res["R2_hand_agg"] = agg_r2(G[:, :20], P[:, :20])
            res["R2_cam_agg"] = agg_r2(G[:, 20:26], P[:, 20:26])
        print("\n=== SmolVLA held-out 动作预测（归一化空间）===")
        print(f"held-out forward loss : {res['held_out_forward_loss']}")
        print(f"model R²(agg)={res['model_R2_agg']:.3f}  | zero={res['baseline_zero_R2']:.3f} "
              f"persist={res['baseline_persistence_R2']:.3f} shuffle={res['baseline_shuffle_R2']:.3f}")
        if args.dim == 26:
            print(f"R²(camera 6D)={res['R2_cam_agg']:.3f}  vs  R²(hand 20D)={res['R2_hand_agg']:.3f}  "
                  f"({'camera 更可预测' if res['R2_cam_agg']>res['R2_hand_agg'] else 'hand 更可预测'})")
    else:
        print(f"[!] 仅得 forward loss = {res['held_out_forward_loss']}（predict 不可用）")

    out = os.path.expanduser(args.out) if args.out else os.path.dirname(pp)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "smolvla_eval.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[saved] {os.path.join(out, 'smolvla_eval.json')}")


if __name__ == "__main__":
    main()

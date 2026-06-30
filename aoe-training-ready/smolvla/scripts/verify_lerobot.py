#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
verify_lerobot.py — 校验 aoe_to_lerobot.py --full 产出的 LeRobot 数据集（任务 #2）

在装有 lerobot 的环境（通常是远程 GPU 机器）运行：把数据集加载回来，
遍历几个样本，打印 shape / instruction / 数值范围，确认 schema 合法、可被训练管线消费。
这是"训练前的最后一道数据校验"。

用法：
  python3 scripts/verify_lerobot.py --repo-id aoe26 --root /PATH_TO/lerobot_aoe/aoe26
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="aoe_hands_eef")
    ap.add_argument("--root", default=None, help="数据集本地根目录（aoe_to_lerobot.py --out）")
    ap.add_argument("--n", type=int, default=4, help="抽查样本数")
    args = ap.parse_args()

    try:
        import torch  # noqa
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset       # lerobot >=0.3
        except Exception:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # 旧版
    except Exception as e:
        print("[ERROR] 需在装有 lerobot + torch 的环境运行：", e, file=sys.stderr)
        sys.exit(2)

    print(f"加载数据集 repo_id={args.repo_id} root={args.root} ...")
    # video_backend=pyav：避免依赖系统 ffmpeg dylib 的 torchcodec（macOS 无 brew ffmpeg 时）
    try:
        ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    except TypeError:
        ds = LeRobotDataset(args.repo_id, root=args.root)

    print("\n--- 元信息 ---")
    print(f"episode 数 : {ds.num_episodes}")
    print(f"帧总数     : {ds.num_frames}")
    print(f"fps        : {ds.fps}")
    print(f"features   : {list(ds.features.keys())}")
    for k, v in ds.features.items():
        print(f"   {k:28s} {v.get('dtype'):8s} shape={v.get('shape')}")

    print(f"\n--- 抽查前 {args.n} 帧 ---")
    seen_tasks = set()
    for i in range(min(args.n, len(ds))):
        item = ds[i]
        st = item["observation.state"]
        ac = item["action"]
        img = item.get("observation.images.ego")
        task = item.get("task", "")
        seen_tasks.add(str(task))
        ishape = tuple(img.shape) if img is not None else None
        print(f"  [{i}] state{tuple(st.shape)} action{tuple(ac.shape)} image{ishape} "
              f"| task=\"{str(task)[:50]}\"")

    # 断言：维度符合转换器 v2 约定（state=22: 双手 xyz3+rot6d6+grip1 + 2 有效位；action=20 运动量）
    item = ds[0]
    assert tuple(item["observation.state"].shape)[-1] == 22, "state 维度应为 22"
    assert tuple(item["action"].shape)[-1] == 20, "action 维度应为 20"
    assert item.get("observation.images.ego") is not None, "缺少图像观测"
    print(f"\n示例任务指令: {list(seen_tasks)[:3]}")
    print("[OK] LeRobot 数据集 schema 合法、可被加载与遍历，可进入训练。")


if __name__ == "__main__":
    main()

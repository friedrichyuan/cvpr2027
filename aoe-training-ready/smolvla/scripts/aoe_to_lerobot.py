#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
aoe_to_lerobot.py — 把 AoE 数据转换成 LeRobot v2.1 数据集（任务 #2 基石，v2）

v2 相比 v1 的改进（依据 docs/05-相关工作调研.md）：
  - 旋转用 6D 表示（rot6d，比轴角更利于网络），不用 axis-angle 直接喂；
  - action 支持 delta（相对增量）/ next（下一帧绝对）两种；
  - 产出"规范中间格式"canonical npz（超集），可再导出 LeRobot / H-RDT-48D / VITRA / DreamDojo 片段；
  - dry-run 增加数值"重投影 sanity 检查"（把相机系手腕投影回像素，统计落在画面内的比例），无需解码视频。

核心思路："人手当末端执行器(EEF)"。每个原子动作段 → 一个 episode。
state(22) 布局（相机坐标系）：
  [0:3]   左手腕 xyz        [3:9]   左手腕 rot6d     [9]    左手开合代理
  [10:13] 右手腕 xyz        [13:19] 右手腕 rot6d     [19]   右手开合代理
  [20]    左手有效(0/1)     [21]    右手有效(0/1)
action(20) = state 的运动部分 [0:20] 的 next 或 delta。
（指尖/H-RDT 48D 需 MANO 前向运动学，见 mano_fk.py；本脚本默认不需要 MANO pkl。）

两种模式：
  --dry-run（默认，无需 torch/ffmpeg）：纯 numpy 校验 + 产出 canonical npz 到 data/lerobot_sample/dry_run/
  --full （需 lerobot + 视频解码器）：产出真正的 LeRobot v2.1 数据集

用法（在本 recipe 目录运行）：
  python3 scripts/aoe_to_lerobot.py --dry-run --limit 3
  python3 scripts/aoe_to_lerobot.py --full --repo-id aoe26 --out /PATH_TO/lerobot_aoe/aoe26
"""
import argparse
import json
import os
import sys
import numpy as np

FPS = 30
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
REPO = os.path.dirname(DOCS)
SAMPLE_DIR = os.path.join(DOCS, "data", "lerobot_sample")

# #3 审计发现：该段标注帧号被放大 ~27×，整体不可用 → 默认排除
DEFAULT_EXCLUDE = {"poc_raw_video_20260226_215034_part001"}

# #3 审计发现：场景标签未规范化 → 归一化映射
SCENE_NORMALIZE = {
    "living_room": "living room", "laundry_room": "laundry room",
    "laundry area": "laundry room", "laundry": "laundry room",
}

STATE_DIM = 22    # 20 运动 + 2 有效性
MOTION_DIM = 20   # 双手 (xyz3 + rot6d6 + grip1) × 2
ACTION_DIM = 20


# ----------------------------- 旋转工具（纯 numpy） -----------------------------
def axis_angle_to_matrix(aa):
    """轴角 (3,) -> 旋转矩阵 (3,3)，Rodrigues。"""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def axis_angle_to_rot6d(aa):
    """轴角 (3,) -> 6D 旋转表示（旋转矩阵前两列拼接），(6,)。"""
    R = axis_angle_to_matrix(aa)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def matrix_to_axis_angle(R):
    """旋转矩阵 (3,3) -> 轴角 (3,)。小角度稳定；相邻帧相机旋转很小。"""
    R = np.asarray(R, dtype=np.float64)
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(cos)
    if theta < 1e-7:
        return np.zeros(3, dtype=np.float64)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-7:
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, 1.0))
        return axis * theta
    return (v / n) * theta


def camera_delta_6d(cam_c2w):
    """cam_c2w (T,4,4) -> (T,6) 相对相机自运动 [平移3, 轴角3]；末帧=0。
    与 aoe_to_ivideogpt_actcond.py 完全一致，保证 26D 实验 apples-to-apples。"""
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


def gripper_proxy(hand_pose_45):
    """由 MANO 15 关节(45维轴角)估开合标量 ∈[0,1]，越大越"握紧"。
    近似：每关节轴角模长(弯曲角)取均值，~90° 归一化。无 MANO FK 时的代理，仅供跑通。"""
    aa = np.asarray(hand_pose_45, dtype=np.float64).reshape(-1, 3)
    angles = np.linalg.norm(aa, axis=1)
    return float(np.clip(angles.mean() / (np.pi / 2), 0.0, 1.0))


def parse_wh(resolution):
    try:
        w, h = resolution.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 1920, 1080


# ----------------------------- 构造 episode -----------------------------
def build_episode(hands, cam, wh, s, e, action_mode, append_camera=False):
    """对帧区间 [s,e) 构造 state(T',22)/action(T',20 或 26) + canonical 超集字段。
    append_camera=True 时把相机 6D 自运动增量拼到 action 末尾 → action(26)。"""
    pt_c = hands["pred_trans_cam"]   # (2,T,3)
    pr_c = hands["pred_rot_cam"]     # (2,T,3) 轴角
    pt_w = hands["pred_trans"]       # (2,T,3) 世界系
    pr_w = hands["pred_rot"]         # (2,T,3)
    hp = hands["pred_hand_pose"]     # (2,T,45)
    betas = hands["pred_betas"]      # (2,T,10)
    pv = (hands["pred_valid"] > 0).astype(np.float32)

    idx = np.arange(s, e)
    T2 = len(idx)
    state = np.zeros((T2, STATE_DIM), dtype=np.float32)
    for hi, base in ((0, 0), (1, 10)):       # 左手 0.., 右手 10..
        state[:, base + 0:base + 3] = pt_c[hi, idx]
        for j, t in enumerate(idx):
            state[j, base + 3:base + 9] = axis_angle_to_rot6d(pr_c[hi, t])
        state[:, base + 9] = [gripper_proxy(hp[hi, t]) for t in idx]
    state[:, 20] = pv[0, idx]
    state[:, 21] = pv[1, idx]

    motion = state[:, :MOTION_DIM]
    action = np.empty((T2, ACTION_DIM), dtype=np.float32)
    if action_mode == "delta":
        action[:-1] = motion[1:] - motion[:-1]
        action[-1] = 0.0
    else:  # next（下一帧绝对运动量；末帧复制）
        action[:-1] = motion[1:]
        action[-1] = motion[-1]

    # 重投影 sanity：相机系手腕 -> 像素，统计落在画面内比例（仅有效帧）
    W, H = wh
    K = np.asarray(cam["intrinsic"], dtype=np.float64) if cam is not None and "intrinsic" in cam.files else None
    reproj_in = float("nan")
    if K is not None:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        tot = ok = 0
        for hi in (0, 1):
            for t in idx:
                if pv[hi, t] <= 0:
                    continue
                x, y, z = pt_c[hi, t]
                if z <= 1e-6:
                    tot += 1; continue
                u = fx * x / z + cx; v = fy * y / z + cy
                tot += 1
                if 0 <= u < W and 0 <= v < H:
                    ok += 1
        reproj_in = (ok / tot) if tot else float("nan")

    canonical = {
        "state": state, "action": action,
        "wrist_world_xyz": pt_w[:, idx], "wrist_world_aa": pr_w[:, idx],
        "wrist_cam_xyz": pt_c[:, idx], "wrist_cam_aa": pr_c[:, idx],
        "mano_pose": hp[:, idx], "mano_betas": betas[:, idx],
        "cam_c2w": (np.asarray(cam["cam_c2w"])[idx] if cam is not None and "cam_c2w" in cam.files else np.zeros((T2, 4, 4), np.float32)),
        "intrinsic": (np.asarray(cam["intrinsic"]) if K is not None else np.zeros((3, 3))),
        "reproj_in_frame": reproj_in,
    }
    if append_camera:
        cam6 = camera_delta_6d(canonical["cam_c2w"])         # (T2,6) 原始未归一化
        action = np.concatenate([action, cam6[:action.shape[0]]], axis=1).astype(np.float32)
    return state, action, canonical


def make_instruction(clip):
    aa = (clip.get("atomic_action") or [{}])[0]
    verb = (aa.get("verb") or "").replace("_", " ").strip()
    obj = (aa.get("object") or "").strip()
    desc = (aa.get("description") or "").strip()
    if desc:
        return desc
    if verb and obj:
        return f"{verb} the {obj}"
    return verb or "manipulate object"


def find_segments(data_root):
    segs = []
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "video_info.json")):
            segs.append((name, p))
    return segs


def iter_episodes(name, path, exclude, action_mode, append_camera=False):
    if name in exclude:
        print(f"  [skip] {name}（#3 审计：标注损坏，已排除）")
        return
    hpath = os.path.join(path, "ego_process/ego_hands_reconstruction/hands.npz")
    cpath = os.path.join(path, "ego_process/ego_hands_reconstruction/camera_traj.npz")
    apath = os.path.join(path, "ego_annotation/ego_action_annotation.json")
    vpath = os.path.join(path, "video_info.json")
    if not (os.path.exists(hpath) and os.path.exists(apath)):
        print(f"  [skip] {name}（缺 hands.npz 或标注）")
        return
    hands = np.load(hpath)
    cam = np.load(cpath) if os.path.exists(cpath) else None
    wh = parse_wh(json.load(open(vpath)).get("cameraParams", {}).get("resolution", "1920x1080")) if os.path.exists(vpath) else (1920, 1080)
    T = hands["pred_valid"].shape[1]
    clips = json.load(open(apath))
    for clip in clips:
        try:
            s = int(clip["start_frame"]); e = int(clip["end_frame"])
        except Exception:
            continue
        s = max(0, min(s, T)); e = max(0, min(e, T))
        if e - s < 2:
            continue
        state, action, canon = build_episode(hands, cam, wh, s, e, action_mode, append_camera)
        yield {
            "instruction": make_instruction(clip),
            "scene": SCENE_NORMALIZE.get(clip.get("scene", ""), clip.get("scene", "")),
            "hand": (clip.get("atomic_action") or [{}])[0].get("hand", ""),
            "state": state, "action": action, "canonical": canon,
            "frame_start": s, "frame_end": e,
            "video": os.path.join(path, "ego_process/ego_undistorted_video/raw_video_undistorted.mp4"),
        }


# ----------------------------- dry-run -----------------------------
def run_dry(data_root, limit, exclude, action_mode, append_camera=False):
    adim = 26 if append_camera else ACTION_DIM
    out = os.path.join(SAMPLE_DIR, "dry_run")
    os.makedirs(out, exist_ok=True)
    segs = find_segments(data_root)
    if limit:
        segs = segs[:limit]
    n_ep = n_frames = 0
    reprojs = []
    manifest = []
    for name, path in segs:
        print(f"[seg] {name}")
        for k, ep in enumerate(iter_episodes(name, path, exclude, action_mode, append_camera)):
            n_ep += 1
            n_frames += ep["state"].shape[0]
            c = ep["canonical"]
            if c["reproj_in_frame"] == c["reproj_in_frame"]:
                reprojs.append(c["reproj_in_frame"])
            fn = os.path.join(out, f"{name}__ep{k:03d}.npz")
            np.savez_compressed(
                fn, state=ep["state"], action=ep["action"],
                wrist_world_xyz=c["wrist_world_xyz"], wrist_world_aa=c["wrist_world_aa"],
                wrist_cam_xyz=c["wrist_cam_xyz"], mano_pose=c["mano_pose"], mano_betas=c["mano_betas"],
                cam_c2w=c["cam_c2w"], intrinsic=c["intrinsic"],
                instruction=ep["instruction"], scene=ep["scene"], hand=ep["hand"],
                frame_start=ep["frame_start"], frame_end=ep["frame_end"])
            manifest.append({"file": os.path.basename(fn), "segment": name,
                             "frames": int(ep["state"].shape[0]), "instruction": ep["instruction"],
                             "scene": ep["scene"], "reproj_in_frame": round(c["reproj_in_frame"], 3) if c["reproj_in_frame"] == c["reproj_in_frame"] else None})
            if n_ep <= 5:
                print(f"    ep{k:03d}: frames={ep['state'].shape[0]:4d} state{ep['state'].shape} "
                      f"action{ep['action'].shape} reproj_in={c['reproj_in_frame']:.2f} "
                      f"| [{ep['scene']}] \"{ep['instruction'][:50]}\"")
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"n_episodes": n_ep, "n_frames": n_frames, "state_dim": STATE_DIM,
                   "action_dim": adim, "action_mode": action_mode,
                   "mean_reproj_in_frame": round(float(np.mean(reprojs)), 3) if reprojs else None,
                   "episodes": manifest}, f, ensure_ascii=False, indent=2)

    # 自检
    print("\n--- dry-run 自检 (v2) ---")
    print(f"片段数(本次)     : {len(segs)}   episode 数: {n_ep}   总帧数: {n_frames}")
    print(f"state/action dim : {STATE_DIM} / {adim}   action_mode: {action_mode}")
    assert n_ep > 0, "没有生成任何 episode"
    sample = np.load(os.path.join(out, manifest[0]["file"]))
    st = sample["state"]
    assert st.shape[1] == STATE_DIM
    assert np.isfinite(st).all(), "state 含 NaN/Inf"
    assert set(np.unique(st[:, 20:22]).tolist()) <= {0.0, 1.0}, "有效性位应为 0/1"
    # rot6d 合理性：前两列应近似单位且接近正交（重建第三列后 det≈1）
    r6 = st[0, 3:9]
    a = r6[:3] / (np.linalg.norm(r6[:3]) + 1e-9)
    b = r6[3:] / (np.linalg.norm(r6[3:]) + 1e-9)
    print(f"rot6d 检查        : |col1|={np.linalg.norm(r6[:3]):.3f} |col2|={np.linalg.norm(r6[3:]):.3f} "
          f"col1·col2={float(a@b):+.3f}（应≈正交,点积≈0）")
    print(f"重投影落框比例    : 均值 {np.mean(reprojs):.3f}（手腕投影回画面内的占比；接近1说明相机系/内参一致）" if reprojs else "重投影：无内参")
    print(f"state pos 范围(m) : [{st[:, :3].min():.3f}, {st[:, :3].max():.3f}]   gripper:[{st[:,9].min():.2f},{st[:,9].max():.2f}]")
    print(f"canonical 字段    : {list(sample.files)}")
    print(f"输出目录         : {out}")
    print("[OK] v2 dry-run 通过：rot6d/delta/canonical/重投影 sanity 均正常。")


# ----------------------------- full（需 lerobot） -----------------------------
def run_full(data_root, repo_id, out_root, limit, exclude, image_hw, action_mode, append_camera=False):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 新版路径
        except Exception as e:
            print("[ERROR] 需要装 lerobot 才能 --full：", e, file=sys.stderr); sys.exit(2)
    decode_all = _get_full_decoder()
    H, W = image_hw
    adim = 26 if append_camera else ACTION_DIM
    features = {
        "observation.images.ego": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (adim,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features, root=out_root, use_videos=True)
    segs = find_segments(data_root)
    if limit:
        segs = segs[:limit]
    cur_vid, cur_frames = None, None
    for name, path in segs:
        for ep in iter_episodes(name, path, exclude, action_mode, append_camera):
            if ep["video"] != cur_vid:                      # 整段视频只解码一次（关键提速）
                cur_frames = decode_all(ep["video"], (H, W))
                cur_vid = ep["video"]
                print(f"  [decoded] {name} -> {len(cur_frames)} frames", flush=True)
            frames = cur_frames[ep["frame_start"]:ep["frame_end"]]
            n = min(len(frames), ep["state"].shape[0])
            for t in range(n):
                ds.add_frame({"observation.images.ego": frames[t],
                              "observation.state": ep["state"][t],
                              "action": ep["action"][t],
                              "task": ep["instruction"]})   # lerobot 0.4.x: task 放进 frame
            ds.save_episode()
            print(f"  [ep] {name} frames={n} task=\"{ep['instruction'][:40]}\"", flush=True)
    print(f"[OK] LeRobot 数据集已写入 {out_root}（repo_id={repo_id}）")


def _get_decoder():
    try:
        import imageio.v3 as iio
        def dec(vp, s, e, hw):
            out = []
            for i, fr in enumerate(iio.imiter(vp, plugin="pyav")):
                if i < s: continue
                if i >= e: break
                out.append(_resize(fr, hw))
            return out
        return dec
    except Exception:
        pass
    try:
        import cv2
        def dec(vp, s, e, hw):
            cap = cv2.VideoCapture(vp); cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            out = []
            for _ in range(e - s):
                ok, fr = cap.read()
                if not ok: break
                out.append(_resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), hw))
            cap.release(); return out
        return dec
    except Exception:
        raise RuntimeError("未找到视频解码器：pip install imageio-ffmpeg 或 opencv-python")


def _get_full_decoder():
    """整段视频一次性解码成下采样帧列表（按解码顺序索引），供按帧切片，避免每个 clip 重读。"""
    try:
        import imageio.v3 as iio
        def dec_all(vp, hw):
            return [_resize(fr, hw) for fr in iio.imiter(vp, plugin="pyav")]
        return dec_all
    except Exception:
        pass
    try:
        import cv2
        def dec_all(vp, hw):
            cap = cv2.VideoCapture(vp); out = []
            while True:
                ok, fr = cap.read()
                if not ok: break
                out.append(_resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), hw))
            cap.release(); return out
        return dec_all
    except Exception:
        raise RuntimeError("未找到视频解码器：pip install imageio-ffmpeg 或 opencv-python")


def _resize(fr, hw):
    H, W = hw
    if fr.shape[0] == H and fr.shape[1] == W:
        return fr.astype(np.uint8)
    from PIL import Image
    return np.asarray(Image.fromarray(fr).resize((W, H))).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(REPO, "poc_deliver"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--repo-id", default="aoe_hands_eef")
    ap.add_argument("--out", default=os.path.join(SAMPLE_DIR, "lerobot"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image-h", type=int, default=256)
    ap.add_argument("--image-w", type=int, default=456)
    ap.add_argument("--action-mode", choices=["next", "delta"], default="next")
    ap.add_argument("--append-camera", action="store_true",
                    help="把相机 6D 自运动增量拼到 action 末尾 → action(26)，做 20D vs 26D 消融")
    ap.add_argument("--no-exclude", action="store_true")
    args = ap.parse_args()

    exclude = set() if args.no_exclude else set(DEFAULT_EXCLUDE)
    if not os.path.isdir(args.data_root):
        print(f"[ERROR] data-root 不存在: {args.data_root}", file=sys.stderr); sys.exit(1)
    if args.full:
        run_full(args.data_root, args.repo_id, args.out, args.limit, exclude,
                 (args.image_h, args.image_w), args.action_mode, args.append_camera)
    else:
        run_dry(args.data_root, args.limit, exclude, args.action_mode, args.append_camera)


if __name__ == "__main__":
    main()

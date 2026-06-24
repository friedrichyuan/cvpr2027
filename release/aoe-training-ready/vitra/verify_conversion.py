# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""End-to-end verification of an AoE → VITRA conversion.

Three verification layers:
  1) Numerical self-consistency between converted world-space pose and source pred_*_cam.
  2) Wrist 3D point projection: overlay green dots on a few frames; visually the dot must
     fall on the hand wrist for the conversion to be correct.
  3) Per-episode summary stats so anomalies (NaNs, exploding stats) surface immediately.

Usage::

    python verify_conversion.py \\
        --aoe_root /path/to/aoe_source_data \\
        --out_root /path/to/vitra_converted_data \\
        --dataset_name aoe \\
        --out_dir output/verify
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def axis_angle_to_matrix(aa):
    flat = aa.reshape(-1, 3)
    return R.from_rotvec(flat).as_matrix().reshape(*aa.shape[:-1], 3, 3)


def load_episode(out_root: Path, dataset_name: str, ep_id: str):
    p = out_root / "Annotation" / dataset_name / "episodic_annotations" / f"{ep_id}.npy"
    return np.load(p, allow_pickle=True).item()


def find_clip_dir(aoe_root: Path, ep_id: str, dataset_name: str) -> Path:
    """Decode the clip directory path from an episode_id.

    ep_id format: ``<dataset_name>_<encoded_clip_path>_ep_NNNNNN``.

    ``encoded_clip_path`` joins the clip's path components with ``__`` (a
    separator chosen by the converter because it doesn't appear in directory
    names). For example ``delivery__zhiyuan__20260525__raw_X_seg_Y`` decodes
    back to ``delivery/zhiyuan/20260525/raw_X_seg_Y``.
    """
    parts = ep_id.split("_")
    # Drop the dataset_name prefix and the trailing '_ep_NNNNNN' suffix
    prefix_parts = dataset_name.split("_")
    for p in prefix_parts:
        if parts and parts[0] == p:
            parts.pop(0)
        else:
            break
    clip_id = "_".join(parts[:-2])
    # Decode '__' back into path separators
    if "__" in clip_id:
        rel = clip_id.replace("__", "/")
    else:
        rel = clip_id
    candidate = aoe_root / rel
    return candidate


def _find_hands_npz(clip_dir: Path) -> Optional[Path]:
    """Find hands.npz under AoE layout."""
    cand = clip_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    if cand.exists():
        return cand
    return None


def _find_undist_video(clip_dir: Path) -> Optional[Path]:
    """Find the undistorted .mp4 under AoE layout."""
    cand = clip_dir / "ego_process" / "ego_undistorted_video"
    if cand.exists():
        videos = list(cand.glob("*.mp4"))
        if videos:
            return videos[0]
    return None


def numerical_check(ep: dict, hands: np.lib.npyio.NpzFile) -> dict:
    """Compare converted world-space rotations/translations transformed via extrinsics to
    ant's own pred_rot_cam / pred_trans_cam — they should match to ~1e-4.

    The episode may be a SLICE of the source clip (multi-atomic-action splitting),
    so we map local frame indices through `video_decode_frame` before indexing
    the source hands.npz arrays.
    """
    R_w2c = ep["extrinsics"][:, :3, :3]
    t_w2c = ep["extrinsics"][:, :3, 3]
    decode_frame = np.asarray(ep["video_decode_frame"]).astype(int)
    pred_rot = hands["pred_rot"]
    pred_trans = hands["pred_trans"]
    pred_rot_cam = hands["pred_rot_cam"]
    pred_trans_cam = hands["pred_trans_cam"]
    pred_valid = np.asarray(hands["pred_valid"]) > 0.5

    T_local = R_w2c.shape[0]
    rot_errs, trans_errs = [], []
    for hand_idx in (0, 1):
        # Sample local frames whose corresponding source frame is valid
        valid_local = np.array([
            t for t in range(T_local) if pred_valid[hand_idx, decode_frame[t]]
        ])
        if len(valid_local) == 0:
            continue
        sample = valid_local[:: max(1, len(valid_local) // 30)][:30]
        for t in sample:
            src_t = decode_frame[t]
            Rw = axis_angle_to_matrix(pred_rot[hand_idx, src_t])
            Rcam_pred = R_w2c[t] @ Rw
            Rcam_ant = axis_angle_to_matrix(pred_rot_cam[hand_idx, src_t])
            rot_errs.append(np.linalg.norm(Rcam_pred - Rcam_ant))
            tcam_pred = R_w2c[t] @ pred_trans[hand_idx, src_t] + t_w2c[t]
            tcam_ant = pred_trans_cam[hand_idx, src_t]
            trans_errs.append(np.linalg.norm(tcam_pred - tcam_ant))

    return {
        "rot_err_max": float(np.max(rot_errs)) if rot_errs else 0.0,
        "rot_err_mean": float(np.mean(rot_errs)) if rot_errs else 0.0,
        "trans_err_max": float(np.max(trans_errs)) if trans_errs else 0.0,
        "trans_err_mean": float(np.mean(trans_errs)) if trans_errs else 0.0,
    }


def project_world_to_image(p_world, R_w2c, t_w2c, K):
    """p_world: (3,) → (u, v) on image plane (returns None if z<=0)."""
    p_cam = R_w2c @ p_world + t_w2c
    if p_cam[2] <= 1e-3:
        return None
    p_img = K @ (p_cam / p_cam[2])
    return float(p_img[0]), float(p_img[1])


def projection_overlay(ep: dict, src_video_path: Path, out_dir: Path, ep_id: str,
                       num_frames: int = 5):
    """Overlay green dot at projected wrist + red dot at projected palm centre, on a few frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    K = ep["intrinsics"]
    extr = ep["extrinsics"]
    decode_frame = np.asarray(ep["video_decode_frame"]).astype(int)
    T = extr.shape[0]
    sample_local = np.linspace(0, T - 1, num_frames).astype(int)

    cap = cv2.VideoCapture(str(src_video_path))
    if not cap.isOpened():
        print(f"[verify] cannot open {src_video_path}")
        return

    # Pre-read all needed frames sequentially (cv2 seek is unreliable on
    # compressed mp4 – it can return wrong frames, causing projection drift).
    max_src_f = int(decode_frame[sample_local[-1]])
    needed = {int(decode_frame[fl]): fl for fl in sample_local}
    frame_cache: dict[int, np.ndarray] = {}
    for idx in range(max_src_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in needed:
            frame_cache[idx] = frame
    cap.release()

    saved = []
    for f_local in sample_local:
        src_f = int(decode_frame[f_local])
        frame = frame_cache.get(src_f)
        if frame is None:
            continue
        Rw, tw = extr[f_local, :3, :3], extr[f_local, :3, 3]
        for side, color in [("left", (255, 80, 80)), ("right", (80, 255, 80))]:
            kept = ep[side]["kept_frames"]
            if not kept[f_local]:
                continue
            wrist_w = ep[side]["joints_worldspace"][f_local, 0]
            palm_w = ep[side]["joints_worldspace"][f_local, [0, 2, 5, 9, 13, 17]].mean(axis=0)
            for p_w, sz in [(wrist_w, 9), (palm_w, 5)]:
                uv = project_world_to_image(p_w, Rw, tw, K)
                if uv is None:
                    continue
                u, v = int(round(uv[0])), int(round(uv[1]))
                if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                    cv2.circle(frame, (u, v), sz, color, -1)
            cv2.putText(frame, side, (10, 30 if side == "left" else 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"local {int(f_local)}/{T-1}  src {src_f}",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        path = out_dir / f"{ep_id}_proj_f{int(f_local):04d}_src{src_f:04d}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(str(path))
    print(f"[verify] saved {len(saved)} projection frames to {out_dir}")
    return saved


def episode_stats(ep: dict) -> dict:
    extr = ep["extrinsics"]
    T = extr.shape[0]
    out = {"T": T, "anno_type": ep["anno_type"]}
    for side in ["left", "right"]:
        kept = ep[side]["kept_frames"].astype(bool)
        nv = int(kept.sum())
        out[f"{side}_valid"] = nv
        if nv:
            tr = ep[side]["transl_worldspace"][kept]
            out[f"{side}_transl_min"] = tr.min(axis=0).round(3).tolist()
            out[f"{side}_transl_max"] = tr.max(axis=0).round(3).tolist()
            jw = ep[side]["joints_worldspace"][kept]
            out[f"{side}_joints_nan"] = bool(np.isnan(jw).any())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoe_root", required=True, type=str,
                    help="The directory passed as --root_dir to the converter "
                         "(i.e. the parent under which clip_id paths are encoded).")
    ap.add_argument("--out_root", required=True, type=str)
    ap.add_argument("--dataset_name", default="aoe", type=str,
                    help="VITRA dataset name used at conversion time. Default: 'aoe'.")
    ap.add_argument("--episode", default=None, type=str,
                    help="episode_id; if omitted, picks the first episode in the index")
    ap.add_argument("--out_dir", default="output/verify", type=str)
    ap.add_argument("--num_frames", default=5, type=int)
    args = ap.parse_args()

    aoe_root = Path(args.aoe_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    dataset_name = args.dataset_name

    if args.episode is None:
        idx = np.load(
            out_root / "Annotation" / dataset_name / "episode_frame_index.npz",
            allow_pickle=True,
        )
        ep_id = str(idx["index_to_episode_id"][0])
    else:
        ep_id = args.episode
    print(f"[verify] dataset={dataset_name} episode={ep_id}")

    ep = np.load(
        out_root / "Annotation" / dataset_name / "episodic_annotations" / f"{ep_id}.npy",
        allow_pickle=True,
    ).item()
    clip_dir = find_clip_dir(aoe_root, ep_id, dataset_name=dataset_name)
    hands_path = _find_hands_npz(clip_dir)
    src_video_path = _find_undist_video(clip_dir)
    if hands_path is None or src_video_path is None:
        sys.exit(f"[verify] could not locate hands.npz or undistorted .mp4 under "
                 f"{clip_dir}")
    hands = np.load(hands_path)

    print("[verify] === stats ===")
    print(json.dumps(episode_stats(ep), indent=2, default=str))

    print("[verify] === numerical cam-space self-consistency ===")
    nc = numerical_check(ep, hands)
    for k, v in nc.items():
        print(f"  {k} = {v:.4e}")
    pass_thr = nc["rot_err_max"] < 1e-3 and nc["trans_err_max"] < 5e-3
    print("  PASS" if pass_thr else "  FAIL — coordinate convention mismatch")

    print("[verify] === projection overlay ===")
    projection_overlay(ep, src_video_path, out_dir, ep_id, num_frames=args.num_frames)
    print(f"[verify] visually inspect {out_dir} — green dots = right hand wrist/palm, "
          f"blue dots = left hand. Dots must land ON the hand for conversion to be correct.")


if __name__ == "__main__":
    import sys
    main()

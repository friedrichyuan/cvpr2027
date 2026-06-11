"""Convert AoE ego-centric datasets to VITRA Human-VLA episode_info format.

Source data layout (AoE standard):

    clip_dir/
    ├── raw_video.mp4
    ├── video_info.json
    ├── ego_annotation/ego_action_annotation.json
    └── ego_process/
        ├── ego_hands_reconstruction/hands.npz
        └── ego_undistorted_video/{raw_video_undistorted.mp4, undistorted_video_info.json}

    Discovered recursively under --root_dir by scanning for hands.npz files.

For each clip we read the per-clip MANO reconstruction once, then SLICE it
into one episode per ``atomic_action`` segment in the action JSON.  Each
episode gets its own **physically sliced** .mp4 video (via ffmpeg or
OpenCV re-encode) so that ``video_decode_frame = np.arange(0, T_seg)`` —
no seeking into long videos required at training time.  This avoids
OpenCV seek reliability issues on H.264 long-GOP videos.

Hand assignment: when the source JSON provides a ``hand`` field per
atomic_action we trust it directly; "both" duplicates the description to
both hands and defaults ``anno_type`` to right.  Otherwise we fall back
to a keyword + validity-ratio heuristic.

Usage example::

    python convert_aoe_to_vitra.py \\
        --root_dir /path/to/aoe_data \\
        --out_root /path/to/vitra_data \\
        --dataset_name aoe \\
        --mano_dir /path/to/mano_weights

See the accompanying README.md for full instructions.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import smplx


# MANO J16 → 21-keypoint mapping for non-fingertip joints.
J16_TO_J21 = {
    0: 0,    # wrist
    1: 1,    # thumb_CMC
    2: 2,    # thumb_MCP
    3: 3,    # thumb_IP
    4: 5,    # index_MCP
    5: 6,    # index_PIP
    6: 7,    # index_DIP
    7: 9,    # middle_MCP
    8: 10,   # middle_PIP
    9: 11,   # middle_DIP
    10: 13,  # ring_MCP
    11: 14,  # ring_PIP
    12: 15,  # ring_DIP
    13: 17,  # pinky_MCP
    14: 18,  # pinky_PIP
    15: 19,  # pinky_DIP
}


def vertex_based_fingertips(joints16: np.ndarray, vertices: np.ndarray, j21: np.ndarray) -> None:
    """Fill the 5 fingertip slots of `j21` using MANO mesh vertices.

    For each finger, take the vertex farthest from the wrist along the
    wrist→DIP direction, restricted to vertices near the DIP joint. Mirrors the
    algorithm used in EgoVerse/egomimic/scripts/ant_aoe_process/
    ant_aoe_retarget_to_g1.py.
    """
    T = joints16.shape[0]
    wrist = joints16[:, 0, :]                                     # (T, 3)
    finger_info = [
        (2, 3, 4),      # thumb (MCP→IP)
        (5, 6, 8),      # index
        (8, 9, 12),     # middle
        (11, 12, 16),   # ring
        (14, 15, 20),   # pinky
    ]
    for pip_idx, dip_idx, tip_kp in finger_info:
        dip = joints16[:, dip_idx, :]
        pip = joints16[:, pip_idx, :]
        direction = dip - wrist
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
        pip_dip_dist = np.linalg.norm(dip - pip, axis=1)          # (T,)
        v_from_wrist = vertices - wrist[:, None, :]
        proj = np.einsum("tvi,ti->tv", v_from_wrist, direction)
        dist_from_dip = np.linalg.norm(vertices - dip[:, None, :], axis=2)
        close = dist_from_dip < pip_dip_dist[:, None] * 3.0
        masked_proj = np.where(close, proj, -np.inf)
        best = np.argmax(masked_proj, axis=1)
        j21[:, tip_kp] = vertices[np.arange(T), best]


def joints16_and_verts_to_j21(joints16: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    T = joints16.shape[0]
    j21 = np.zeros((T, 21, 3), dtype=np.float32)
    for j16_idx, j21_idx in J16_TO_J21.items():
        j21[:, j21_idx] = joints16[:, j16_idx]
    vertex_based_fingertips(joints16, vertices, j21)
    return j21


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    flat = aa.reshape(-1, 3)
    mat = R.from_rotvec(flat).as_matrix()
    return mat.reshape(*aa.shape[:-1], 3, 3)


def load_intrinsics(undist_info_path: Path):
    info = json.loads(undist_info_path.read_text())
    cam = info["cameraParams"]
    fx, fy = float(cam["fx_pixels"]), float(cam["fy_pixels"])
    cx, cy = float(cam["cx_pixels"]), float(cam["cy_pixels"])
    W, H = (int(x) for x in cam["resolution"].split("x"))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, (W, H), info


def build_extrinsics(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    T = R_w2c.shape[0]
    M = np.zeros((T, 4, 4), dtype=np.float32)
    M[:, :3, :3] = R_w2c
    M[:, :3, 3] = t_w2c
    M[:, 3, 3] = 1.0
    return M


def _slice_video(src_video: Path, dst_video: Path, start_frame: int, end_frame: int, fps: float = 30.0):
    """Slice [start_frame, end_frame) from src_video into an independent .mp4.

    Uses ffmpeg with re-encoding for frame-accurate cutting.  The input-seeking
    (`-ss` before `-i`) + output duration (`-frames:v`) guarantees exact frame
    count regardless of GOP structure.

    Falls back to OpenCV if ffmpeg is unavailable.
    """
    n_frames = end_frame - start_frame
    start_sec = start_frame / fps

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_sec:.6f}",
                "-i", str(src_video),
                "-frames:v", str(n_frames),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17",
                "-an",
                str(dst_video),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and dst_video.exists() and dst_video.stat().st_size > 0:
            # Verify frame count
            import cv2
            cap = cv2.VideoCapture(str(dst_video))
            actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if actual == n_frames:
                return
            # Frame count mismatch — fall through to OpenCV
            dst_video.unlink(missing_ok=True)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: sequential read + write with OpenCV (always frame-accurate)
    import cv2
    cap = cv2.VideoCapture(str(src_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_video), fourcc, real_fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    cap.release()


# ---------------------------------------------------------------------------
# Layout-aware path resolution + clip discovery
# ---------------------------------------------------------------------------

def _resolve_clip_paths(clip_dir: Path) -> Optional[dict]:
    """Return paths-to-files for a clip in AoE layout, or None if not a clip.

    Uses the presence of ``ego_process/ego_hands_reconstruction/hands.npz``
    as the "is a valid clip" marker.
    """
    hands_path = clip_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    if not hands_path.exists():
        return None
    return {
        "hands": hands_path,
        "undist_dir": clip_dir / "ego_process" / "ego_undistorted_video",
        "action_json": clip_dir / "ego_annotation" / "ego_action_annotation.json",
    }


def discover_clips(root_dir: Path, max_depth: int = 10):
    """Find every directory under ``root_dir`` that contains a usable AoE clip.

    Scans for ``ego_process/ego_hands_reconstruction/hands.npz`` and walks up
    to the clip root directory. Caps recursion depth to avoid traversing huge
    subtrees with no hand reconstructions.
    """
    seen = set()
    clip_dirs = []
    root = root_dir.resolve()
    for hands_path in root.rglob("hands.npz"):
        try:
            rel = hands_path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) > max_depth:
            continue
        # AoE layout: .../<clip>/ego_process/ego_hands_reconstruction/hands.npz
        if "ego_process" in hands_path.parts:
            clip_dir = hands_path.parent.parent.parent
        else:
            continue
        key = str(clip_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        clip_dirs.append(clip_dir)
    return sorted(clip_dirs)


# ---------------------------------------------------------------------------
# Atomic-action normalisation (handles all three JSON variants)
# ---------------------------------------------------------------------------

def _normalize_segments(action_anno, T: int):
    """Normalise ant action JSON to a list of segments::

        {start, end, desc, hand}

    `hand` is one of {"left", "right", "both", None}.  `None` means the
    source didn't provide a hand label and we'll fall back to a heuristic.

    Variants handled:
      - V2 qwen: top-level dict with `atomic_actions` (no per-segment hand,
        spans the whole clip)
      - V2 homestay/hema/crowdsouring: list of segments with start/end_frame
        and atomic_action[*] (no hand field)
      - agibot: list of segments with start/end_frame and atomic_action[*]
        that DOES have a `hand` field
    """
    segs = []

    def _first_desc(items):
        if not items or not isinstance(items[0], dict):
            return None, None
        a = items[0]
        d = (a.get("description") or "").strip()
        if not d:
            v = (a.get("verb") or "").strip()
            o = (a.get("object") or "").strip()
            d = f"{v} {o}".strip()
        if not d:
            return None, None
        if not d.endswith("."):
            d = d + "."
        hand = a.get("hand")
        if hand not in ("left", "right", "both"):
            hand = None
        return d, hand

    if isinstance(action_anno, dict):
        items = action_anno.get("atomic_actions") or []
        d, hand = _first_desc(items)
        if d:
            segs.append({"start": 0, "end": T, "desc": d, "hand": hand})
    elif isinstance(action_anno, list):
        for seg in action_anno:
            if not isinstance(seg, dict):
                continue
            try:
                s = int(seg.get("start_frame", -1))
                e = int(seg.get("end_frame", -1)) + 1
            except (TypeError, ValueError):
                continue
            s = max(0, min(s, T))
            e = max(0, min(e, T))
            if s >= e:
                continue
            items = seg.get("atomic_action") or seg.get("atomic_actions") or []
            d, hand = _first_desc(items)
            if not d:
                continue
            segs.append({"start": s, "end": e, "desc": d, "hand": hand})
    return segs


def _heuristic_hand(desc: str, l_ratio: float, r_ratio: float) -> str:
    """Fallback hand picker for sources without an explicit `hand` field."""
    low = desc.lower()
    if "left hand" in low or "with the left" in low or " left-hand" in low:
        return "left"
    if "right hand" in low or "with the right" in low or " right-hand" in low:
        return "right"
    if r_ratio - l_ratio > 0.2:
        return "right"
    if l_ratio - r_ratio > 0.2:
        return "left"
    return "right"


# ---------------------------------------------------------------------------
# MANO FK (unchanged from the previous commit)
# ---------------------------------------------------------------------------

class ManoFK:
    def __init__(self, mano_dir: str, device: str = "cpu"):
        self.device = device
        self.right = smplx.MANOLayer(
            model_path=mano_dir, use_pca=False, is_rhand=True, flat_hand_mean=False,
        ).to(device).eval()
        self.left = smplx.MANOLayer(
            model_path=mano_dir, use_pca=False, is_rhand=False, flat_hand_mean=False,
        ).to(device).eval()
        self.left.shapedirs[:, 0, :] *= -1

    @torch.no_grad()
    def fk(self, betas, global_orient_aa, hand_pose_aa, transl, is_left: bool):
        T = global_orient_aa.shape[0]
        global_orient_rm = R.from_rotvec(global_orient_aa.reshape(-1, 3)).as_matrix()
        global_orient_rm = global_orient_rm.reshape(T, 1, 3, 3)
        hand_pose_rm = R.from_rotvec(hand_pose_aa.reshape(-1, 3)).as_matrix()
        hand_pose_rm = hand_pose_rm.reshape(T, 15, 3, 3)
        gt = torch.from_numpy(global_orient_rm).float().to(self.device)
        pt = torch.from_numpy(hand_pose_rm).float().to(self.device)
        bt = torch.from_numpy(betas).float().to(self.device)
        tt = torch.from_numpy(transl).float().to(self.device)
        model = self.left if is_left else self.right
        out = model(global_orient=gt, hand_pose=pt, betas=bt, transl=tt, pose2rot=False)
        joints16 = out.joints.cpu().numpy()
        vertices = out.vertices.cpu().numpy()
        return joints16_and_verts_to_j21(joints16, vertices), joints16


# ---------------------------------------------------------------------------
# Per-clip → many episodes
# ---------------------------------------------------------------------------

def _clip_id_from(root: Path, clip_dir: Path) -> str:
    """Build a flat, filesystem-safe ID from the path of the clip relative to root."""
    rel = clip_dir.resolve().relative_to(root.resolve())
    return "__".join(rel.parts) or clip_dir.name


def convert_one_clip(
    clip_dir: Path,
    clip_id: str,
    dataset_name: str,
    out_video_dir: Path,
    out_anno_dir: Path,
    mano_fk: ManoFK,
    verify: bool = False,
):
    """Convert one clip → potentially MANY episodes (one per atomic_action segment).

    Returns: list of (episode_id, T_segment).
    """
    paths = _resolve_clip_paths(clip_dir)
    if paths is None:
        return []

    hands = np.load(paths["hands"])
    pred_rot = hands["pred_rot"]
    pred_trans = hands["pred_trans"]
    pred_hand_pose = hands["pred_hand_pose"]
    pred_betas = hands["pred_betas"]
    # pred_valid is bool (V2) or float32 with 0./1. (agibot); normalise to bool
    pred_valid = np.asarray(hands["pred_valid"]) > 0.5
    R_w2c = hands["R_w2c"].astype(np.float32)
    t_w2c = hands["t_w2c"].astype(np.float32)
    T = R_w2c.shape[0]

    undist_info_files = list(paths["undist_dir"].glob("*.json"))
    if not undist_info_files:
        return []
    K, (W_orig, H_orig), _ = load_intrinsics(undist_info_files[0])
    video_files = sorted(paths["undist_dir"].glob("*.mp4"))
    if not video_files:
        return []
    src_video = video_files[0]

    if paths["action_json"].exists():
        action_anno = json.loads(paths["action_json"].read_text())
    else:
        action_anno = []
    raw_segments = _normalize_segments(action_anno, T)
    if not raw_segments:
        return []

    # Resolve hand for each segment (use source label if present, otherwise heuristic)
    segments = []
    for seg in raw_segments:
        s, e = seg["start"], seg["end"]
        desc, hand = seg["desc"], seg["hand"]
        if hand is None:
            l_ratio = float(pred_valid[0, s:e].mean()) if e > s else 0.0
            r_ratio = float(pred_valid[1, s:e].mean()) if e > s else 0.0
            hand = _heuristic_hand(desc, l_ratio, r_ratio)
        segments.append({"start": s, "end": e, "desc": desc, "hand": hand})

    # Run MANO FK ONCE per clip (full T frames); slice per-episode below.
    #
    # IMPORTANT — two non-obvious conventions VITRA imposes on the stored fields,
    # both of which differ from HaWoR's raw output:
    #
    #   (1) `transl_worldspace` is the J0 (wrist joint) position in world space,
    #       NOT MANO root translation. VITRA's visualize_core renders mesh as
    #       `R @ (V - J0_canonical) + transl_worldspace`, which requires
    #       transl_worldspace == J0_world. HaWoR's pred_trans is the MANO root
    #       translation, off by ~9.6cm (J0_canonical magnitude in local frame).
    #       We use joints_worldspace[:, 0] from our FK output.
    #
    #   (2) Left-hand `hand_pose` is stored in MANO_RIGHT convention — i.e. the
    #       axis-angle that, when fed into MANO_RIGHT and then X-mirrored, yields
    #       the true left-hand mesh. The mirror relation is
    #       `(rx, ry, rz) → (rx, -ry, -rz)`. HaWoR outputs MANO_LEFT-style
    #       axis-angle which would render with bent fingers under VITRA's
    #       MANO_RIGHT + X-flip pipeline. (VITRA's data.md L212/L216:
    #       "based on the MANO_RIGHT model")
    #
    # global_orient_worldspace and beta need no per-side adjustment in either
    # convention — the X-mirror cancels out for the global rotation, and beta
    # is a PCA coefficient that's invariant under X-flip.
    full_sides = {}
    for hand_idx, side_name in {0: "left", 1: "right"}.items():
        is_left = side_name == "left"
        joints_w, _ = mano_fk.fk(
            pred_betas[hand_idx].astype(np.float32),
            pred_rot[hand_idx].astype(np.float32),
            pred_hand_pose[hand_idx].astype(np.float32),
            pred_trans[hand_idx].astype(np.float32),
            is_left=is_left,
        )
        global_orient_world = axis_angle_to_matrix(pred_rot[hand_idx])
        # (2) X-mirror hand_pose axis-angle for left hand BEFORE converting to rotmat
        hand_pose_aa = pred_hand_pose[hand_idx].reshape(T, 15, 3).copy()
        if is_left:
            hand_pose_aa[..., 1] *= -1
            hand_pose_aa[..., 2] *= -1
        hand_pose_mat = axis_angle_to_matrix(hand_pose_aa)
        global_orient_cam = np.einsum("tij,tjk->tik", R_w2c, global_orient_world)
        # (1) transl_worldspace = J0_world (from FK), not pred_trans
        transl_world_J0 = joints_w[:, 0, :].astype(np.float32)
        full_sides[side_name] = {
            "joints_worldspace": joints_w.astype(np.float32),
            "global_orient_world": global_orient_world.astype(np.float32),
            "global_orient_cam": global_orient_cam.astype(np.float32),
            "hand_pose_mat": hand_pose_mat.astype(np.float32),
            "transl_world": transl_world_J0,
            "kept": pred_valid[hand_idx].astype(np.int64),
            "betas_per_frame": pred_betas[hand_idx].astype(np.float32),
        }

    if verify:
        verify_cam_space(hands, pred_rot, pred_trans, pred_valid, R_w2c, t_w2c, clip_id)

    # Episode_id prefix matches `episode_id.split('_')[0]` rule used by
    # human_dataset.py to determine dataset prefix.

    episodes = []  # list of (episode_id, T_seg)
    for ep_idx, seg in enumerate(segments):
        s, e = seg["start"], seg["end"]
        primary_hand = "right" if seg["hand"] == "both" else seg["hand"]
        T_seg = e - s
        if T_seg <= 1:
            continue

        episode_id = f"{dataset_name}_{clip_id}_ep_{ep_idx:06d}"

        # --- Slice video to independent .mp4 per episode ---
        dst_video = out_video_dir / f"{episode_id}.mp4"
        if not dst_video.exists():
            _slice_video(src_video, dst_video, s, e)

        # Slice per-side fields to [s:e]
        side_dicts = {}
        for side_name in ("left", "right"):
            full = full_sides[side_name]
            valid_slice = full["kept"][s:e]
            if valid_slice.any():
                beta_avg = full["betas_per_frame"][s:e][valid_slice.astype(bool)].mean(axis=0)
            else:
                beta_avg = np.zeros(10, dtype=np.float32)
            side_dicts[side_name] = {
                "beta": beta_avg.astype(np.float32),
                "global_orient_camspace": full["global_orient_cam"][s:e],
                "global_orient_worldspace": full["global_orient_world"][s:e],
                "hand_pose": full["hand_pose_mat"][s:e],
                "transl_camspace": np.zeros((T_seg, 3), dtype=np.float32),  # deprecated
                "transl_worldspace": full["transl_world"][s:e],
                "kept_frames": valid_slice,
                "joints_camspace": np.zeros((T_seg, 21, 3), dtype=np.float32),
                "joints_worldspace": full["joints_worldspace"][s:e],
                "wrist": np.zeros(3, dtype=np.float32),
                "max_translation_movement": 0.0,
                "max_wrist_rotation_movement": 0.0,
                "max_finger_joint_angle_movement": 0.0,
            }

        # Build text dict for this episode:
        #   text[primary_hand][0] = (segment_desc, (0, T_seg))            ← VITRA only reads [0]
        #   text[other_hand]      = all OTHER segments overlapping [s, e) whose hand
        #                             is 'other' or 'both', mapped to LOCAL frames
        other_hand = "left" if primary_hand == "right" else "right"
        text = {"left": [], "right": []}
        text[primary_hand].append((seg["desc"], (0, T_seg)))
        if seg["hand"] == "both":
            text[other_hand].append((seg["desc"], (0, T_seg)))

        for other_seg in segments:
            if other_seg is seg:
                continue
            os_, oe = other_seg["start"], other_seg["end"]
            ov_s, ov_e = max(os_, s), min(oe, e)
            if ov_s >= ov_e:
                continue
            local_s, local_e = ov_s - s, ov_e - s
            o_hand = other_seg["hand"]
            if o_hand == other_hand or o_hand == "both":
                text[other_hand].append((other_seg["desc"], (local_s, local_e)))

        # video_decode_frame starts at 0 since we have a sliced video
        epi = {
            "video_clip_id_segment": [s, e],
            "extrinsics": build_extrinsics(R_w2c[s:e], t_w2c[s:e]),
            "intrinsics": K.astype(np.float64),
            "video_decode_frame": np.arange(0, T_seg, dtype=np.int64),
            "video_name": episode_id,
            "avg_speed": 0.0,
            "total_rotvec_degree": 0.0,
            "total_transl_dist": 0.0,
            "anno_type": primary_hand,
            "text": text,
            "text_rephrase": None,
            "left": side_dicts["left"],
            "right": side_dicts["right"],
        }
        out_path = out_anno_dir / f"{episode_id}.npy"
        np.save(out_path, epi, allow_pickle=True)
        episodes.append((episode_id, T_seg))
    return episodes


def verify_cam_space(hands, pred_rot, pred_trans, pred_valid, R_w2c, t_w2c, clip_id):
    """Numerical self-consistency: R_w2c · world == pred_*_cam (≤ 1e-3)."""
    pred_rot_cam = hands["pred_rot_cam"]
    pred_trans_cam = hands["pred_trans_cam"]
    rot_err_max, trans_err_max = 0.0, 0.0
    for hand_idx in (0, 1):
        valid = pred_valid[hand_idx]
        idxs = np.where(valid)[0]
        if len(idxs) == 0:
            continue
        sample = idxs[:: max(1, len(idxs) // 5)][:5]
        for t in sample:
            Rw = axis_angle_to_matrix(pred_rot[hand_idx, t])
            Rcam_pred = R_w2c[t] @ Rw
            Rcam_ant = axis_angle_to_matrix(pred_rot_cam[hand_idx, t])
            rot_err_max = max(rot_err_max, np.linalg.norm(Rcam_pred - Rcam_ant))
            tcam_pred = R_w2c[t] @ pred_trans[hand_idx, t] + t_w2c[t]
            tcam_ant = pred_trans_cam[hand_idx, t]
            trans_err_max = max(trans_err_max, np.linalg.norm(tcam_pred - tcam_ant))
    if rot_err_max > 1e-3 or trans_err_max > 5e-3:
        print(f"[VERIFY-WARN] {clip_id}: rot_err_max={rot_err_max:.4e}, "
              f"trans_err_max={trans_err_max:.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, type=str,
                    help="Root directory containing one or more clips, in either layout")
    ap.add_argument("--dataset_name", default="aoe", type=str,
                    help="VITRA dataset identifier. Default: 'aoe'. The episode_id prefix "
                         "is derived from this and must match the VITRA patch's dataset "
                         "routing in dataset.py / human_dataset.py.")
    ap.add_argument("--out_root", required=True, type=str)
    ap.add_argument("--max_clips", default=-1, type=int,
                    help="limit number of clips; -1 = all")
    ap.add_argument("--mano_dir", default="weights/mano", type=str)
    ap.add_argument("--device", default="cpu", type=str, help="cpu | cuda for MANO FK")
    ap.add_argument("--verify", action="store_true",
                    help="Run numerical cam-space self-consistency check per clip")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    out_root = Path(args.out_root).resolve()
    dataset_name = args.dataset_name

    out_video_dir = out_root / "Video" / dataset_name
    out_anno_dir = out_root / "Annotation" / dataset_name / "episodic_annotations"
    out_video_dir.mkdir(parents=True, exist_ok=True)
    out_anno_dir.mkdir(parents=True, exist_ok=True)

    clip_dirs = discover_clips(root_dir)
    if args.max_clips > 0:
        clip_dirs = clip_dirs[: args.max_clips]
    print(f"[convert] dataset={dataset_name}: {len(clip_dirs)} candidate clips under "
          f"{root_dir} → {out_root}")
    if not clip_dirs:
        sys.exit("no clips discovered")

    mano_fk = ManoFK(args.mano_dir, device=args.device)

    successes = []
    n_clips_ok = n_clips_skip = n_clips_err = 0
    for clip_dir in tqdm(clip_dirs, desc="convert"):
        clip_id = _clip_id_from(root_dir, clip_dir)
        try:
            ep_list = convert_one_clip(
                clip_dir, clip_id, dataset_name,
                out_video_dir, out_anno_dir, mano_fk, verify=args.verify,
            )
        except Exception as e:
            n_clips_err += 1
            print(f"[ERR] {clip_dir}: {e}")
            traceback.print_exc()
            continue
        if not ep_list:
            n_clips_skip += 1
            continue
        n_clips_ok += 1
        successes.extend(ep_list)

    print(f"[convert] clips: ok={n_clips_ok} skip={n_clips_skip} err={n_clips_err}; "
          f"total episodes={len(successes)}")
    if not successes:
        sys.exit("no episodes converted; aborting index build")

    index_to_episode_id = []
    index_frame_pair = []
    for ep_idx, (ep_id, T) in enumerate(successes):
        index_to_episode_id.append(ep_id)
        for f in range(T):
            index_frame_pair.append((ep_idx, f))
    index_frame_pair = np.asarray(index_frame_pair, dtype=np.int64)
    index_to_episode_id = np.asarray(index_to_episode_id, dtype=object)

    idx_path = out_root / "Annotation" / dataset_name / "episode_frame_index.npz"
    np.savez(idx_path,
             index_frame_pair=index_frame_pair,
             index_to_episode_id=index_to_episode_id)
    print(f"[convert] wrote {idx_path}: {len(index_frame_pair)} frames over "
          f"{len(index_to_episode_id)} episodes")


if __name__ == "__main__":
    main()

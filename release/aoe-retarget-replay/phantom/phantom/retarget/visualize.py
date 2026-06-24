# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""MuJoCo visualization for retargeted G1 trajectories.

Streaming MP4 writer avoids holding all frames in memory at once,
supporting long segments (5000+ frames).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import mujoco
import numpy as np

from phantom.constants import (
    FPS,
    G1_STANDING_HEIGHT,
)
from phantom.constants.aoe import HANDS_RECON_SUBDIRS, SIDECAR_NPZ_NAME
from phantom.robots import RobotSpec, get_spec

logger = logging.getLogger(__name__)


# ── OpenPose-21 hand bone connectivity ──────────────────────────────────────

_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Fingertip + MCP indices for SAM2 seeding
_SEED_KPT_IDS = [4, 8, 12, 16, 20, 5, 9, 13, 17]


# ── Camera intrinsics ───────────────────────────────────────────────────────


def _load_camera_intrinsics(
    episode_dir: Path,
    target_w: int,
    target_h: int,
) -> tuple[float, float, float, float, int, int]:
    """Load camera intrinsics and compute scaled (fx, fy, cx, cy).

    Reads from undistorted_video_info.json. Falls back to hands.npz focal
    with detected video resolution if the JSON is unavailable.

    Returns (fx_scaled, fy_scaled, cx_scaled, cy_scaled, orig_w, orig_h).
    """
    episode_dir = Path(episode_dir)
    fx_orig = fy_orig = cx_orig = cy_orig = None
    orig_w = orig_h = None

    # Try undistorted_video_info.json
    from phantom.constants.aoe import UNDISTORTED_VIDEO_SUBDIRS
    for sub in UNDISTORTED_VIDEO_SUBDIRS:
        info_path = episode_dir / sub / "undistorted_video_info.json"
        if not info_path.exists():
            continue
        try:
            with open(info_path) as f:
                info = json.load(f)
            cp = info.get("cameraParams", {})
            fx_orig = cp.get("fx_pixels")
            fy_orig = cp.get("fy_pixels")
            cx_orig = cp.get("cx_pixels")
            cy_orig = cp.get("cy_pixels")
            res_str = cp.get("resolution", "")
            if "x" in res_str:
                parts = res_str.split("x")
                orig_w, orig_h = int(parts[0]), int(parts[1])
            if fx_orig is not None and orig_w is not None:
                break
        except Exception:
            continue

    # Fallback: detect resolution from video + use hands.npz focal
    if fx_orig is None or orig_w is None:
        for sub in UNDISTORTED_VIDEO_SUBDIRS:
            vdir = episode_dir / sub
            if not vdir.exists():
                continue
            for f in vdir.iterdir():
                if f.suffix == ".mp4":
                    cap = cv2.VideoCapture(str(f))
                    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    break
            if orig_w is not None:
                break

        if orig_w is None:
            orig_w, orig_h = 1920, 1080

        # Get focal from hands.npz
        hands_path = None
        for sub in HANDS_RECON_SUBDIRS:
            hp = episode_dir / sub / "hands.npz"
            if hp.exists():
                hands_path = hp
                break
        if hands_path:
            focal = float(np.load(hands_path)["focal"])
        else:
            focal = 788.0

        if fx_orig is None:
            fx_orig = fy_orig = focal
            cx_orig = orig_w / 2.0
            cy_orig = orig_h / 2.0

    sx = target_w / orig_w
    sy = target_h / orig_h
    return (fx_orig * sx, fy_orig * sy, cx_orig * sx, cy_orig * sy, orig_w, orig_h)


def _project_3d_to_pixel(
    pts_3d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) cam-local points to pixel coords.

    Returns (pixels (N, 2), in_frame (N,) bool).
    """
    N = pts_3d.shape[0]
    pix = np.zeros((N, 2), dtype=np.float32)
    in_frame = np.zeros(N, dtype=bool)

    z = pts_3d[:, 2]
    valid_z = z > 0.05
    if valid_z.any():
        u = fx * pts_3d[valid_z, 0] / z[valid_z] + cx
        v = fy * pts_3d[valid_z, 1] / z[valid_z] + cy
        pix[valid_z, 0] = u
        pix[valid_z, 1] = v
        in_frame[valid_z] = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    return pix, in_frame


# ── Helpers ─────────────────────────────────────────────────────────────────


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay a small white-on-black label in the top-left corner."""
    out = img.copy()
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(out, (4, 4), (8 + tw, 8 + th + baseline), (0, 0, 0), thickness=-1)
    cv2.putText(
        out, text, (6, 6 + th),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def _read_first_n_frames(video_path: Path, n: int) -> np.ndarray:
    """Read the FIRST n frames from an MP4 as RGB uint8 (n, H, W, 3).

    Does NOT resample — takes frames 0..n-1 sequentially.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return np.zeros((n, 480, 640, 3), dtype=np.uint8)

    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return np.zeros((n, 480, 640, 3), dtype=np.uint8)

    arr = np.stack(frames)
    if arr.shape[0] < n:
        pad = np.broadcast_to(arr[-1:], (n - arr.shape[0],) + arr.shape[1:]).copy()
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def _make_mp4_writer(out_path: Path, fps: int):
    """Create an imageio MP4 writer."""
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


def _setup_mujoco(spec: RobotSpec, width: int, height: int):
    """Set up MuJoCo model, data, qpos_writer for a spec."""
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    qpos_writer = spec.qpos_writer_factory(model)
    return model, data, qpos_writer


def _set_action(data, qpos_writer, action):
    """Set action into MuJoCo data."""
    data.qpos[:] = 0.0
    data.qpos[2] = G1_STANDING_HEIGHT
    data.qpos[3] = 1.0
    qpos_writer(data.qpos, action)
    mujoco.mj_forward(data.model, data)


def _make_camera(azimuth: float, elevation: float, distance: float,
                 lookat: tuple = (0.0, 0.0, 1.1)) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat[:] = list(lookat)
    return cam


# ── Keypoint projection ────────────────────────────────────────────────────


def _load_keypoint_projections(
    episode_dir: Path | None,
    T: int,
    width: int,
    height: int,
) -> dict | None:
    """Preload and project all MANO keypoints to pixel coordinates.

    Returns dict with 'left_pix' (T,21,2), 'right_pix' (T,21,2),
    'left_valid' (T,), 'right_valid' (T,), 'left_in_frame' (T,21) bool,
    'right_in_frame' (T,21) bool. Returns None if data unavailable.
    """
    if episode_dir is None:
        return None

    episode_dir = Path(episode_dir)
    hands_dir = None
    for sub in HANDS_RECON_SUBDIRS:
        candidate = episode_dir / sub
        if candidate.exists():
            hands_dir = candidate
            break
    if hands_dir is None:
        return None

    sidecar_path = hands_dir / SIDECAR_NPZ_NAME
    if not sidecar_path.exists():
        return None

    sidecar = np.load(sidecar_path, allow_pickle=True)

    fx, fy, cx, cy, _, _ = _load_camera_intrinsics(episode_dir, width, height)

    result = {}
    for side in ("left", "right"):
        kpts_3d = sidecar[f"{side}_keypoints_cam"]  # (N, 21, 3)
        valid = sidecar[f"{side}_valid"].astype(bool)

        n = min(T, kpts_3d.shape[0], len(valid))
        pix = np.zeros((T, 21, 2), dtype=np.float32)
        in_frame = np.zeros((T, 21), dtype=bool)
        padded_valid = np.zeros(T, dtype=bool)
        padded_valid[:n] = valid[:n]

        for t in range(n):
            if not valid[t]:
                continue
            p, inf = _project_3d_to_pixel(kpts_3d[t], fx, fy, cx, cy, width, height)
            pix[t] = p
            in_frame[t] = inf

        result[f"{side}_pix"] = pix
        result[f"{side}_valid"] = padded_valid
        result[f"{side}_in_frame"] = in_frame

    return result


def _draw_keypoints_on_frame(
    ego_frame: np.ndarray,
    kpt_proj: dict | None,
    t: int,
) -> np.ndarray:
    """Draw projected MANO keypoints + bones on ego frame."""
    vis = ego_frame.copy()
    if kpt_proj is None:
        return vis

    H, W = vis.shape[:2]
    colors = {"left": (255, 140, 0), "right": (0, 200, 80)}

    for side in ("left", "right"):
        valid = kpt_proj[f"{side}_valid"]
        if t >= len(valid) or not valid[t]:
            continue
        pix = kpt_proj[f"{side}_pix"][t]
        inf = kpt_proj[f"{side}_in_frame"][t]
        color = colors[side]

        for i, j in _HAND_BONES:
            if inf[i] and inf[j]:
                p1 = (int(pix[i, 0]), int(pix[i, 1]))
                p2 = (int(pix[j, 0]), int(pix[j, 1]))
                cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)

        for k in range(21):
            u, v = int(pix[k, 0]), int(pix[k, 1])
            if inf[k]:
                cv2.circle(vis, (u, v), 4, color, -1, cv2.LINE_AA)
                cv2.circle(vis, (u, v), 4, (255, 255, 255), 1, cv2.LINE_AA)
            elif 0 <= u < W + 50 and 0 <= v < H + 50:
                cu = max(0, min(W - 1, u))
                cv_pt = max(0, min(H - 1, v))
                dim_color = tuple(c // 3 for c in color)
                cv2.circle(vis, (cu, cv_pt), 3, dim_color, -1)

    return vis


# ── Main renderers ──────────────────────────────────────────────────────────


def render_trajectory_streaming(
    actions: np.ndarray,
    out_path: str | Path,
    spec: RobotSpec | None = None,
    fps: int = FPS,
    width: int = 640,
    height: int = 480,
    cam_azimuth: float = 135.0,
    cam_elevation: float = -20.0,
    cam_distance: float = 2.2,
    cam_lookat: tuple[float, float, float] = (0.0, 0.0, 1.1),
) -> Path:
    """Render trajectory to MP4 frame-by-frame (constant memory)."""
    if spec is None:
        spec = get_spec("g1_inspire")
    out_path = Path(out_path)

    model, data, qpos_writer = _setup_mujoco(spec, width, height)
    cam = _make_camera(cam_azimuth, cam_elevation, cam_distance, cam_lookat)
    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = _make_mp4_writer(out_path, fps)

    T = actions.shape[0]
    try:
        for t in range(T):
            _set_action(data, qpos_writer, actions[t])
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    logger.info("Wrote MuJoCo render (%s): %s (%d frames)", spec.name, out_path, T)
    return out_path


def render_2x3_video(
    actions: np.ndarray,
    ego_video_path: str | Path,
    out_path: str | Path,
    spec: RobotSpec | None = None,
    fps: int = FPS,
    width: int = 640,
    height: int = 480,
    cam_azimuth_ext: float = 135.0,
    cam_elevation_ext: float = -20.0,
    cam_distance_ext: float = 2.2,
    use_stage3: bool = False,
    episode_dir: str | Path | None = None,
) -> Path:
    """Render 2x3 layout video.

    Layout:
      Row 1: [ego video | MuJoCo ext | MuJoCo front]
      Row 2: [SAM2 mask | E2FGVI inpaint | composed robot]

    If use_stage3=False, row 2 shows MuJoCo front + placeholders.
    """
    if spec is None:
        spec = get_spec("g1_inspire")
    out_path = Path(out_path)
    ego_video_path = Path(ego_video_path)
    T = actions.shape[0]

    ego_arr = _read_first_n_frames(ego_video_path, T)

    model, data, qpos_writer = _setup_mujoco(spec, width, height)
    cam_ext = _make_camera(cam_azimuth_ext, cam_elevation_ext, cam_distance_ext)
    cam_front = _make_camera(180.0, -15.0, 2.0, (0.0, 0.0, 1.0))
    renderer_ext = mujoco.Renderer(model, height=height, width=width)
    renderer_front = mujoco.Renderer(model, height=height, width=width)

    # Derive episode_dir for keypoint overlay + SAM2 seeding
    ep_dir = Path(episode_dir) if episode_dir else None
    if ep_dir is None:
        for parent in ego_video_path.parents:
            if (parent / "ego_process").exists():
                ep_dir = parent
                break

    kpt_proj = _load_keypoint_projections(ep_dir, T, width, height)

    mask_frames = None
    inpaint_frames = None
    composed_frames = None

    if use_stage3:
        ego_resized = np.empty((T, height, width, 3), dtype=np.uint8)
        for t in range(T):
            ego_resized[t] = cv2.resize(ego_arr[t], (width, height),
                                        interpolation=cv2.INTER_AREA)
        mask_frames, inpaint_frames, composed_frames = _run_stage3(
            ego_resized, actions, spec, model, data, qpos_writer, width, height,
            episode_dir=ep_dir,
        )

    writer = _make_mp4_writer(out_path, fps)
    try:
        for t in range(T):
            _set_action(data, qpos_writer, actions[t])

            renderer_ext.update_scene(data, camera=cam_ext)
            ext_frame = renderer_ext.render()

            ego_frame = ego_arr[t]
            if ego_frame.shape[:2] != (height, width):
                ego_frame = cv2.resize(ego_frame, (width, height),
                                       interpolation=cv2.INTER_AREA)

            ego_labeled = _draw_label(ego_frame, "1. Ego Video")
            ext_labeled = _draw_label(ext_frame, f"2. {spec.display_name} (ext)")

            kpt_frame = _draw_keypoints_on_frame(ego_frame, kpt_proj, t)
            kpt_labeled = _draw_label(kpt_frame, "3. Keypoints")

            if use_stage3 and mask_frames is not None:
                mask_vis = _mask_to_rgb(mask_frames[t], ego_frame)
                mask_labeled = _draw_label(mask_vis, "4. SAM2 Mask")
                inpaint_labeled = _draw_label(inpaint_frames[t], "5. E2FGVI Inpaint")
                comp_labeled = _draw_label(composed_frames[t], "6. Composed")
            else:
                renderer_front.update_scene(data, camera=cam_front)
                front_frame = renderer_front.render()
                mask_labeled = _draw_label(front_frame, f"4. {spec.display_name} (front)")
                gray = np.full((height, width, 3), 48, dtype=np.uint8)
                inpaint_labeled = _draw_label(gray.copy(), "5. E2FGVI Inpaint (skip)")
                comp_labeled = _draw_label(gray.copy(), "6. Composed (skip)")

            top = np.concatenate([ego_labeled, ext_labeled, kpt_labeled], axis=1)
            bottom = np.concatenate([mask_labeled, inpaint_labeled, comp_labeled], axis=1)
            composed_frame = np.concatenate([top, bottom], axis=0)
            writer.append_data(composed_frame)
    finally:
        writer.close()
        renderer_ext.close()
        renderer_front.close()

    logger.info("Wrote 2x3 render (%s): %s (%d frames)", spec.name, out_path, T)
    return out_path


def _mask_to_rgb(mask: np.ndarray, ego_frame: np.ndarray) -> np.ndarray:
    """Visualize a boolean mask as red overlay on ego frame."""
    vis = ego_frame.copy()
    if mask.any():
        vis[mask] = (
            vis[mask].astype(np.float32) * 0.4
            + np.array([200, 50, 50], dtype=np.float32) * 0.6
        ).astype(np.uint8)
    return vis


# ── Stage 3: SAM2 + E2FGVI + Compose ───────────────────────────────────────


def _run_stage3(
    ego_arr: np.ndarray,
    actions: np.ndarray,
    spec: RobotSpec,
    model, data, qpos_writer,
    width: int,
    height: int,
    episode_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SAM2 segmentation + E2FGVI inpainting + MuJoCo ego compose.

    Returns:
        (mask_frames (T,H,W) bool,
         inpaint_frames (T,H,W,3) uint8,
         composed_frames (T,H,W,3) uint8)
    """
    import torch
    from phantom.compose import compose, SceneStatsEMA

    T, H, W = ego_arr.shape[:3]

    logger.info("Stage 3.1: SAM2 segmentation (%d frames)...", T)
    mask_frames = _segment_hands_sam2(ego_arr, episode_dir, width, height)
    coverage = mask_frames.astype(float).mean() * 100
    logger.info("Stage 3.1 done. Mask coverage: %.1f%%", coverage)

    logger.info("Stage 3.2: E2FGVI inpainting (%d frames)...", T)
    inpaint_frames = _inpaint_e2fgvi(ego_arr, mask_frames)
    logger.info("Stage 3.2 done.")

    logger.info("Stage 3.3: MuJoCo ego render + compose (%d frames)...", T)

    from phantom.robots.mjcf_patch import (
        EGO_CAMERA_NAME,
        patch_mjcf_local,
    )

    patcher = spec.patch_mjcf or patch_mjcf_local
    patched_xml = Path(patcher(spec.mjcf_path, 70.0))
    ego_model = mujoco.MjModel.from_xml_path(str(patched_xml))
    ego_data = mujoco.MjData(ego_model)
    ego_qpos_writer = spec.qpos_writer_factory(ego_model)

    ego_cam_id = mujoco.mj_name2id(
        ego_model, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAMERA_NAME
    )

    lid = mujoco.mj_name2id(ego_model, mujoco.mjtObj.mjOBJ_BODY,
                            "left_shoulder_pitch_link")
    rid = mujoco.mj_name2id(ego_model, mujoco.mjtObj.mjOBJ_BODY,
                            "right_shoulder_pitch_link")
    ego_data.qpos[:] = 0.0
    ego_data.qpos[2] = G1_STANDING_HEIGHT
    ego_data.qpos[3] = 1.0
    mujoco.mj_forward(ego_model, ego_data)
    shoulder_anchor = 0.5 * (ego_data.xpos[lid] + ego_data.xpos[rid])

    head_names = {"head_link", "logo_link"}
    head_geom_ids = []
    for gid in range(ego_model.ngeom):
        mid = int(ego_model.geom_dataid[gid])
        if mid >= 0:
            mname = mujoco.mj_id2name(ego_model, mujoco.mjtObj.mjOBJ_MESH, mid)
            if mname in head_names:
                head_geom_ids.append(gid)
    head_rgba_backup = ego_model.geom_rgba[head_geom_ids].copy()

    world_geom_ids = set()
    for gid in range(ego_model.ngeom):
        if int(ego_model.geom_bodyid[gid]) == 0:
            world_geom_ids.add(gid)

    renderer_ego = mujoco.Renderer(ego_model, height=height, width=width)
    renderer_seg = mujoco.Renderer(ego_model, height=height, width=width)
    renderer_seg.enable_segmentation_rendering()

    scene_ema = SceneStatsEMA(alpha=0.1)
    composed_frames = np.empty_like(ego_arr)

    cam_offset = np.array([0.12, 0.0, 0.12])

    for t in range(T):
        ego_data.qpos[:] = 0.0
        ego_data.qpos[2] = G1_STANDING_HEIGHT
        ego_data.qpos[3] = 1.0
        ego_qpos_writer(ego_data.qpos, actions[t])
        mujoco.mj_forward(ego_model, ego_data)

        lwrist_id = mujoco.mj_name2id(
            ego_model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        rwrist_id = mujoco.mj_name2id(
            ego_model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        wrist_mid = 0.5 * (ego_data.xpos[lwrist_id] + ego_data.xpos[rwrist_id])

        cam_pos = shoulder_anchor + cam_offset

        fwd = wrist_mid - cam_pos
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm > 1e-6:
            fwd = fwd / fwd_norm
        else:
            fwd = np.array([1.0, 0.0, 0.0])
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
        else:
            right = np.array([0.0, 1.0, 0.0])
        up = np.cross(right, fwd)

        R_cam = np.column_stack([right, up, -fwd])

        from scipy.spatial.transform import Rotation as R_cls
        quat_xyzw = R_cls.from_matrix(R_cam).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                              quat_xyzw[1], quat_xyzw[2]])

        ego_model.cam_pos[ego_cam_id] = cam_pos
        ego_model.cam_quat[ego_cam_id] = quat_wxyz
        ego_data.cam_xpos[ego_cam_id] = cam_pos
        ego_data.cam_xmat[ego_cam_id] = R_cam.flatten()

        ego_model.geom_rgba[head_geom_ids] = [0, 0, 0, 0]

        renderer_ego.update_scene(ego_data, camera=EGO_CAMERA_NAME)
        robot_rgb = renderer_ego.render()

        renderer_seg.update_scene(ego_data, camera=EGO_CAMERA_NAME)
        seg = renderer_seg.render()
        geom_ids = seg[..., 0]
        robot_mask = geom_ids >= 0
        for gid in world_geom_ids:
            robot_mask &= (geom_ids != gid)
        for gid in head_geom_ids:
            robot_mask &= (geom_ids != gid)

        ego_model.geom_rgba[head_geom_ids] = head_rgba_backup

        composed_frames[t] = compose(
            scene_rgb=inpaint_frames[t],
            robot_rgb=robot_rgb,
            robot_mask=robot_mask,
            feather_sigma=1.5,
            harmonize=True,
            shadow=False,
            scene_ema=scene_ema,
        )

    renderer_ego.close()
    renderer_seg.close()

    import shutil, tempfile
    tmpdir = patched_xml.parent
    for candidate in (tmpdir, tmpdir.parent):
        if str(candidate).startswith(tempfile.gettempdir()):
            shutil.rmtree(candidate, ignore_errors=True)
            break

    torch.cuda.empty_cache()
    logger.info("Stage 3.3 done.")

    return mask_frames, inpaint_frames, composed_frames


def _segment_hands_sam2(
    ego_arr: np.ndarray,
    episode_dir: Path | None = None,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Run SAM2 hand segmentation using FK keypoint projections as seeds.

    Simple approach: for each hand, pick the 3 best-spread frames with
    the most in-frame fingertip keypoints. Use those as point prompts.
    """
    from phantom.inpaint import HandArmSegmenter

    T, H, W, _ = ego_arr.shape
    segmenter = HandArmSegmenter()

    if episode_dir is None:
        logger.warning("No episode_dir for SAM2 seeding — mask will be empty")
        return np.zeros((T, H, W), dtype=bool)

    episode_dir = Path(episode_dir)
    hands_dir = None
    for sub in HANDS_RECON_SUBDIRS:
        candidate = episode_dir / sub
        if candidate.exists():
            hands_dir = candidate
            break

    if hands_dir is None:
        logger.warning("No hands_recon_dir found — mask will be empty")
        return np.zeros((T, H, W), dtype=bool)

    sidecar_path = hands_dir / SIDECAR_NPZ_NAME
    if not sidecar_path.exists():
        logger.warning("Sidecar missing — mask will be empty")
        return np.zeros((T, H, W), dtype=bool)

    sidecar = np.load(sidecar_path, allow_pickle=True)

    fx, fy, cx, cy, _, _ = _load_camera_intrinsics(episode_dir, width, height)

    # Per-hand: find best seed frames
    per_hand_seed_frames = {}
    per_hand_seed_points = {}

    for side in ("left", "right"):
        kpts_3d = sidecar[f"{side}_keypoints_cam"]
        valid = sidecar[f"{side}_valid"].astype(bool)
        n_frames = min(T, len(valid), kpts_3d.shape[0])

        candidates = []
        for t in range(n_frames):
            if not valid[t]:
                continue
            # Project fingertip + MCP keypoints
            seeds = []
            for kid in _SEED_KPT_IDS:
                z = kpts_3d[t, kid, 2]
                if z > 0.05:
                    u = fx * kpts_3d[t, kid, 0] / z + cx
                    v = fy * kpts_3d[t, kid, 1] / z + cy
                    if 0 <= u < width and 0 <= v < height:
                        seeds.append([u, v])
            if len(seeds) >= 3:
                candidates.append((len(seeds), t, seeds))

        candidates.sort(key=lambda x: -x[0])

        # Pick up to 3 well-spread frames
        best_frames = []
        seed_pts = {}
        for score, fidx, pts in candidates:
            if all(abs(fidx - bf) > max(T // 10, 5) for bf in best_frames) or not best_frames:
                best_frames.append(fidx)
                seed_pts[fidx] = pts
                if len(best_frames) >= 3:
                    break

        per_hand_seed_frames[side] = best_frames
        per_hand_seed_points[side] = seed_pts
        logger.info("SAM2 %s: %d seed frames %s (best has %d points)",
                    side, len(best_frames), best_frames,
                    len(seed_pts[best_frames[0]]) if best_frames else 0)

    # Run SAM2 inference
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = segmenter._build_inference_state(ego_arr)
        segmenter.predictor.reset_state(state)

        obj_id_map = {"left": 0, "right": 1}

        for side in ("left", "right"):
            oid = obj_id_map[side]
            for fidx in per_hand_seed_frames[side]:
                seeds = per_hand_seed_points[side].get(fidx, [])
                if not seeds:
                    continue
                seed_arr = np.array(seeds, dtype=np.float32)
                segmenter.predictor.add_new_points_or_box(
                    state, frame_idx=int(fidx), obj_id=oid,
                    points=seed_arr,
                    labels=np.ones(len(seed_arr), dtype=np.int32),
                )

        masks = np.zeros((T, H, W), dtype=bool)

        def _collect(reverse: bool):
            for out_idx, _out_obj_ids, out_logits in \
                    segmenter.predictor.propagate_in_video(state, reverse=reverse):
                for i in range(out_logits.shape[0]):
                    m = (out_logits[i] > 0.0).cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    masks[out_idx] |= m

        _collect(reverse=False)
        _collect(reverse=True)

    torch.cuda.empty_cache()
    return masks


def _inpaint_e2fgvi(
    ego_arr: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Run E2FGVI inpainting."""
    from phantom.inpaint_runner import InpaintRunner

    runner = InpaintRunner()
    result = runner.run(
        ego_video=ego_arr,
        arm_mask=masks,
        neighbor_stride=5,
        ref_length=10,
    )
    return result


# Keep backward compat alias
render_2x2_video = render_2x3_video

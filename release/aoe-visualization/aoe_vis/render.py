"""End-to-end per-sample rendering pipeline.

Combines, for every frame:
  * the (undistorted) ego video with an ``AoE`` wordmark,
  * the camera-frame MANO mesh + 21-keypoint skeleton overlay,
  * the future wrist-trajectory trails,
  * an annotation info panel,
  * a world-frame 3D hand panel,
  * a unified top title bar and a bottom annotation timeline scrubber bar.

The camera-frame hand mesh is rendered with a high-quality offscreen OpenGL
renderer (:mod:`aoe_vis.gl_render`) when its GL stack is available, matching the
reference high-quality pipeline's smooth-shaded, lit MANO surface; otherwise it
falls back to the pure-NumPy shaded mesh in :mod:`aoe_vis.mesh`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import gl_render
from . import keypoints as kp
from . import mesh as meshmod
from . import overlays as ov
from . import shading
from . import trajectory as traj
from .sample import SamplePaths

# Defaults baked into the code (the CLI only exposes input/output).
TARGET_HEIGHT = 540
STRIDE = 1
FUTURE_WINDOW = 30
PANEL_FRAC = 0.72
WORLD_FRAC = 0.78
TIMELINE_H = 72


def _fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    return cv2.resize(img, (int(round(w * target_h / h)), target_h),
                      interpolation=cv2.INTER_AREA)


def _transcode_or_move(tmp_path: str, out_path: str, fps: float,
                       transcode_h264: bool) -> None:
    if transcode_h264 and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-r", f"{fps}", "-i", tmp_path,
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-pix_fmt", "yuv420p", out_path]
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode
        if rc == 0 and os.path.exists(out_path):
            os.remove(tmp_path)
            return
    shutil.move(tmp_path, out_path)


class _RenderContext:
    """Bundle of everything needed to compose one frame (shared by video/preview)."""

    def __init__(self, paths: SamplePaths, target_height: int, with_world: bool,
                 with_trails: bool, with_mesh: bool, with_keypoints: bool,
                 future_window: int, fix_left_shapedirs: bool) -> None:
        self.paths = paths
        self.with_world = with_world
        self.with_trails = with_trails
        self.with_mesh = with_mesh
        self.with_keypoints = with_keypoints
        self.future_window = future_window
        self.fix_left_shapedirs = fix_left_shapedirs

        self.annotations = (ov.load_annotations(paths.annotation)
                            if paths.has_annotation() else [])
        self.have_hands = paths.has_hands()
        self.kpd: Dict[str, np.ndarray] = {}
        if self.have_hands:
            print(f"  computing MANO keypoints ({paths.name}) ...")
            self.kpd = kp.compute_keypoints(paths.hands_npz)

        cap = cv2.VideoCapture(paths.base_video(prefer_undistorted=True))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        n_kp = self.kpd["joints_cam"].shape[1] if self.have_hands else self.total
        self.n_frames = min(self.total, n_kp)

        self.out_h = target_height
        self.s = self.out_h / self.vid_h
        self.video_w = int(round(self.vid_w * self.s))
        self.panel_w = int(round(self.out_h * PANEL_FRAC))
        self.world_w = int(round(self.out_h * WORLD_FRAC)) if with_world else 0
        self.header_h = ov.HEADER_H
        self.timeline_h = TIMELINE_H
        self.top_w = self.video_w + self.panel_w + self.world_w
        self.out_size = (self.top_w, self.header_h + self.out_h + self.timeline_h)

        # reference convention: fx=fy=focal, principal at image centre
        focal = self.kpd["focal"] if self.have_hands else self.vid_w
        self.fx = self.fy = focal * self.s
        self.cx = (self.vid_w / 2.0) * self.s
        self.cy = (self.vid_h / 2.0) * self.s

        self.world_view = None
        self.cam_positions = None
        self.cam_poses = None
        if with_world and self.have_hands:
            if os.path.exists(paths.camera_traj):
                cam = np.load(paths.camera_traj, allow_pickle=True)
                if "cam_c2w" in cam.files:
                    self.cam_poses = np.asarray(cam["cam_c2w"])
                    self.cam_positions = self.cam_poses[:, :3, 3]
            self.world_view = traj.WorldView(
                self.kpd["joints_world"], self.kpd["pred_valid"],
                self.cam_positions, self.world_w, self.out_h)

        # high-quality offscreen GL mesh renderer (optional)
        self.gl: Optional[gl_render.GLHandRenderer] = None
        if with_mesh and gl_render.is_available():
            try:
                self.gl = gl_render.GLHandRenderer(self.video_w, self.out_h)
                print("  [mesh] using offscreen OpenGL renderer (high quality)")
            except Exception as exc:  # pragma: no cover - depends on environment
                print(f"  [mesh] GL renderer unavailable, using NumPy mesh ({exc})")
                self.gl = None
        elif with_mesh:
            print(f"  [mesh] GL stack not importable, using NumPy mesh "
                  f"({gl_render.import_error()})")

        # static unified header strip
        sections: List[Tuple[int, str]] = [(0, "Ego View"),
                                            (self.video_w, "Action Annotation")]
        if with_world:
            sections.append((self.video_w + self.panel_w, "World Frame (3D)"))
        self.header = ov.draw_top_header(self.top_w, sections)

    def close(self) -> None:
        if self.gl is not None:
            self.gl.close()

    def compose(self, frame_idx: int, frame_bgr: np.ndarray) -> np.ndarray:
        """Compose the full multi-panel frame for ``frame_idx``."""
        ctx = self
        frame = _fit_height(frame_bgr, ctx.out_h)
        if frame.shape[1] != ctx.video_w:
            frame = cv2.resize(frame, (ctx.video_w, ctx.out_h))

        ann = (ov.annotation_for_frame(ctx.annotations, frame_idx)
               if ctx.annotations else None)

        meshes: Dict[int, tuple] = {}
        if ctx.have_hands:
            if ctx.with_mesh:
                meshes = kp.mesh_world_for_frame(
                    ctx.kpd, frame_idx, fix_left_shapedirs=ctx.fix_left_shapedirs)
                R = ctx.kpd["R_w2c"][frame_idx]
                t = ctx.kpd["t_w2c"][frame_idx]
                gl_hands = []
                for hand_idx, (verts_world, faces) in meshes.items():
                    verts_cam = (R @ verts_world.T).T + t[None, :]
                    if ctx.gl is not None:
                        gl_hands.append((verts_cam, verts_world, faces,
                                         gl_render.hand_color_for(hand_idx)))
                    else:
                        pts2d = kp.project_cam_to_2d(verts_cam, ctx.fx, ctx.fy,
                                                     ctx.cx, ctx.cy)
                        fcols = shading.face_colors_bgr(
                            verts_world, faces, gl_render.hand_color_for(hand_idx))
                        meshmod.draw_mesh(frame, verts_cam, pts2d, faces,
                                          traj.MESH_COLORS[hand_idx], alpha=0.85,
                                          face_colors=fcols)
                if ctx.gl is not None and gl_hands:
                    ctx.gl.render_overlay(frame, gl_hands, ctx.fx, ctx.fy,
                                          ctx.cx, ctx.cy)
            if ctx.with_trails:
                traj.draw_future_wrist_trails(
                    frame, ctx.kpd["joints_world"], ctx.kpd["pred_valid"],
                    ctx.kpd["R_w2c"], ctx.kpd["t_w2c"], frame_idx,
                    ctx.fx, ctx.fy, ctx.cx, ctx.cy,
                    window=ctx.future_window, scale=ctx.s)
            if ctx.with_keypoints:
                kp.draw_keypoints_on_frame(
                    frame, ctx.kpd["joints_cam"], ctx.kpd["pred_valid"],
                    frame_idx, ctx.fx, ctx.fy, ctx.cx, ctx.cy, scale=ctx.s)

        ov.draw_logo(frame)

        panel = ov.draw_info_panel(ctx.panel_w, ctx.out_h, ann, frame_idx,
                                   ctx.n_frames, ctx.fps, ctx.paths.name)
        parts = [frame, panel]
        if ctx.with_world and ctx.world_view is not None:
            parts.append(traj.render_world_panel(
                ctx.world_view, ctx.kpd["joints_world"], ctx.kpd["pred_valid"],
                frame_idx, ctx.cam_positions, cam_poses=ctx.cam_poses,
                meshes=meshes, with_mesh=ctx.with_mesh,
                with_keypoints=ctx.with_keypoints))
        top_row = np.hstack(parts)

        # blue separators between sections (panel|world is the requested one)
        cv2.line(top_row, (ctx.video_w, 0), (ctx.video_w, ctx.out_h),
                 ov.C_OUTLINE, 1, cv2.LINE_AA)
        if ctx.with_world:
            xb = ctx.video_w + ctx.panel_w
            cv2.line(top_row, (xb, 0), (xb, ctx.out_h), ov.C_HEADER, 3, cv2.LINE_AA)

        timeline = ov.draw_timeline_bar(top_row.shape[1], ctx.timeline_h,
                                        ctx.annotations, frame_idx, ctx.n_frames)
        combined = np.vstack([ctx.header, top_row, timeline])
        if combined.shape[1] != ctx.out_size[0] or combined.shape[0] != ctx.out_size[1]:
            combined = cv2.resize(combined, ctx.out_size)
        return combined


def _write_video(ctx: "_RenderContext", out_dir: str, stride: int,
                 transcode_h264: bool) -> str:
    """Render the end-to-end video from ``ctx`` and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "AoE_output_vis.mp4")
    tmp_path = os.path.join(out_dir, "AoE_output_vis.raw.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             ctx.fps / stride, ctx.out_size)
    cap = cv2.VideoCapture(ctx.paths.base_video(prefer_undistorted=True))
    written = 0
    for frame_idx in range(ctx.n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            continue
        writer.write(ctx.compose(frame_idx, frame))
        written += 1
    cap.release()
    writer.release()
    _transcode_or_move(tmp_path, out_path, ctx.fps / stride, transcode_h264)
    print(f"  [ok] end-to-end -> {out_path} ({written} frames, "
          f"{ctx.out_size[0]}x{ctx.out_size[1]})")
    return out_path


def render_end_to_end(paths: SamplePaths, out_dir: str,
                      target_height: int = TARGET_HEIGHT, stride: int = STRIDE,
                      future_window: int = FUTURE_WINDOW,
                      with_world: bool = True, with_trails: bool = True,
                      with_mesh: bool = True, with_keypoints: bool = True,
                      transcode_h264: bool = True,
                      fix_left_shapedirs: bool = True) -> Optional[str]:
    """Render the combined visualization video for one sample.

    Returns the output video path, or ``None`` if prerequisites are missing.
    """
    if paths.base_video(prefer_undistorted=True) is None:
        print(f"  [skip] no base video for {paths.name}")
        return None
    ctx = _RenderContext(paths, target_height, with_world, with_trails,
                         with_mesh, with_keypoints, future_window, fix_left_shapedirs)
    if ctx.total == 0 or ctx.vid_w == 0 or ctx.vid_h == 0:
        print(f"  [skip] unreadable video for {paths.name}")
        ctx.close()
        return None
    out_path = _write_video(ctx, out_dir, stride, transcode_h264)
    ctx.close()
    return out_path


def render_sample(paths: SamplePaths, out_dir: str) -> Optional[str]:
    """Render the single per-sample output (``AoE_output_vis.mp4``).

    Returns the output video path, or ``None`` if prerequisites are missing.
    """
    if paths.base_video(prefer_undistorted=True) is None:
        print(f"  [skip] no base video for {paths.name}")
        return None

    ctx = _RenderContext(paths, TARGET_HEIGHT, True, True, True, True,
                         FUTURE_WINDOW, True)
    if ctx.total == 0 or ctx.vid_w == 0 or ctx.vid_h == 0:
        print(f"  [skip] unreadable video for {paths.name}")
        ctx.close()
        return None

    out_path = _write_video(ctx, out_dir, STRIDE, True)
    ctx.close()
    return out_path

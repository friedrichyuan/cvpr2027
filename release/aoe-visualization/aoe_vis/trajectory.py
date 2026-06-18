"""Wrist trajectory trails and world-frame hand rendering.

Provides two pieces used by the end-to-end visualizer:

* :func:`draw_future_wrist_trails` -- overlays the *next* ``window`` frames of
  each hand's wrist position, projected into the **current** frame's camera, as
  a fading dotted trail (per-hand colour). This previews where the hands are
  about to move.
* :func:`render_world_panel` -- a fast pure-OpenCV orthographic render of both
  hands' shaded MANO meshes + 21-keypoint skeletons in the **world** frame, with
  the camera frustum and path, from a fixed virtual viewpoint.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from . import gl_render
from . import mesh as meshmod
from . import shading
from .keypoints import LEFT, RIGHT, draw_skeleton, project_cam_to_2d
from .overlays import (C_PANEL_BG, C_TEXT_MUTED)

# All per-hand colours derive from the reference MANO mesh colours
# (director-purple = left, director-blue = right) so the hands look identical in
# the camera-frame overlay and this world-frame panel, and the trails match.
MESH_COLORS: Dict[int, Tuple[int, int, int]] = {
    LEFT: gl_render.hand_color_bgr(LEFT),
    RIGHT: gl_render.hand_color_bgr(RIGHT),
}

# Future-trail colour per hand = that hand's mesh colour.
TRAIL_COLORS: Dict[int, Tuple[int, int, int]] = dict(MESH_COLORS)

# Camera-marker geometry (local OpenCV camera frame), matching the reference
# world render: a small 4-sided pyramid with a square base and an apex behind it.
_CAM_MARKER_VERTS = np.array([
    [-0.05, -0.05, 0.0], [0.05, -0.05, 0.0],
    [0.05, 0.05, 0.0], [-0.05, 0.05, 0.0],
    [0.0, 0.0, -0.1],
], dtype=np.float64)
# wireframe edges: base square + apex spokes.
_CAM_MARKER_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
                     (0, 4), (1, 4), (2, 4), (3, 4))
_CAM_MARKER_COLOR = (128, 128, 128)   # gray (matches reference)
_CAM_TRAIL_COLOR = (150, 150, 150)


def draw_future_wrist_trails(frame: np.ndarray, joints_world: np.ndarray,
                             pred_valid: np.ndarray, R_w2c: np.ndarray,
                             t_w2c: np.ndarray, frame_idx: int,
                             fx: float, fy: float, cx: float, cy: float,
                             window: int = 30, scale: float = 1.0) -> np.ndarray:
    """Draw each hand's future wrist path projected into the current camera.

    For the current frame ``t`` it takes wrist world positions at
    ``t+1 .. t+window`` and maps them through frame ``t``'s extrinsics
    (``R_w2c[t]``, ``t_w2c[t]``) before projecting with the intrinsics, so the
    trail is anchored to the present view. Points fade and shrink with horizon.

    Args:
        frame: BGR image to draw on (modified in place).
        joints_world: ``(2, N, 21, 3)`` world joints (wrist = index 0).
        pred_valid: ``(2, N)`` validity mask.
        R_w2c, t_w2c: ``(N,3,3)`` / ``(N,3)`` world->camera extrinsics.
        frame_idx: current frame ``t``.
        fx, fy, cx, cy: intrinsics matching ``frame``.
        window: number of future frames to preview.
        scale: size factor for downscaled frames.
    """
    n_frames = joints_world.shape[1]
    end = min(n_frames, frame_idx + 1 + window)
    if end <= frame_idx + 1:
        return frame

    R = R_w2c[frame_idx]
    t = t_w2c[frame_idx]
    overlay = frame.copy()
    drew_any = False

    for hand_idx in range(2):
        if pred_valid[hand_idx, frame_idx] < 0.5:
            continue
        color = TRAIL_COLORS[hand_idx]
        future_idx = np.arange(frame_idx + 1, end)
        valid_future = future_idx[pred_valid[hand_idx, future_idx] > 0.5]
        if len(valid_future) == 0:
            continue
        wrists_world = joints_world[hand_idx, valid_future, 0, :]   # (M,3)
        wrists_cam = (R @ wrists_world.T).T + t[None, :]            # (M,3)
        pts = project_cam_to_2d(wrists_cam, fx, fy, cx, cy)
        zs = wrists_cam[:, 2]
        horizon = (valid_future - frame_idx).astype(np.float64) / max(window, 1)
        for (u, v), z, hz in zip(pts, zs, horizon):
            if z <= 0:
                continue
            radius = max(1, int(round((4.0 * (1.0 - hz) + 1.0) * scale)))
            cv_pt = (int(u), int(v))
            import cv2
            cv2.circle(overlay, cv_pt, radius, color, -1, cv2.LINE_AA)
            drew_any = True

    if drew_any:
        import cv2
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, dst=frame)
    return frame


def _view_rotation(azim_deg: float, elev_deg: float) -> np.ndarray:
    """Rotation mapping world points into a virtual view frame."""
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    # rotate about world Y (azimuth) then about view X (elevation)
    ca, sa = np.cos(az), np.sin(az)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)
    ce, se = np.cos(el), np.sin(el)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float64)
    return Rx @ Ry


class WorldView:
    """Follow-the-hands orthographic projector for world-frame joints.

    Uses a fixed virtual viewpoint (azimuth/elevation) and a fixed metric scale
    (pixels per metre) chosen so a ``window_m`` cube fits the panel. The view
    centre tracks the current frame's hand centroid so the 3D hand pose stays
    large and centred, while recent wrist trails convey the global motion.
    """

    def __init__(self, joints_world: np.ndarray, pred_valid: np.ndarray,
                 cam_positions: Optional[np.ndarray], width: int, height: int,
                 azim_deg: float = 40.0, elev_deg: float = 20.0,
                 window_m: float = 0.6, margin: float = 0.12) -> None:
        self.width = width
        self.height = height
        self.R = _view_rotation(azim_deg, elev_deg)
        self.joints_world = joints_world
        self.pred_valid = pred_valid

        # global centroid as a fallback when the current frame has no valid hand
        valid = pred_valid > 0.5
        pts = [joints_world[h][valid[h]].reshape(-1, 3)
               for h in range(2) if valid[h].any()]
        self.global_center = (np.concatenate(pts, axis=0).mean(axis=0)
                              if pts else np.zeros(3))
        self._last_center = self.global_center.copy()

        self.scale = (1.0 - 2 * margin) * min(width, height) / max(window_m, 1e-3)
        self.ox = width / 2.0
        self.oy = height / 2.0

    def center_for_frame(self, frame_idx: int) -> np.ndarray:
        """Centroid of the valid wrists at ``frame_idx`` (with fallback)."""
        wr = []
        for h in range(2):
            if self.pred_valid[h, frame_idx] > 0.5:
                wr.append(self.joints_world[h, frame_idx, 0, :])
        if wr:
            self._last_center = np.mean(wr, axis=0)
        return self._last_center

    def project(self, pts_world: np.ndarray, center: np.ndarray) -> np.ndarray:
        """World ``(...,3)`` -> screen ``(...,2)`` around ``center``."""
        p = (self.R @ (pts_world.reshape(-1, 3) - center).T).T
        u = self.ox + self.scale * p[:, 0]
        v = self.oy - self.scale * p[:, 1]
        out = np.stack([u, v], axis=-1)
        return out.reshape(pts_world.shape[:-1] + (2,))


def _draw_camera_frustum(panel: np.ndarray, view: "WorldView",
                         cam_poses: np.ndarray, frame_idx: int,
                         center: np.ndarray) -> None:
    """Draw the current camera as a small pyramid/frustum (world frame).

    Transforms the local camera-marker geometry by the frame's c2w pose into
    world space, then projects it through the panel's virtual viewpoint, so the
    camera is shown as an oriented pyramid rather than a single point -- matching
    the reference world visualization's camera marker.
    """
    import cv2
    pose = cam_poses[frame_idx]
    R = pose[:3, :3]
    t = pose[:3, 3]
    verts_world = (R @ _CAM_MARKER_VERTS.T).T + t[None, :]   # (5,3)
    pts = view.project(verts_world, center).astype(int)
    for a, b in _CAM_MARKER_EDGES:
        cv2.line(panel, tuple(pts[a]), tuple(pts[b]), _CAM_MARKER_COLOR, 2, cv2.LINE_AA)
    apex = tuple(pts[4])
    cv2.putText(panel, "cam", (apex[0] + 6, apex[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)


def render_world_panel(view: WorldView, joints_world: np.ndarray,
                       pred_valid: np.ndarray, frame_idx: int,
                       cam_positions: Optional[np.ndarray] = None,
                       cam_poses: Optional[np.ndarray] = None,
                       trail_len: int = 60,
                       meshes: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
                       with_mesh: bool = True,
                       with_keypoints: bool = True) -> np.ndarray:
    """Render a world-frame mesh/skeleton panel for ``frame_idx`` (pure OpenCV).

    Mirrors the reference world visualization: both hands are drawn as shaded
    MANO meshes in the director-purple / director-blue colours, the camera is a
    small oriented pyramid following its trajectory, and recent wrist paths are
    drawn in each hand's colour.

    Args:
        view: configured :class:`WorldView`.
        joints_world: ``(2, N, 21, 3)`` world joints.
        pred_valid: ``(2, N)`` validity mask.
        frame_idx: frame to render.
        cam_positions: optional ``(N, 3)`` camera centres for the path overlay.
        cam_poses: optional ``(N, 4, 4)`` c2w poses for the camera frustum.
        trail_len: number of past frames in the wrist/camera trails.
        meshes: optional ``{hand_idx: (verts_world, faces)}`` for this frame.
        with_mesh: draw the shaded MANO surface.
        with_keypoints: draw the 21-keypoint skeleton on top.

    Returns a BGR image of size ``(view.height, view.width, 3)`` on a light theme.
    """
    import cv2
    # The section title lives in the shared top header bar; this panel draws only
    # the 3D content (with a light theme background).
    panel = np.full((view.height, view.width, 3), C_PANEL_BG, dtype=np.uint8)

    center = view.center_for_frame(frame_idx)

    # camera trajectory path up to the current frame
    if cam_positions is not None and frame_idx < len(cam_positions):
        cam_screen = view.project(cam_positions[max(0, frame_idx - trail_len): frame_idx + 1],
                                  center)
        for i in range(1, len(cam_screen)):
            cv2.line(panel, tuple(cam_screen[i - 1].astype(int)),
                     tuple(cam_screen[i].astype(int)), _CAM_TRAIL_COLOR, 1, cv2.LINE_AA)

    # current camera as an oriented pyramid/frustum
    if cam_poses is not None and frame_idx < len(cam_poses):
        _draw_camera_frustum(panel, view, cam_poses, frame_idx, center)

    # shaded MANO mesh first, so the skeleton sits on top. Uses the *same*
    # reference world-light shading (baked from world normals) as the camera
    # overlay, so the hands look identical in both views.
    if with_mesh and meshes:
        for hand_idx, (verts_world, faces) in meshes.items():
            p = (view.R @ (verts_world - center).T).T          # view space (ordering)
            pts2d = view.project(verts_world, center)
            fcols = shading.face_colors_bgr(verts_world, faces,
                                            gl_render.hand_color_for(hand_idx))
            meshmod.draw_mesh(panel, p, pts2d, faces, MESH_COLORS[hand_idx],
                              alpha=0.85, require_positive_z=False, ortho=True,
                              face_colors=fcols)

    for hand_idx in range(2):
        base_bgr = MESH_COLORS[hand_idx]
        lo = max(0, frame_idx - trail_len)
        seg = np.arange(lo, frame_idx + 1)
        seg = seg[pred_valid[hand_idx, seg] > 0.5]
        if len(seg) > 1:
            wr = view.project(joints_world[hand_idx, seg, 0, :], center)
            for i in range(1, len(wr)):
                cv2.line(panel, tuple(wr[i - 1].astype(int)),
                         tuple(wr[i].astype(int)), base_bgr, 1, cv2.LINE_AA)

        if not with_keypoints or pred_valid[hand_idx, frame_idx] < 0.5:
            continue
        j2d = view.project(joints_world[hand_idx, frame_idx], center)  # (21,2)
        draw_skeleton(panel, j2d, thickness=1, point_radius=2)
    # scale reference (window size in cm)
    cv2.putText(panel, f"~{int(view.width / view.scale * 100)}cm view",
                (16, view.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                C_TEXT_MUTED, 1, cv2.LINE_AA)
    return panel

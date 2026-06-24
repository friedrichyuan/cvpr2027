# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""MANO -> 21 keypoints conversion, projection and on-frame drawing.

This module follows the reference keypoint convention so the keypoint overlay
produced here matches the delivery convention exactly:

* World joints are obtained by running MANO forward kinematics with the
  *world-frame* parameters (``pred_rot`` / ``pred_trans``) and taking the first
  21 joints in OpenPose order.
* Camera-frame joints are obtained by the rigid transform
  ``joints_cam = R_w2c @ joints_world + t_w2c`` (NOT by re-running MANO with the
  camera-space root, which would introduce a non-linear LBS offset).
* 2D projection uses a plain pinhole model ``u = fx*x/z + cx`` with
  ``fx = fy = focal`` (from ``hands.npz``) and the principal point at the image
  centre.
* The skeleton connectivity, per-finger colours and point styling follow the
  reference keypoint convention.

This implementation uses the vendored pure-NumPy :mod:`aoe_vis.mano_layer`, so it
loads only from inside the release folder.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from . import gl_render
from .mano_layer import HAND_SKELETON, get_mano_layer

# hand index convention: 0 = left, 1 = right.
LEFT, RIGHT = 0, 1

# Per-finger keypoint colours (BGR), following the standard OpenPose/MANO hand
# convention: one bright, distinct colour per finger so the skeleton reads
# clearly over both the video and the shaded mesh. Indexed by ``bone_i // 4`` and
# by ``(joint_i - 1) // 4`` (thumb, index, middle, ring, pinky).
FINGER_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (60, 60, 255),     # thumb  - red
    (0, 165, 255),     # index  - orange
    (40, 220, 40),     # middle - green
    (255, 130, 30),    # ring   - blue
    (230, 70, 230),    # pinky  - magenta
)
WRIST_COLOR: Tuple[int, int, int] = (245, 245, 245)   # near-white wrist joint


def hand_base_bgr(hand_idx: int) -> Tuple[int, int, int]:
    """Per-hand base colour = that hand's MANO mesh colour (BGR); used for labels."""
    return gl_render.hand_color_bgr(hand_idx)


def compute_keypoints(hands_npz_path: str) -> Dict[str, np.ndarray]:
    """Convert a ``hands.npz`` into 21-keypoint world & camera joints.

    Follows the reference keypoint conversion.

    Args:
        hands_npz_path: path to the pipeline ``hands.npz``.

    Returns:
        dict with ``joints_world`` / ``joints_cam`` ``(2, N, 21, 3)``,
        ``pred_valid`` ``(2, N)``, the camera pose arrays and ``focal``.
    """
    data = np.load(hands_npz_path, allow_pickle=True)
    pred_betas = data["pred_betas"]        # (2, N, 10)
    pred_hand_pose = data["pred_hand_pose"]  # (2, N, 45)
    pred_rot = data["pred_rot"]            # (2, N, 3) world
    pred_trans = data["pred_trans"]        # (2, N, 3) world
    pred_valid = data["pred_valid"]        # (2, N)
    R_w2c = data["R_w2c"]                  # (N, 3, 3)
    t_w2c = np.asarray(data["t_w2c"])      # (N, 3) or (N,3,1)
    if t_w2c.ndim == 3:
        t_w2c = t_w2c[..., 0]

    n_frames = pred_betas.shape[1]
    joints_world = np.zeros((2, n_frames, 21, 3), dtype=np.float64)
    for hand_idx, is_rhand in ((LEFT, False), (RIGHT, True)):
        layer = get_mano_layer(is_rhand=is_rhand, fix_left_shapedirs=not is_rhand)
        out = layer.forward(
            pred_betas[hand_idx], pred_rot[hand_idx],
            pred_hand_pose[hand_idx], pred_trans[hand_idx],
            return_verts=False,
        )
        joints_world[hand_idx] = out["joints"]

    joints_cam = np.zeros_like(joints_world)
    for hand_idx in range(2):
        joints_cam[hand_idx] = (
            np.einsum("nij,nkj->nki", R_w2c, joints_world[hand_idx])
            + t_w2c[:, None, :]
        )

    for hand_idx in range(2):
        invalid = pred_valid[hand_idx] < 0.5
        joints_world[hand_idx, invalid] = 0.0
        joints_cam[hand_idx, invalid] = 0.0

    return {
        "joints_world": joints_world.astype(np.float32),
        "joints_cam": joints_cam.astype(np.float32),
        "pred_valid": pred_valid,
        "R_c2w": data["R_c2w"],
        "t_c2w": data["t_c2w"],
        "R_w2c": R_w2c,
        "t_w2c": t_w2c,
        "focal": float(data["focal"]),
        # Raw per-hand parameters kept so the full mesh can be reconstructed
        # on demand (per rendered frame) without re-loading the archive.
        "pred_betas": pred_betas,
        "pred_rot": pred_rot,
        "pred_hand_pose": pred_hand_pose,
        "pred_trans": pred_trans,
    }


def mesh_world_for_frame(kpd: Dict[str, np.ndarray], frame_idx: int,
                         fix_left_shapedirs: bool = True
                         ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Reconstruct both hands' full MANO mesh (world frame) for one frame.

    Args:
        kpd: dict returned by :func:`compute_keypoints`.
        frame_idx: frame to reconstruct.
        fix_left_shapedirs: apply the left-hand shapedirs sign fix.

    Returns:
        ``{hand_idx: (vertices_world (778,3), faces (F,3))}`` for the hands that
        are valid at ``frame_idx`` (invalid hands are omitted).
    """
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    sl = slice(frame_idx, frame_idx + 1)
    for hand_idx, is_rhand in ((LEFT, False), (RIGHT, True)):
        if kpd["pred_valid"][hand_idx, frame_idx] < 0.5:
            continue
        layer = get_mano_layer(is_rhand=is_rhand,
                               fix_left_shapedirs=(not is_rhand) and fix_left_shapedirs)
        res = layer.forward(
            kpd["pred_betas"][hand_idx, sl], kpd["pred_rot"][hand_idx, sl],
            kpd["pred_hand_pose"][hand_idx, sl], kpd["pred_trans"][hand_idx, sl],
            return_verts=True,
        )
        out[hand_idx] = (res["vertices"][0], res["faces"])
    return out


def project_cam_to_2d(joints_cam: np.ndarray, fx: float, fy: float,
                      cx: float, cy: float) -> np.ndarray:
    """Project camera-frame 3D keypoints to 2D pixels (pinhole).

    Args:
        joints_cam: ``(..., 3)`` camera-frame points.
        fx, fy, cx, cy: intrinsics.

    Returns:
        ``(..., 2)`` pixel coordinates.
    """
    x = joints_cam[..., 0]
    y = joints_cam[..., 1]
    z = joints_cam[..., 2]
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    return np.stack([u, v], axis=-1)


def intrinsics_from(focal: float, width: int, height: int) -> Tuple[float, float, float, float]:
    """Return ``(fx, fy, cx, cy)`` following the reference keypoint convention."""
    return focal, focal, width / 2.0, height / 2.0


def draw_skeleton(image: np.ndarray, joints_2d: np.ndarray,
                  thickness: int = 1, point_radius: int = 2,
                  label: Optional[str] = None,
                  label_color: Tuple[int, int, int] = (255, 255, 255),
                  halo: Tuple[int, int, int] = (40, 35, 30)) -> np.ndarray:
    """Draw a 21-keypoint hand skeleton with a vivid per-finger colour scheme.

    Each finger (thumb/index/middle/ring/pinky) is drawn in its own bright
    colour (see :data:`FINGER_COLORS`); the wrist joint is near-white. A thin
    dark halo under the lines/points keeps the skeleton legible over both the
    video and the shaded mesh.

    Args:
        image: BGR image (drawn in place).
        joints_2d: ``(21, 2)`` pixel coordinates in OpenPose order.
        thickness: skeleton line thickness (kept thin).
        point_radius: joint dot radius (wrist drawn slightly larger).
        label: optional short text near the wrist (e.g. ``"L"``/``"R"``).
        label_color: colour of ``label`` (per-hand mesh colour distinguishes L/R).
        halo: dark outline colour drawn under bones/points for contrast.
    """
    import cv2

    # bones: 5 fingers x 4 segments, coloured per finger.
    for bone_i, (start, end) in enumerate(HAND_SKELETON):
        pt1 = tuple(joints_2d[start].astype(int))
        pt2 = tuple(joints_2d[end].astype(int))
        col = FINGER_COLORS[bone_i // 4]
        cv2.line(image, pt1, pt2, halo, thickness + 1, cv2.LINE_AA)   # thin halo
        cv2.line(image, pt1, pt2, col, thickness, cv2.LINE_AA)

    for joint_idx, point in enumerate(joints_2d):
        center = tuple(point.astype(int))
        if joint_idx == 0:
            col = WRIST_COLOR
            radius = point_radius + 1
        else:
            col = FINGER_COLORS[(joint_idx - 1) // 4]
            radius = point_radius
        cv2.circle(image, center, radius + 1, halo, -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, col, -1, cv2.LINE_AA)

    if label:
        wrist = joints_2d[0].astype(int)
        cv2.putText(image, label, (wrist[0] - 10, wrist[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)
    return image


def draw_hand_keypoints(image: np.ndarray, joints_2d: np.ndarray,
                        hand_idx: int, thickness: int = 1,
                        point_radius: int = 2) -> np.ndarray:
    """Draw a single hand's 21-keypoint skeleton on ``image`` in-place.

    Uses the vivid per-finger colour convention; the ``L``/``R`` label is tinted
    with the hand's MANO mesh colour so the two hands remain identifiable.
    """
    label = "L" if hand_idx == LEFT else "R"
    return draw_skeleton(image, joints_2d, thickness=thickness,
                         point_radius=point_radius, label=label,
                         label_color=hand_base_bgr(hand_idx))


def draw_keypoints_on_frame(frame: np.ndarray, joints_cam: np.ndarray,
                            pred_valid: np.ndarray, frame_idx: int,
                            fx: float, fy: float, cx: float, cy: float,
                            scale: float = 1.0) -> np.ndarray:
    """Project + draw both hands' keypoints for ``frame_idx`` onto ``frame``.

    Args:
        frame: image to draw on (modified in place).
        joints_cam: ``(2, N, 21, 3)`` camera-frame joints.
        pred_valid: ``(2, N)`` validity.
        frame_idx: frame index into ``joints_cam``.
        fx, fy, cx, cy: intrinsics matching ``frame`` resolution.
        scale: extra factor to scale line/point sizes for downscaled frames.
    """
    thickness = max(1, int(round(1.4 * scale)))
    radius = max(1, int(round(1.8 * scale)))
    for hand_idx in range(2):
        if pred_valid[hand_idx, frame_idx] < 0.5:
            continue
        joints_2d = project_cam_to_2d(joints_cam[hand_idx, frame_idx], fx, fy, cx, cy)
        draw_hand_keypoints(frame, joints_2d, hand_idx,
                            thickness=thickness, point_radius=radius)
    return frame

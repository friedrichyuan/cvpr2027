# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Lightweight shaded rendering of the MANO triangle mesh.

Both the camera-frame video overlay and the world-frame 3D panel need to draw
the MANO surface (778 vertices / 1538 faces) as a semi-transparent, shaded blob
sitting *under* the keypoint skeleton. This module keeps that purely in NumPy +
OpenCV (no 3D engine):

* faces are sorted back-to-front (painter's algorithm) using their mean depth,
* each face is flat-shaded by the angle between its normal and the view ray,
* triangles are filled on a copy of the frame and alpha-blended on top.

The same routine serves both views: callers pass vertices expressed in a
*view space* whose ``+z`` axis points away from the viewer (camera optical axis
for the overlay; the rotated world axis for the 3D panel) together with the
matching 2D projection of those vertices.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _shade_and_order(verts_view: np.ndarray, faces: np.ndarray, ortho: bool,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-face draw order, shading intensity and min depth.

    Args:
        verts_view: ``(V, 3)`` vertices in view space (``+z`` away from viewer).
        faces: ``(F, 3)`` triangle vertex indices.
        ortho: if True use a fixed orthographic view ray ``(0,0,-1)`` instead of
            the per-face direction towards the (perspective) camera at origin.

    Returns:
        ``order`` ``(F,)`` far-to-near face indices, ``shade`` ``(F,)`` in
        ``[0, 1]`` and ``min_z`` ``(F,)`` nearest-vertex depth per face.
    """
    v0 = verts_view[faces[:, 0]]
    v1 = verts_view[faces[:, 1]]
    v2 = verts_view[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norm, 1e-9, None)

    centroid = (v0 + v1 + v2) / 3.0
    if ortho:
        shade = np.abs(normals[:, 2])
    else:
        view_dir = -centroid / np.clip(
            np.linalg.norm(centroid, axis=1, keepdims=True), 1e-9, None)
        shade = np.abs(np.sum(normals * view_dir, axis=1))
    shade = np.clip(shade, 0.0, 1.0)

    tri_z = centroid[:, 2]
    order = np.argsort(-tri_z)          # far (large +z) first
    min_z = np.minimum(np.minimum(v0[:, 2], v1[:, 2]), v2[:, 2])
    return order, shade, min_z


def draw_mesh(frame: np.ndarray, verts_view: np.ndarray, pts2d: np.ndarray,
              faces: np.ndarray, color: Tuple[int, int, int],
              alpha: float = 0.45, ambient: float = 0.35,
              require_positive_z: bool = True, ortho: bool = False,
              face_colors: Optional[np.ndarray] = None) -> np.ndarray:
    """Alpha-blend a shaded MANO mesh onto ``frame`` (in place).

    Args:
        frame: BGR image to draw on (modified in place).
        verts_view: ``(V, 3)`` vertices in view space (``+z`` away from viewer);
            used only for back-to-front ordering and z-culling.
        pts2d: ``(V, 2)`` projected pixel coordinates of ``verts_view``.
        faces: ``(F, 3)`` triangle indices.
        color: base BGR hand colour (used when ``face_colors`` is None).
        alpha: blend weight of the mesh over the frame.
        ambient: minimum brightness factor so unlit faces stay visible.
        require_positive_z: skip faces with a vertex at/behind the viewer.
        ortho: use an orthographic view ray for shading (world-frame panel).
        face_colors: optional ``(F, 3)`` precomputed BGR colours. When given they
            override the internal shading, so the NumPy path can reuse the exact
            reference (world-light) shading baked elsewhere.

    Returns:
        ``frame`` (for chaining).
    """
    import cv2

    order, shade, min_z = _shade_and_order(verts_view, faces, ortho)
    pts2d_i = np.round(pts2d).astype(np.int32)
    h, w = frame.shape[:2]
    overlay = frame.copy()
    base = np.asarray(color, dtype=np.float64)
    drew = False
    for fi in order:
        tri = faces[fi]
        if require_positive_z and min_z[fi] <= 1e-4:
            continue
        poly = pts2d_i[tri]
        if (poly[:, 0].max() < 0 or poly[:, 0].min() >= w
                or poly[:, 1].max() < 0 or poly[:, 1].min() >= h):
            continue
        if face_colors is not None:
            fc = face_colors[fi]
            c = (int(fc[0]), int(fc[1]), int(fc[2]))
        else:
            intensity = ambient + (1.0 - ambient) * shade[fi]
            c = tuple(int(min(255, base[k] * intensity)) for k in range(3))
        cv2.fillConvexPoly(overlay, poly, c, cv2.LINE_AA)
        drew = True
    if drew:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
    return frame

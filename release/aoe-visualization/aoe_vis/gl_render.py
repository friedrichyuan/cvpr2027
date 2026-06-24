# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""High-quality offscreen OpenGL rendering of the MANO hand mesh.

The reference high-quality visualization path (the non-OpenCV branch of the AoE
processing pipeline) renders the MANO surface with a headless OpenGL renderer:
**smooth-shaded**, lit triangle meshes (no wireframe), one solid colour per hand
(a soft purple for the left hand and a steel blue for the right), composited over
the ego frame through an OpenCV-intrinsics camera.

This module reproduces that link in a self-contained way using an offscreen
``pyrender`` renderer (OpenGL via EGL), so it can run headless on a GPU box while
loading only from inside the release. It exposes:

* :func:`is_available` -- whether the GL stack (pyrender) imported successfully,
* :class:`GLHandRenderer` -- an offscreen renderer that overlays the shaded MANO
  mesh of both hands onto a BGR frame given camera-space vertices + intrinsics.

If the GL stack or a working offscreen context is not available, callers should
fall back to the pure-NumPy painter's-algorithm mesh in :mod:`aoe_vis.mesh`.

Per-hand colours (single source of truth for the whole release)
---------------------------------------------------------------
The reference pipeline assigns one solid colour per hand -- a light purple for
the left hand and a bright blue for the right. The base material colours below
are a lifted (brighter) variant of the upstream ``director-purple`` /
``director-blue`` materials, chosen so that -- after the two-light Lambert
shading (see :mod:`aoe_vis.shading`) -- the rendered hand surface reads as the
brighter, more-vivid look that was selected for the delivery:

* **left  hand = light purple** -> base linear RGB ``(0.867, 0.698, 0.859)``
  == 8-bit RGB ``(221, 178, 219)`` == OpenCV BGR ``(219, 178, 221)`` ;
  shaded palm ~ RGB ``(182, 146, 180)``.
* **right hand = bright blue**  -> base linear RGB ``(0.196, 0.663, 0.898)``
  == 8-bit RGB ``(50, 169, 229)`` == OpenCV BGR ``(229, 169, 50)`` ;
  shaded palm ~ RGB ``(37, 127, 172)``.

(The upstream materials were the darker ``director-purple`` ``(0.804, 0.6, 0.820)``
/ ``director-blue`` ``(0.207, 0.596, 0.792)`` with ambient ``0.2``; the values
here lift those bases for a lighter/more-vivid surface.) Every other module in
this release derives its hand colours from :func:`hand_color_for` /
:func:`hand_color_bgr` so the two hands look identical across the camera-frame
overlay and the world-frame panel.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import shading

# Headless EGL must be selected before pyrender/OpenGL import.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_IMPORT_ERROR: Optional[str] = None
try:  # heavy, optional dependencies -- guarded so the release still runs without them
    import pyrender  # type: ignore
    import trimesh  # type: ignore
    _HAVE_GL = True
except Exception as exc:  # pragma: no cover - depends on environment
    pyrender = None  # type: ignore
    trimesh = None  # type: ignore
    _HAVE_GL = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Reference-renderer hand materials (linear RGB in 0..1). See module docstring
# for the exact 8-bit/BGR equivalents and where they come from. Treated as the
# single source of truth for hand colours across the whole release.
LEFT_COLOR: Tuple[float, float, float] = (0.867, 0.698, 0.859)   # lifted light purple
RIGHT_COLOR: Tuple[float, float, float] = (0.196, 0.663, 0.898)  # lifted bright blue

# Material ambient term used by the reference materials.
REFERENCE_AMBIENT = 0.2

# OpenCV (x right, y down, z forward) -> OpenGL (x right, y up, z back).
_CV_TO_GL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def is_available() -> bool:
    """Return True if the offscreen GL renderer can be imported."""
    return _HAVE_GL


def import_error() -> Optional[str]:
    """Return the GL import error string (or None if the stack imported)."""
    return _IMPORT_ERROR


class GLHandRenderer:
    """Offscreen pyrender renderer for shaded MANO hand meshes.

    One instance owns a single OpenGL context sized to the overlay canvas and is
    reused across frames. Construction raises if no offscreen context can be
    created, so callers can ``try/except`` and fall back to the NumPy mesh.
    """

    def __init__(self, width: int, height: int) -> None:
        if not _HAVE_GL:
            raise RuntimeError(f"GL stack unavailable ({_IMPORT_ERROR})")
        self.width = int(width)
        self.height = int(height)
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=self.width, viewport_height=self.height)

    def close(self) -> None:
        """Release the offscreen GL context."""
        try:
            self._renderer.delete()
        except Exception:
            pass

    def _build_scene(self, fx: float, fy: float, cx: float, cy: float):
        # The reference Lambert shading is baked into per-vertex colours (see
        # :mod:`aoe_vis.shading`), so the scene itself is *unlit*: a full ambient
        # light passes the baked vertex colours straight through, with no extra
        # directional lighting, specular, gamma or tonemapping -- matching the
        # reference renderer's framebuffer output exactly.
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=[1.0, 1.0, 1.0])
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01,
                                           zfar=100.0)
        scene.add(camera, pose=np.eye(4))
        return scene, camera

    def render_overlay(self, frame_bgr: np.ndarray,
                       hands: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                             Tuple[float, float, float]]],
                       fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        """Composite the shaded mesh of each hand onto ``frame_bgr`` (in place).

        Args:
            frame_bgr: ``(H, W, 3)`` BGR canvas (must match this renderer's size).
            hands: sequence of ``(verts_cam (V,3), verts_world (V,3), faces (F,3),
                base_color_rgb)``. ``verts_cam`` are in the OpenCV camera frame
                (used for projection); ``verts_world`` are used only to evaluate
                the view-independent reference shading; ``base_color_rgb`` is the
                hand's linear-RGB material colour in ``0..1``.
            fx, fy, cx, cy: pinhole intrinsics matching the canvas resolution.

        Returns:
            ``frame_bgr`` with the shaded mesh blended over it.
        """
        if not hands:
            return frame_bgr
        import pyrender as _pyrender  # local alias for type checkers

        scene, _ = self._build_scene(fx, fy, cx, cy)
        added = 0
        for verts_cam, verts_world, faces, color in hands:
            if verts_cam is None or len(verts_cam) == 0:
                continue
            faces = np.asarray(faces)
            # bake the exact reference Lambert shading from world-space normals
            normals = shading.vertex_normals(verts_world, faces)
            vcols = shading.shade_rgb(normals, color)            # (V,3) linear RGB
            rgba = np.concatenate([vcols, np.ones((len(vcols), 1))], axis=1)
            rgba = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
            verts_gl = np.asarray(verts_cam, dtype=np.float64) @ _CV_TO_GL.T
            mesh = trimesh.Trimesh(vertices=verts_gl, faces=faces,
                                   vertex_colors=rgba, process=False)
            # No explicit material: pyrender then drives the surface from the
            # trimesh per-vertex colours. Combined with the FLAT flag below this
            # displays the baked colours unlit (no extra lighting / gamma), which
            # is exactly the reference renderer's shaded output.
            scene.add(_pyrender.Mesh.from_trimesh(mesh, smooth=True))
            added += 1
        if added == 0:
            return frame_bgr

        flags = (_pyrender.RenderFlags.RGBA | _pyrender.RenderFlags.SKIP_CULL_FACES
                 | _pyrender.RenderFlags.FLAT)
        color, _ = self._renderer.render(scene, flags=flags)
        color = np.asarray(color)
        rgb = color[:, :, :3].astype(np.float32)
        alpha = (color[:, :, 3:4].astype(np.float32)) / 255.0
        bgr = rgb[:, :, ::-1]
        out = frame_bgr.astype(np.float32) * (1.0 - alpha) + bgr * alpha
        np.copyto(frame_bgr, np.clip(out, 0, 255).astype(np.uint8))
        return frame_bgr


def hand_color_for(hand_idx: int) -> Tuple[float, float, float]:
    """Return the reference base colour (linear RGB) for hand ``0=left/1=right``."""
    return LEFT_COLOR if hand_idx == 0 else RIGHT_COLOR


def hand_color_bgr(hand_idx: int) -> Tuple[int, int, int]:
    """Return the reference hand colour as an 8-bit OpenCV BGR tuple.

    ``0=left`` -> ``(219, 178, 221)`` (light purple),
    ``1=right`` -> ``(229, 169, 50)`` (bright blue).
    """
    r, g, b = hand_color_for(hand_idx)
    return (int(round(b * 255)), int(round(g * 255)), int(round(r * 255)))

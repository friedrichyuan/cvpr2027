# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Faithful reproduction of the reference renderer's lighting/material model.

The reference high-quality hand visualization uses an OpenGL renderer whose
shading is a plain two-light Lambertian model with a flat ambient term and **no**
specular, gamma or tonemapping. The exact constants (read from the reference
renderer) are:

* scene: two white directional lights, each ``strength = 1.0``, at fixed *world*
  positions ``Back = (0, 10, -15)`` and ``Front = (0, 10, 15)`` (y-up), both
  pointing at the origin; global ``ambient_strength = 2.0``;
* material (per hand): ``diffuse = 0.5``, ``ambient = 0.2`` (the ``specular``
  coefficient exists but is unused by the shader);
* fragment shader::

      color = base * (ambient_strength * ambient_coeff
                      + diffuse_coeff * sum_i max(dot(N, L_i), 0))
            = base * (0.4 + 0.5 * (d_back + d_front))

  written straight to the framebuffer (linear, no gamma).

Because the lights live in world space and the hand vertices are expressed in the
world frame, the shading is **view-independent**: we evaluate it once per vertex
from the world-space normals and bake it into per-vertex colours. The same baked
colours are then used by both the camera-frame GL overlay and the world-frame
panel, so the two hands look identical in both views.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# --- exact reference constants -------------------------------------------------
AMBIENT_STRENGTH = 2.0   # scene.ambient_strength
AMBIENT_COEFF = 0.2      # material.ambient (director-purple / director-blue)
DIFFUSE_COEFF = 0.5      # material.diffuse (default)
LIGHT_STRENGTH = 1.0     # per-light strength (both lights)

AMBIENT_TERM = AMBIENT_STRENGTH * AMBIENT_COEFF   # = 0.4


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / (np.linalg.norm(a) + 1e-9)


# Direction from a surface point *towards* each light = normalize(light_position),
# using the reference default y-up light positions.
LIGHT_DIRS = np.stack([
    _unit((0.0, 10.0, -15.0)),   # "Back Light"
    _unit((0.0, 10.0, 15.0)),    # "Front Light"
]) * LIGHT_STRENGTH


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted, outward-oriented per-vertex normals for ``verts``/``faces``."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces)
    vn = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)        # area-weighted face normals
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    lens = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / np.clip(lens, 1e-9, None)
    # orient outward (a hand is roughly star-shaped about its centroid): if the
    # normals globally point inward, flip them so lit faces face the lights.
    outward = verts - verts.mean(axis=0, keepdims=True)
    if np.sum(vn * outward) < 0:
        vn = -vn
    return vn


def shade_factor(normals: np.ndarray) -> np.ndarray:
    """Per-normal brightness multiplier ``0.4 + 0.5*(d_back + d_front)``."""
    n = np.asarray(normals, dtype=np.float64)
    n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)
    diff = np.clip(n @ LIGHT_DIRS.T, 0.0, None).sum(axis=-1)   # sum over 2 lights
    return AMBIENT_TERM + DIFFUSE_COEFF * diff


def shade_rgb(normals: np.ndarray, base_rgb: Tuple[float, float, float]) -> np.ndarray:
    """Bake the reference Lambert shading of ``base_rgb`` (linear RGB 0..1)."""
    base = np.asarray(base_rgb, dtype=np.float64)
    f = shade_factor(normals)[..., None]
    return np.clip(f * base, 0.0, 1.0)


def face_colors_bgr(verts_world: np.ndarray, faces: np.ndarray,
                    base_rgb: Tuple[float, float, float]) -> np.ndarray:
    """Per-face BGR (0..255) colours under the reference shading.

    Used by the NumPy mesh path (fallback overlay and world-frame panel) so it
    matches the GL overlay's baked per-vertex shading.
    """
    faces = np.asarray(faces)
    vcols = shade_rgb(vertex_normals(verts_world, faces), base_rgb)   # (V,3) RGB
    fcols = vcols[faces].mean(axis=1)                                 # (F,3) RGB
    bgr = fcols[:, ::-1] * 255.0
    return bgr

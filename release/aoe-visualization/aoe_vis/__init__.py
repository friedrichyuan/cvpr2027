# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""AoE-Visualization: self-contained visualizer for the Open-AoE delivery.

Modules:
    mano_layer  -- pure-NumPy MANO forward kinematics (vendored model assets)
    keypoints   -- MANO -> 21 OpenPose keypoints, projection, on-frame drawing
    overlays    -- annotation info panel, header bar, bottom timeline scrubber
    mesh        -- pure-NumPy shaded MANO mesh (fallback renderer)
    shading     -- faithful reproduction of the reference lighting/material model
    gl_render   -- high-quality offscreen OpenGL MANO mesh renderer (optional)
    trajectory  -- future wrist trails, world-frame panel, overview plot
    sample      -- per-sample artefact path resolution
    render      -- end-to-end per-sample rendering
"""

__all__ = [
    "mano_layer",
    "keypoints",
    "mesh",
    "shading",
    "gl_render",
    "overlays",
    "trajectory",
    "sample",
    "render",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Resolve the per-sample artefact layout of an AoE delivery directory."""

from __future__ import annotations

import os
from typing import Optional


class SamplePaths:
    """Locate the relevant files inside one sample directory.

    Expected layout (see project CLAUDE.md / dataset README)::

        <sample>/
          raw_video.mp4
          video_info.json
          ego_annotation/ego_action_annotation.json
          ego_process/
            ego_hands_reconstruction/{hands.npz, camera_traj.npz,
                                      visualization/hands_combined.mp4}
            ego_undistorted_video/{raw_video_undistorted.mp4,
                                   undistorted_video_info.json}
    """

    def __init__(self, sample_dir: str) -> None:
        self.sample_dir = os.path.abspath(sample_dir)
        self.name = os.path.basename(self.sample_dir.rstrip("/"))

        j = os.path.join
        self.raw_video = j(self.sample_dir, "raw_video.mp4")
        self.video_info = j(self.sample_dir, "video_info.json")
        self.annotation = j(self.sample_dir, "ego_annotation",
                            "ego_action_annotation.json")

        recon = j(self.sample_dir, "ego_process", "ego_hands_reconstruction")
        self.hands_npz = j(recon, "hands.npz")
        self.camera_traj = j(recon, "camera_traj.npz")
        self.hands_combined = j(recon, "visualization", "hands_combined.mp4")

        undist = j(self.sample_dir, "ego_process", "ego_undistorted_video")
        self.undistorted_video = j(undist, "raw_video_undistorted.mp4")
        self.undistorted_info = j(undist, "undistorted_video_info.json")

    def base_video(self, prefer_undistorted: bool = True) -> Optional[str]:
        """Return the video to draw on.

        The undistorted video is preferred because the camera-frame MANO
        keypoints (projected with the pinhole model) are only geometrically
        correct on the rectified frames.
        """
        if prefer_undistorted and os.path.exists(self.undistorted_video):
            return self.undistorted_video
        if os.path.exists(self.raw_video):
            return self.raw_video
        if os.path.exists(self.undistorted_video):
            return self.undistorted_video
        return None

    def has_annotation(self) -> bool:
        return os.path.exists(self.annotation)

    def has_hands(self) -> bool:
        return os.path.exists(self.hands_npz)

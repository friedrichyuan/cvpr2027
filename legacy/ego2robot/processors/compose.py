from __future__ import annotations

import cv2
import numpy as np

from ..processor import CPU, Artifacts, EpisodeContext, ResourceSpec
from ..video import load_masks, read_rgb_video, write_rgb_video


class ComposeProcessor:
    name = "compose"
    requires = frozenset({Artifacts.INPAINT, Artifacts.ROBOT_RGB, Artifacts.ROBOT_MASK})
    produces = frozenset({Artifacts.COMPOSITE})
    optional_requires = frozenset({Artifacts.DEPTH})
    resources: ResourceSpec = CPU

    def run(self, ctx: EpisodeContext) -> None:
        background, fps = read_rgb_video(ctx.out_dir / Artifacts.INPAINT)
        robot, _ = read_rgb_video(ctx.out_dir / Artifacts.ROBOT_RGB)
        robot_mask = load_masks(ctx.out_dir / Artifacts.ROBOT_MASK)
        if background.shape != robot.shape:
            robot = _resize_video(robot, background.shape[1], background.shape[2])
            robot_mask = _resize_masks(robot_mask, background.shape[1], background.shape[2])
        composite = background.copy()
        composite[robot_mask] = robot[robot_mask]
        write_rgb_video(ctx.out_dir / Artifacts.COMPOSITE, composite, fps)


def _resize_video(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in frames])


def _resize_masks(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool) for mask in masks]
    )

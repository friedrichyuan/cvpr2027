"""Paste the rendered ARX arm onto the inpainted video. The arm is pasted; the gripper uses depth."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from egowhale.media import load_masks, read_rgb, write_rgb
from egowhale.step import COMPOSITE, DEPTH, INPAINT, MASKS, PREFIX, RENDER, Step
from egowhale.visual.render import load_render

_HAND_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class Composite(Step):
    name = "composite"
    needs = (INPAINT, RENDER, DEPTH, MASKS, PREFIX)
    makes = (COMPOSITE,)

    def run(self, src: Path, dst: Path) -> None:
        dst = Path(dst)
        background, fps = read_rgb(dst / INPAINT)
        prefix = np.load(dst / PREFIX)["qpos"]
        robot, robot_mask, gripper_mask, robot_depth = load_render(dst / RENDER)
        frames = min(len(background), len(robot) - len(prefix) + 1)
        background = _hold(background[:frames], len(prefix))
        height, width = background.shape[1], background.shape[2]
        scene_depth = _hold(_match_depth(np.load(dst / DEPTH)["depth"][:frames], height, width), len(prefix))
        hand = _hold(_match_mask(load_masks(dst / MASKS)[:frames], height, width), len(prefix))
        image = _composite(background, robot, robot_mask, gripper_mask, robot_depth, scene_depth, hand)
        write_rgb(dst / COMPOSITE, image, fps)


def _hold(array: np.ndarray, count: int) -> np.ndarray:
    """Keep frame 0 under the approach, then continue from frame 1. Frame 0 is the arrival."""
    return np.concatenate((np.repeat(array[:1], count, axis=0), array[1:]), axis=0)


def _composite(background, robot, robot_mask, gripper_mask, robot_depth, scene_depth, hand):
    count = min(len(background), len(robot), len(scene_depth), len(hand))
    image = background[:count].copy()
    for index in range(count):
        dilated = cv2.dilate(hand[index].astype(np.uint8), _HAND_KERNEL, iterations=1).astype(bool)
        arm = robot_mask[index] & ~gripper_mask[index]
        hidden = gripper_mask[index] & (scene_depth[index] < robot_depth[index]) & ~dilated
        visible = arm | (gripper_mask[index] & ~hidden)
        image[index][visible] = robot[index][visible]
    return image


def _match_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.shape[1:] == (height, width):
        return depth
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth])


def _match_mask(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    if masks.shape[1:] == (height, width):
        return masks
    return np.stack([
        cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        for mask in masks
    ])

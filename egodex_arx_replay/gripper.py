"""Ego2Robot-style conversion from an EgoDex hand skeleton to a gripper pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import EgoDexEpisode
from .geometry import frame_joint_positions

HandSide = Literal["left", "right"]
_SIDES: tuple[HandSide, HandSide] = ("left", "right")
_EPSILON = 1e-8


@dataclass(frozen=True)
class GripperTrajectory:
    """Parallel-gripper reference poses in the fixed ARX scene frame.

    ``rotation[t, side]`` stores the world-space columns ``[approach,
    normal, grasp_axis]``.  ``width`` is the jaw separation inferred directly
    from the thumb and virtual fingertip, in metres.
    """

    position: np.ndarray  # (T, 2, 3)
    rotation: np.ndarray  # (T, 2, 3, 3)
    width: np.ndarray  # (T, 2)
    valid: np.ndarray  # (T, 2)


def hand_to_gripper_pose(
    joint_positions: dict[str, np.ndarray], side: HandSide
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Apply Ego2Robot's hand-to-gripper construction to one hand.

    The virtual fingertip is ``0.7 * index_tip + 0.3 * middle_tip``.  The
    TCP is the midpoint between it and the thumb.  The grasp axis follows the
    thumb-to-virtual-tip line, with the side-dependent sign from Ego2Robot;
    the wrist-to-tip vector resolves the remaining roll degree of freedom.
    """
    required = (
        f"{side}Hand",
        f"{side}ThumbTip",
        f"{side}IndexFingerTip",
        f"{side}MiddleFingerTip",
    )
    if any(name not in joint_positions for name in required):
        return None

    wrist, thumb, index, middle = (joint_positions[name] for name in required)
    virtual_tip = 0.7 * index + 0.3 * middle
    separation = thumb - virtual_tip
    width = float(np.linalg.norm(separation))
    if width < _EPSILON:
        return None

    # Ego2Robot: s=+1 for right and s=-1 for left, making both grippers use a
    # consistent local grasp-axis convention despite mirrored human hands.
    grasp_axis = (1.0 if side == "right" else -1.0) * separation / width
    wrist_to_tip = virtual_tip - wrist
    normal = np.cross(grasp_axis, wrist_to_tip)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < _EPSILON:
        return None
    normal /= normal_norm
    approach = np.cross(normal, grasp_axis)

    rotation = np.column_stack((approach, normal, grasp_axis))
    return 0.5 * (thumb + virtual_tip), rotation, width


def convert_episode_to_grippers(
    episode: EgoDexEpisode, scene_T_egodex: np.ndarray
) -> GripperTrajectory:
    """Convert every tracked frame of an episode using one fixed scene map."""
    frame_count = episode.frame_count
    position = np.full((frame_count, len(_SIDES), 3), np.nan, dtype=np.float64)
    rotation = np.full((frame_count, len(_SIDES), 3, 3), np.nan, dtype=np.float64)
    width = np.full((frame_count, len(_SIDES)), np.nan, dtype=np.float64)
    valid = np.zeros((frame_count, len(_SIDES)), dtype=bool)

    for frame in range(frame_count):
        joints = frame_joint_positions(episode.world_T_joint, frame, scene_T_egodex)
        for side_index, side in enumerate(_SIDES):
            pose = hand_to_gripper_pose(joints, side)
            if pose is None:
                continue
            position[frame, side_index], rotation[frame, side_index], width[frame, side_index] = pose
            valid[frame, side_index] = True

    return GripperTrajectory(position=position, rotation=rotation, width=width, valid=valid)

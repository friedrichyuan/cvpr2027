"""Camera-frame gripper poses from EgoDex world joints."""

from __future__ import annotations

import numpy as np

from egodex_arx_replay.data import EgoDexEpisode
from egodex_arx_replay.gripper import GripperTrajectory, hand_to_gripper_pose

_SIDES = ("left", "right")


def invert_se3(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def convert_episode_to_camera_grippers(episode: EgoDexEpisode) -> GripperTrajectory:
    """TCP / opening / grasp frame in the per-frame camera coordinate system."""
    frames = episode.frame_count
    position = np.full((frames, 2, 3), np.nan, dtype=np.float64)
    rotation = np.full((frames, 2, 3, 3), np.nan, dtype=np.float64)
    width = np.full((frames, 2), np.nan, dtype=np.float64)
    valid = np.zeros((frames, 2), dtype=bool)

    for frame in range(frames):
        joints = {
            name: sequence[frame, :3, 3] for name, sequence in episode.world_T_joint.items()
        }
        camera_from_world = invert_se3(episode.world_T_camera[frame])
        for side_index, side in enumerate(_SIDES):
            pose = hand_to_gripper_pose(joints, side)
            if pose is None:
                continue
            world_position, world_rotation, opening = pose
            position[frame, side_index] = camera_from_world[:3, :3] @ world_position + camera_from_world[:3, 3]
            rotation[frame, side_index] = camera_from_world[:3, :3] @ world_rotation
            width[frame, side_index] = opening
            valid[frame, side_index] = True

    return GripperTrajectory(position=position, rotation=rotation, width=width, valid=valid)

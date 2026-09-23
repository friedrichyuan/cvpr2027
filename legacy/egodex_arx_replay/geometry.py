"""Coordinate mapping and debug geometry for an EgoDex-to-ARX5 replay."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


# EgoDex / Apple world: +X right, +Y up, +Z backwards from the wearer.
# ARX MuJoCo scene:     +X forward, +Y left, +Z up.
# Thus: scene_x = -egodex_z, scene_y = -egodex_x, scene_z = egodex_y.
R_EGODEX_WORLD_TO_ARX = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

# The ARX table top is z=-0.085 in assets/mujoco_arx_scene/scene.xml.  Human
# shoulders are substantially higher than the fixed ARX arm roots.  For the
# stack sample, anchoring the first hip at z=-0.48 puts its initial gripper
# targets 6--12 cm above the z=-0.085 tabletop, instead of 25--40 cm above the
# neutral TCP.  This is one fixed visualization/reference transform, not a
# moving-base optimisation.
DEFAULT_SCENE_ANCHOR = np.array([0.0, 0.0, -0.48], dtype=np.float64)

BODY_BONES: tuple[tuple[str, str], ...] = (
    ("hip", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "spine4"),
    ("spine4", "spine5"),
    ("spine5", "spine6"),
    ("spine6", "spine7"),
    ("spine7", "neck1"),
    ("neck1", "neck2"),
    ("neck2", "neck3"),
    ("neck3", "neck4"),
    ("neck4", "leftShoulder"),
    ("leftShoulder", "leftArm"),
    ("leftArm", "leftForearm"),
    ("leftForearm", "leftHand"),
    ("neck4", "rightShoulder"),
    ("rightShoulder", "rightArm"),
    ("rightArm", "rightForearm"),
    ("rightForearm", "rightHand"),
)


def hand_bones(side: str) -> tuple[tuple[str, str], ...]:
    """Return EgoDex's wrist-to-fingertip chains for one hand."""
    prefix = side
    fingers = ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
    chains: list[tuple[str, str]] = []
    for finger in fingers:
        names = (
            f"{prefix}{finger}Metacarpal",
            f"{prefix}{finger}Knuckle",
            f"{prefix}{finger}IntermediateBase",
            f"{prefix}{finger}IntermediateTip",
            f"{prefix}{finger}Tip",
        )
        chains.append((f"{prefix}Hand", names[0]))
        chains.extend(zip(names[:-1], names[1:]))
    return tuple(chains)


SKELETON_BONES = BODY_BONES + hand_bones("left") + hand_bones("right")


def make_scene_T_egodex(
    world_T_joint: Mapping[str, np.ndarray],
    scene_anchor: np.ndarray = DEFAULT_SCENE_ANCHOR,
    anchor_joint: str = "hip",
) -> np.ndarray:
    """Build one constant EgoDex-world -> ARX-scene transform for an episode."""
    if anchor_joint not in world_T_joint:
        raise KeyError(f"Cannot anchor replay: joint '{anchor_joint}' is absent")

    anchor_source = np.asarray(world_T_joint[anchor_joint][0, :3, 3], dtype=np.float64)
    scene_anchor = np.asarray(scene_anchor, dtype=np.float64)
    if scene_anchor.shape != (3,):
        raise ValueError(f"scene_anchor must have shape (3,), got {scene_anchor.shape}")

    scene_T_egodex = np.eye(4, dtype=np.float64)
    scene_T_egodex[:3, :3] = R_EGODEX_WORLD_TO_ARX
    scene_T_egodex[:3, 3] = scene_anchor - R_EGODEX_WORLD_TO_ARX @ anchor_source
    return scene_T_egodex


def transform_pose(scene_T_egodex: np.ndarray, egodex_T_frame: np.ndarray) -> np.ndarray:
    """Map one EgoDex homogeneous pose into the fixed ARX scene frame."""
    return scene_T_egodex @ egodex_T_frame


def frame_joint_positions(
    world_T_joint: Mapping[str, np.ndarray],
    frame: int,
    scene_T_egodex: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return all tracked joint origins at one frame in ARX scene coordinates."""
    positions: dict[str, np.ndarray] = {}
    for name, sequence in world_T_joint.items():
        positions[name] = transform_pose(scene_T_egodex, sequence[frame])[:3, 3]
    return positions

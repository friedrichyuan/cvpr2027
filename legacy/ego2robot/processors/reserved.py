"""Registered but disabled in Path A. Enable later without changing Manager."""

from __future__ import annotations

from ..processor import CPU, GPU_SAM3, Artifacts, EpisodeContext, ResourceSpec


class _Reserved:
    requires: frozenset[str] = frozenset()
    optional_requires: frozenset[str] = frozenset()
    resources: ResourceSpec = CPU

    def run(self, ctx: EpisodeContext) -> None:
        raise NotImplementedError(f"Processor '{self.name}' is reserved and not implemented yet")


class HandPoseEstProcessor(_Reserved):
    name = "hand_pose_est"
    produces = frozenset({Artifacts.ACTION})
    resources: ResourceSpec = GPU_SAM3


class SubtaskSegProcessor(_Reserved):
    name = "subtask_seg"
    produces = frozenset({Artifacts.SOURCE})


class DepthProcessor(_Reserved):
    name = "depth"
    produces = frozenset({Artifacts.DEPTH})
    resources: ResourceSpec = GPU_SAM3


class L2StatsProcessor(_Reserved):
    name = "l2_stats"
    produces: frozenset[str] = frozenset()


class L3VlmProcessor(_Reserved):
    name = "l3_vlm"
    requires = frozenset({Artifacts.COMPOSITE})
    produces: frozenset[str] = frozenset()


class ColorRandProcessor(_Reserved):
    name = "color_rand"
    requires = frozenset({Artifacts.ROBOT_RGB})
    produces = frozenset({Artifacts.ROBOT_RGB})

"""Built-in processor table and the Path A profile."""

from __future__ import annotations

from .manager import Registry
from .processors import (
    ActionAlignProcessor,
    ArmSegmentProcessor,
    BaseIkProcessor,
    ColorRandProcessor,
    ComposeProcessor,
    DepthProcessor,
    HandPoseEstProcessor,
    InpaintProcessor,
    L1FilterProcessor,
    L2StatsProcessor,
    L3VlmProcessor,
    RenderProcessor,
    SubtaskSegProcessor,
)

PATH_A_EGODEX = (
    "action_align",
    "arm_segment",
    "inpaint",
    "base_ik",
    "render",
    "compose",
    "l1_filter",
)

PROFILES = {
    "path_a_egodex": PATH_A_EGODEX,
    "action_only": ("action_align",),
}


def build_registry() -> Registry:
    registry = Registry()
    for processor in (
        ActionAlignProcessor(),
        ArmSegmentProcessor(),
        InpaintProcessor(),
        BaseIkProcessor(),
        RenderProcessor(),
        ComposeProcessor(),
        L1FilterProcessor(),
        HandPoseEstProcessor(),
        SubtaskSegProcessor(),
        DepthProcessor(),
        L2StatsProcessor(),
        L3VlmProcessor(),
        ColorRandProcessor(),
    ):
        registry.register(processor)
    return registry

"""SAM3 arm masks. Model is loaded lazily so CPU stages stay cheap."""

from __future__ import annotations

from ..processor import GPU_SAM3, Artifacts, EpisodeContext, ResourceSpec
from ..video import save_masks


class ArmSegmentProcessor:
    name = "arm_segment"
    requires: frozenset[str] = frozenset()
    produces = frozenset({Artifacts.MASKS})
    optional_requires: frozenset[str] = frozenset()
    resources: ResourceSpec = GPU_SAM3

    def run(self, ctx: EpisodeContext) -> None:
        if ctx.video_path is None or not ctx.video_path.is_file():
            raise FileNotFoundError(f"arm_segment needs an mp4 next to {ctx.hdf5_path}")
        from ..actors.sam3 import segment_person, unload

        masks = segment_person(ctx.video_path)
        save_masks(ctx.out_dir / Artifacts.MASKS, masks)
        unload()

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..processor import GPU_INPAINT, Artifacts, EpisodeContext, ResourceSpec
from ..video import load_masks


class InpaintProcessor:
    name = "inpaint"
    requires = frozenset({Artifacts.MASKS})
    produces = frozenset({Artifacts.INPAINT})
    optional_requires: frozenset[str] = frozenset()
    resources: ResourceSpec = GPU_INPAINT

    def run(self, ctx: EpisodeContext) -> None:
        if ctx.video_path is None or not ctx.video_path.is_file():
            raise FileNotFoundError(f"inpaint needs an mp4 next to {ctx.hdf5_path}")
        from ..actors.propainter import inpaint_video

        masks = load_masks(ctx.out_dir / Artifacts.MASKS)
        inpaint_video(
            ctx.video_path,
            masks,
            Path(ctx.out_dir / Artifacts.INPAINT),
        )

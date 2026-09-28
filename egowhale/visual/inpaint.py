"""Remove the person with ProPainter. The weights stay on the inpaint actor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from egowhale.media import load_masks, read_rgb, write_rgb
from egowhale.step import INPAINT, MASKS, ROOT, Step

_ROOT = ROOT / "thirdparty" / "propainter"


def _import():
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from inference_propainter import inpaint_video, load_models

    return load_models, inpaint_video


class Inpaint(Step):
    name = "inpaint"
    needs = (MASKS,)
    makes = (INPAINT,)
    gpus = 1

    def load(self):
        load_models, _inpaint = _import()
        self._held = load_models(str(_ROOT / "weights"))
        self._inpaint = _inpaint

    def run(self, src: Path, dst: Path) -> None:
        video = Path(src).with_suffix(".mp4")
        if not (_ROOT / "inference_propainter.py").is_file():
            raise FileNotFoundError(_ROOT)
        frames, fps = read_rgb(video)
        masks = load_masks(Path(dst) / MASKS)
        if len(frames) != len(masks):
            raise ValueError(f"{len(frames)} frames vs {len(masks)} masks")
        held = getattr(self, "_held", None)
        if held is None:
            self.load()
            held = self._held
        painted = self._inpaint(
            held,
            frames,
            masks,
            resize_ratio=0.5,
            mask_dilation=4,
            ref_stride=10,
            neighbor_length=10,
            subvideo_length=80,
            raft_iter=20,
            fp16=True,
        )
        out = Path(dst) / INPAINT
        write_rgb(out, np.stack(painted), fps)

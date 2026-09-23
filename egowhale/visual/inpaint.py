"""Remove the person with ProPainter. Depth is not used."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from egowhale.media import load_masks, read_rgb
from egowhale.step import INPAINT, MASKS, ROOT, Step

_ROOT = ROOT / "thirdparty" / "propainter"


class Inpaint(Step):
    name = "inpaint"
    needs = (MASKS,)
    makes = (INPAINT,)
    gpus = 1

    def run(self, src: Path, dst: Path) -> None:
        video = Path(src).with_suffix(".mp4")
        script = _ROOT / "inference_propainter.py"
        if not script.is_file():
            raise FileNotFoundError(script)
        frames, _fps = read_rgb(video)
        masks = load_masks(Path(dst) / MASKS)
        if len(frames) != len(masks):
            raise ValueError(f"{len(frames)} frames vs {len(masks)} masks")
        out = Path(dst) / INPAINT
        with tempfile.TemporaryDirectory(prefix="egowhale_inpaint_") as tmp:
            tmp_dir = Path(tmp)
            video_dir, mask_dir, out_dir = tmp_dir / "video", tmp_dir / "masks", tmp_dir / "out"
            video_dir.mkdir()
            mask_dir.mkdir()
            for index, (frame, mask) in enumerate(zip(frames, masks)):
                cv2.imwrite(str(video_dir / f"{index:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_dir / f"{index:05d}.png"), mask.astype(np.uint8) * 255)
            subprocess.run(
                [
                    os.environ.get("PROPAINTER_PYTHON", sys.executable),
                    str(script),
                    "--video",
                    str(video_dir),
                    "--mask",
                    str(mask_dir),
                    "--output",
                    str(out_dir),
                    "--fp16",
                    "--neighbor_length",
                    "10",
                    "--ref_stride",
                    "10",
                    "--subvideo_length",
                    "80",
                    "--mask_dilation",
                    "4",
                    "--raft_iter",
                    "20",
                    "--resize_ratio",
                    "0.5",
                    "--save_fps",
                    "30",
                ],
                check=True,
                cwd=str(_ROOT),
            )
            result = next(out_dir.rglob("inpaint_out.mp4"))
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result, out)

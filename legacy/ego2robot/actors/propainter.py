"""Run vendored ProPainter as a subprocess so it can keep its own env."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..video import read_rgb_video

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPAINTER = REPO_ROOT / "thirdparty" / "propainter"


def inpaint_video(video_path: Path, masks: np.ndarray, output_path: Path) -> None:
    root = Path(os.environ.get("PROPAINTER_ROOT", DEFAULT_PROPAINTER))
    script = root / "inference_propainter.py"
    if not script.is_file():
        raise FileNotFoundError(f"ProPainter script not found: {script}")
    python = os.environ.get("PROPAINTER_PYTHON", sys.executable)
    frames, fps = read_rgb_video(video_path)
    if len(frames) != len(masks):
        raise ValueError(f"Video has {len(frames)} frames but masks has {len(masks)}")
    with tempfile.TemporaryDirectory(prefix="ego2robot_inpaint_") as tmp:
        tmp_dir = Path(tmp)
        video_dir = tmp_dir / "video"
        mask_dir = tmp_dir / "masks"
        out_dir = tmp_dir / "out"
        video_dir.mkdir()
        mask_dir.mkdir()
        for index, (frame, mask) in enumerate(zip(frames, masks)):
            cv2.imwrite(str(video_dir / f"{index:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_dir / f"{index:05d}.png"), (mask.astype(np.uint8) * 255))
        command = [
            python,
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
            os.environ.get("PROPAINTER_RESIZE_RATIO", "0.5"),
        ]
        subprocess.run(command, check=True, cwd=str(root))
        result = next(out_dir.rglob("inpaint_out.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, output_path)

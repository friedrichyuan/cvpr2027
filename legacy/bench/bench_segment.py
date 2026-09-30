"""Archived EfficientSAM3 microbench. Production segment is SAM 3 plus Cutie.

Single-GPU EfficientSAM3 throughput.

Times the segment path on one visible GPU. torch.compile covers the detection
head and the image backbone. The first call pays for compilation and is
reported separately.

    python scripts/bench_segment.py --video clip.mp4
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_segment.py --video clip.mp4 --no-compile
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from egowhale.media import read_rgb
from egowhale.visual.segment import _segment, load_predictor


def _prepare(frames: np.ndarray) -> np.ndarray:
    if len(frames) >= 16:
        return frames
    reps = 16 // len(frames) + 1
    return np.concatenate([frames] * reps)[:16]


def _compile(processor) -> None:
    processor.model.forward_grounding = torch.compile(
        processor.model.forward_grounding, mode="default"
    )
    processor.model.backbone.forward_image = torch.compile(
        processor.model.backbone.forward_image, mode="default"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU EfficientSAM3 segment throughput")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    frames, _fps = read_rgb(args.video)
    frames = _prepare(frames)
    processor = load_predictor()
    if not args.no_compile:
        _compile(processor)

    with torch.inference_mode():
        torch.cuda.synchronize()
        warm0 = time.perf_counter()
        _segment(processor, frames=frames[:8])
        _segment(processor, frames=frames[:4])
        torch.cuda.synchronize()
        warmup = time.perf_counter() - warm0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        masks = _segment(processor, frames=frames)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"gpu={torch.cuda.get_device_name(0)} compile={not args.no_compile} "
        f"frames={len(masks)} warmup={warmup:.1f}s {elapsed:.2f}s {len(masks)/elapsed:.2f}fps "
        f"peak={peak:.0f}MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()

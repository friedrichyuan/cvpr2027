"""Time production crop inpainting and compare its output with an earlier run.

Run once per code version with --save, then pass --ref to diff against a saved run.
Two runs of the same code give the run-to-run floor.

    CUDA_VISIBLE_DEVICES=8 python legacy/bench/bench_compose.py --save /tmp/pp_a
    CUDA_VISIBLE_DEVICES=8 python legacy/bench/bench_compose.py --save /tmp/pp_b --ref /tmp/pp_a
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

from egowhale.media import load_masks, read_rgb
from egowhale.visual.paint import load_models, paint_clips, prepare_crop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="/home/ps/code/egowhale/outputs/stack80rt/stack")
    parser.add_argument("--videos", default="/data/datasets/EgoDex/test/stack")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--ref", type=Path)
    args = parser.parse_args()
    args.save.mkdir(parents=True, exist_ok=True)

    models = load_models(str(ROOT / "thirdparty" / "propainter" / "weights"))
    folders = sorted(Path(args.episodes).iterdir(), key=lambda path: int(path.name))[: args.count]
    clips = []
    for folder in folders:
        frames, _fps = read_rgb(Path(args.videos) / f"{folder.name}.mp4")
        masks = load_masks(folder / "visual" / "masks.npz")
        clips.append((folder.name, prepare_crop(frames, masks)))
    paint_clips(models, [clips[0][1]])
    torch.cuda.synchronize()

    total_frames, total_s = 0, 0.0
    for name, clip in clips:
        started = time.perf_counter()
        painted = paint_clips(models, [clip])[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        total_frames += len(painted)
        total_s += seconds
        np.save(args.save / f"{name}.npy", painted)
        line = f"{name:>4s} frames {len(painted):4d} fps {len(painted) / seconds:6.2f}"
        if args.ref:
            ref = np.load(args.ref / f"{name}.npy")
            diff = np.abs(ref.astype(np.int16) - painted.astype(np.int16))
            line += f"  max diff {diff.max():3d}  pixels differing {100 * (diff > 0).any(-1).mean():.4f}%"
        print(line, flush=True)
    print(f"total frames {total_frames} fps {total_frames / total_s:.2f}  peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()

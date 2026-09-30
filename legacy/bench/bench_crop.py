"""ProPainter on the mask's bounding box versus the whole frame.

Same weights and arguments as production. The crop is the union of the clip's
masks plus a margin, aligned to 16 pixels, resized by the same 0.5 ratio, then
pasted back. Timing covers paint_clips only. Quality is PSNR inside the dilated
hole against the whole-frame run; the saved inpaint.mp4 gives the run-to-run floor.

    CUDA_VISIBLE_DEVICES=0 python legacy/bench/bench_crop.py --out /tmp/pp_crop.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from egowhale.media import load_masks, read_rgb
from egowhale.visual.paint import load_models, paint_clips, prepare_clip


def _box(masks: np.ndarray, margin: float) -> tuple[int, int, int, int]:
    height, width = masks.shape[1:]
    ys, xs = np.where(masks.any(0))
    pad = int(margin * max(height, width))
    y0 = max(0, (ys.min() - pad) // 16 * 16)
    x0 = max(0, (xs.min() - pad) // 16 * 16)
    y1 = min(height, -(-(ys.max() + 1 + pad) // 16) * 16)
    x1 = min(width, -(-(xs.max() + 1 + pad) // 16) * 16)
    return y0, y1, x0, x1


def _half(frames: np.ndarray) -> np.ndarray:
    height, width = frames.shape[1:3]
    size = (int(width * 0.5), int(height * 0.5))
    return np.stack([np.asarray(Image.fromarray(frame).resize(size)) for frame in frames])


def _hole(masks: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.stack([
        scipy.ndimage.binary_dilation(
            cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST), iterations=4
        )
        for mask in masks
    ])


def _psnr(a: np.ndarray, b: np.ndarray, where: np.ndarray) -> float:
    diff = (a.astype(np.float64) - b.astype(np.float64))[where]
    mse = float(np.mean(diff**2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _paint(models, frames, masks):
    clip = prepare_clip(frames, masks)
    started = time.perf_counter()
    painted = paint_clips(models, [clip])[0]
    return painted, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="/tmp/stack_dbg/stack")
    parser.add_argument("--videos", default="/data/datasets/EgoDex/test/stack")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--out", default="/tmp/pp_crop.json")
    args = parser.parse_args()

    models = load_models(str(ROOT / "thirdparty" / "propainter" / "weights"))
    rows = []
    warm = True
    for folder in sorted(Path(args.episodes).iterdir(), key=lambda path: int(path.name)):
        masks = load_masks(folder / "visual" / "masks.npz").astype(bool)
        frames, _fps = read_rgb(Path(args.videos) / f"{folder.name}.mp4")
        count = min(len(frames), len(masks))
        frames, masks = frames[:count], masks[:count]
        if warm:
            _paint(models, frames[:16], masks[:16])
            warm = False

        full, full_s = _paint(models, frames, masks)
        y0, y1, x0, x1 = _box(masks, args.margin)
        crop, crop_s = _paint(models, np.ascontiguousarray(frames[:, y0:y1, x0:x1]), masks[:, y0:y1, x0:x1])

        pasted = _half(frames)
        pasted[:, y0 // 2 : y0 // 2 + crop.shape[1], x0 // 2 : x0 // 2 + crop.shape[2]] = crop
        height, width = full.shape[1:3]
        hole = _hole(masks, width, height)
        saved, _ = read_rgb(folder / "visual" / "inpaint.mp4")
        saved = saved[:count]
        area = (y1 - y0) * (x1 - x0) / (masks.shape[1] * masks.shape[2])
        per_frame = [_psnr(pasted[index], full[index], hole[index]) for index in range(count) if hole[index].any()]
        row = {
            "episode": folder.name,
            "frames": count,
            "area": area,
            "full_fps": count / full_s,
            "crop_fps": count / crop_s,
            "crop_vs_full_db": _psnr(pasted, full, hole),
            "crop_worst_frame_db": min(per_frame),
            "full_vs_saved_db": _psnr(full, saved, hole),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    frames = sum(row["frames"] for row in rows)
    summary = {
        "margin": args.margin,
        "episodes": len(rows),
        "frames": frames,
        "area_mean": float(np.mean([row["area"] for row in rows])),
        "full_fps": frames / sum(row["frames"] / row["full_fps"] for row in rows),
        "crop_fps": frames / sum(row["frames"] / row["crop_fps"] for row in rows),
        "crop_vs_full_db_mean": float(np.mean([row["crop_vs_full_db"] for row in rows])),
        "crop_vs_full_db_min": float(min(row["crop_vs_full_db"] for row in rows)),
        "full_vs_saved_db_mean": float(np.mean([row["full_vs_saved_db"] for row in rows])),
    }
    print(json.dumps(summary), flush=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

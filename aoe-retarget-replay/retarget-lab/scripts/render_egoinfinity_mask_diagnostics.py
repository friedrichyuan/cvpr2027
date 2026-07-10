#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def decode_rgb(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))


def unpack_mask(obj: dict) -> np.ndarray | None:
    packed = obj.get("mask_packed")
    shape = obj.get("mask_shape")
    if packed is None or shape is None:
        return None
    h, w = [int(x) for x in shape]
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: h * w]
    return bits.reshape(h, w).astype(bool)


def object_prompt(result: dict, obj_id) -> str:
    mapping = result.get("sam3_prompt_mapping") or []
    try:
        idx = int(obj_id)
    except Exception:
        return ""
    if 0 <= idx < len(mapping) and isinstance(mapping[idx], dict):
        return str(mapping[idx].get("prompt", ""))
    return ""


def selected_object_ids(result: dict, target_prompt: str | None, object_id: int | None) -> list:
    if object_id is not None:
        return [object_id]
    mesh_info = result.get("sam3_mesh_info") or {}
    out = []
    for obj_id in mesh_info.keys():
        if target_prompt and object_prompt(result, obj_id) != target_prompt:
            continue
        out.append(obj_id)
    return out


def draw_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = rgb.copy()
    overlay[mask] = np.asarray(color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, rgb, 1.0 - alpha, 0, dst=rgb)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render EgoInfinity SAM3.1 object mask diagnostics before SAM3D.")
    parser.add_argument("--pipeline-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--target-prompt", default=None)
    parser.add_argument("--object-id", type=int, default=None)
    args = parser.parse_args()

    with gzip.open(args.pipeline_result, "rb") as handle:
        result = pickle.load(handle)
    frames = result.get("frame_data") or []
    if not frames:
        raise RuntimeError("pipeline result has no frame_data")
    object_ids = selected_object_ids(result, args.target_prompt, args.object_id)
    if not object_ids:
        raise RuntimeError("no matching EgoInfinity object masks found")

    first = decode_rgb(frames[0]["img_rgb"])
    h, w = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {args.output}")

    colors = [(30, 225, 85), (255, 180, 40), (80, 170, 255), (255, 80, 140)]
    for frame_idx, frame in enumerate(frames):
        rgb = decode_rgb(frame["img_rgb"]).copy()
        obj_data = frame.get("sam3_obj_data") or {}
        for idx, obj_id in enumerate(object_ids):
            obj = obj_data.get(obj_id) or obj_data.get(str(obj_id))
            if not isinstance(obj, dict):
                continue
            mask = unpack_mask(obj)
            if mask is not None and mask.any():
                draw_mask(rgb, mask, colors[idx % len(colors)], alpha=0.35)
        cv2.putText(rgb, f"frame {frame_idx:04d}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

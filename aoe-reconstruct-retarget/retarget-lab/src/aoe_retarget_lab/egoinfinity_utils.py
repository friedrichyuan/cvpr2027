from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np


def load_result(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def oid_get(mapping: dict, obj_id):
    return mapping.get(obj_id) or mapping.get(str(obj_id)) or mapping.get(int(obj_id))


def object_prompt(result: dict, obj_id) -> str:
    mapping = result.get("sam3_prompt_mapping") or []
    try:
        idx = int(obj_id)
    except Exception:
        return ""
    if 0 <= idx < len(mapping) and isinstance(mapping[idx], dict):
        return str(mapping[idx].get("prompt", ""))
    return ""


def object_prompt_score(result: dict, obj_id) -> float:
    mapping = result.get("sam3_prompt_mapping") or []
    try:
        idx = int(obj_id)
    except Exception:
        return 0.0
    if 0 <= idx < len(mapping) and isinstance(mapping[idx], dict):
        try:
            return float(mapping[idx].get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def mask_centroid(obj: dict | None) -> np.ndarray | None:
    if not isinstance(obj, dict):
        return None
    packed = obj.get("mask_packed")
    shape = obj.get("mask_shape")
    if packed is None or shape is None:
        return None
    h, w = [int(value) for value in shape]
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: h * w]
    mask = bits.reshape(h, w).astype(bool)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return np.array(
        [(xs.min() + xs.max()) * 0.5, (ys.min() + ys.max()) * 0.5],
        dtype=np.float32,
    )

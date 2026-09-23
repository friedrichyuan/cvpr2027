"""Thin SAM3 video wrapper. Loads the predictor once per process."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PREDICTOR = None


def segment_person(video_path: Path, prompt: str = "person") -> np.ndarray:
    predictor = _predictor()
    response = predictor.handle_request(request=dict(type="start_session", resource_path=str(video_path)))
    session_id = response["session_id"]
    mid = _frame_count(video_path) // 2
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=mid,
            text=prompt,
        )
    )
    outputs = {}
    for item in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs[item["frame_index"]] = item["outputs"]
    if not outputs:
        raise RuntimeError("SAM3 returned no frames")
    ordered = [outputs[index] for index in sorted(outputs)]
    masks = [_union_masks(frame) for frame in ordered]
    return np.stack(masks)


_DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "thirdparty" / "sam3" / "weights" / "sam3" / "sam3.pt"


def _predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        import os

        from sam3.model_builder import build_sam3_video_predictor

        checkpoint = Path(os.environ.get("SAM3_CHECKPOINT", _DEFAULT_CKPT))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
        _PREDICTOR = build_sam3_video_predictor(checkpoint_path=str(checkpoint))
    return _PREDICTOR


def unload() -> None:
    """Drop the cached predictor so the next GPU stage can use the full card."""
    global _PREDICTOR
    _PREDICTOR = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _union_masks(outputs: dict) -> np.ndarray:
    masks = np.asarray(outputs.get("out_binary_masks", []))
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.size == 0:
        raise RuntimeError("SAM3 produced empty masks")
    return np.any(masks.astype(bool), axis=0)


def _frame_count(video_path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    capture.release()
    return max(count, 1)

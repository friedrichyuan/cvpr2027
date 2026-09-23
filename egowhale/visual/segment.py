"""SAM3 person masks. The predictor is dropped before the next GPU stage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from egowhale.media import save_masks
from egowhale.step import MASKS, ROOT, Step

_CKPT = ROOT / "thirdparty" / "sam3" / "weights" / "sam3" / "sam3.pt"


class Segment(Step):
    name = "segment"
    makes = (MASKS,)
    gpus = 1

    def run(self, src: Path, dst: Path) -> None:
        video = Path(src).with_suffix(".mp4")
        if not video.is_file():
            raise FileNotFoundError(video)
        if not _CKPT.is_file():
            raise FileNotFoundError(_CKPT)
        from sam3.model_builder import build_sam3_video_predictor

        predictor = build_sam3_video_predictor(checkpoint_path=str(_CKPT))
        try:
            masks = _segment(predictor, video)
        finally:
            del predictor
        save_masks(Path(dst) / MASKS, masks)


def _segment(predictor, video: Path) -> np.ndarray:
    session = predictor.handle_request(request={"type": "start_session", "resource_path": str(video)})
    mid = _frame_count(video) // 2
    predictor.handle_request(
        request={"type": "add_prompt", "session_id": session["session_id"], "frame_index": mid, "text": "person"}
    )
    outputs = {}
    for item in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session["session_id"]}
    ):
        outputs[item["frame_index"]] = item["outputs"]
    if not outputs:
        raise RuntimeError("SAM3 returned no frames")
    masks = np.stack([_union(outputs[index]) for index in sorted(outputs)])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = [cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool) for mask in masks]
    return np.stack(closed)


def _union(outputs: dict) -> np.ndarray:
    masks = np.asarray(outputs.get("out_binary_masks", []))
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.size == 0:
        raise RuntimeError("SAM3 produced an empty mask")
    return np.any(masks.astype(bool), axis=0)


def _frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    capture.release()
    return max(count, 1)

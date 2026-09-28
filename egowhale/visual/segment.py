"""SAM 3 person masks. The predictor stays on the segment actor."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from egowhale.media import save_masks
from egowhale.step import MASKS, ROOT, Step

_ROOT = ROOT / "thirdparty" / "sam3"
_CKPT = _ROOT / "weights" / "sam3" / "sam3.pt"


def load_predictor():
    """SAM 3 ships in thirdparty and is not installed on the worker by default."""
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from sam3.model_builder import build_sam3_video_predictor

    return build_sam3_video_predictor(checkpoint_path=str(_CKPT))


def _slice_batch(value, local: int, batch: int):
    """Keep one frame from a backbone batch. The tracker still consumes a single frame."""
    import torch

    if torch.is_tensor(value):
        if value.ndim >= 2 and value.shape[0] == batch:
            return value[local : local + 1]
        return value
    if isinstance(value, dict):
        return {key: _slice_batch(item, local, batch) for key, item in value.items()}
    if isinstance(value, list):
        return [_slice_batch(item, local, batch) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_batch(item, local, batch) for item in value)
    tensors = getattr(value, "tensors", None)
    if torch.is_tensor(tensors):
        mask = getattr(value, "mask", None)
        return type(value)(_slice_batch(tensors, local, batch), None if mask is None else _slice_batch(mask, local, batch))
    return value


def install_frame_batch(predictor, frames: int = 16) -> None:
    """Batch the image backbone across frames of one video. The tracker still steps in order."""
    import torch

    detector = predictor.model.detector
    original = detector._get_img_feats
    state = {"budget": max(1, int(frames)), "start": None, "end": None, "feats": None, "token": None}

    def wrapped(backbone_out, img_ids):
        if "backbone_fpn" in backbone_out or not torch.is_tensor(img_ids) or img_ids.numel() != 1:
            return original(backbone_out, img_ids)
        img_batch = backbone_out.get("img_batch_all_stages")
        if not torch.is_tensor(img_batch):
            return original(backbone_out, img_ids)
        index = int(img_ids.reshape(-1)[0].item())
        token = (img_batch.data_ptr(), int(img_batch.shape[0]))
        if state["token"] != token or state["start"] is None or not (state["start"] <= index < state["end"]):
            budget = max(1, state["budget"])
            while True:
                start = max(0, index - budget // 2)
                end = min(int(img_batch.shape[0]), start + budget)
                start = max(0, end - budget)
                images = img_batch[start:end].to(device=detector.device, dtype=torch.float32)
                try:
                    state["feats"] = detector.backbone.forward_image(images)
                    state["start"], state["end"], state["token"] = start, end, token
                    break
                except Exception as exc:
                    oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                    if not oom:
                        raise
                    torch.cuda.empty_cache()
                    if budget <= 1:
                        raise
                    budget = max(1, budget // 2)
                    state["budget"] = budget
                    state["feats"] = None
        local = index - state["start"]
        chunk = state["end"] - state["start"]
        mapping = torch.full((int(img_batch.shape[0]),), -1, dtype=torch.long, device=img_ids.device)
        mapping[index] = 0
        merged = {**backbone_out, **_slice_batch(state["feats"], local, chunk), "id_mapping": mapping}
        return original(merged, img_ids)

    detector._get_img_feats = wrapped


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
        held = getattr(self, "_held", None)
        predictor = held if held is not None else load_predictor()
        try:
            masks = _segment(predictor, video)
        finally:
            if held is None:
                del predictor
        save_masks(Path(dst) / MASKS, masks)

    def consume(self, payload: dict) -> dict:
        """Segment one prefetched video. The backbone batches frames inside the video."""
        import time

        results = []
        predictor = getattr(self, "_held", None)
        if predictor is None:
            predictor = load_predictor()
            self._held = predictor
        for item in payload["items"]:
            if not item.get("ok"):
                results.append((False, 0.0, item.get("error") or "feed failed"))
                continue
            started = time.perf_counter()
            try:
                masks = _segment(predictor, frames=item["frames"])
                save_masks(Path(item["dst"]), masks)
            except Exception as exc:
                results.append((False, 0.0, str(exc).splitlines()[-1]))
                continue
            results.append((True, time.perf_counter() - started, ""))
        return {"results": results}


def _segment(predictor, video: Path | None = None, frames: np.ndarray | None = None) -> np.ndarray:
    if frames is not None:
        from PIL import Image

        resource = [Image.fromarray(frame) for frame in frames]
        mid = len(frames) // 2
    else:
        resource = str(video)
        mid = _frame_count(video) // 2
    session = predictor.handle_request(request={"type": "start_session", "resource_path": resource})
    session_id = session["session_id"]
    try:
        predictor.handle_request(
            request={"type": "add_prompt", "session_id": session_id, "frame_index": mid, "text": "person"}
        )
        outputs = {}
        for item in predictor.handle_stream_request(
            request={"type": "propagate_in_video", "session_id": session_id}
        ):
            outputs[item["frame_index"]] = item["outputs"]
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id, "run_gc_collect": False})
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

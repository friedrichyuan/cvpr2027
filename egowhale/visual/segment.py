"""EfficientSAM3 person masks. The image model stays on the segment actor."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from egowhale.media import read_rgb, save_masks
from egowhale.step import MASKS, ROOT, Step

_PKG = ROOT / "thirdparty" / "efficientsam3" / "sam3"
_CKPT = ROOT / "thirdparty" / "efficientsam3" / "sam3_checkpoints" / "efficientsam3_tinyvit.pt"
_BPE = _PKG / "assets" / "bpe_simple_vocab_16e6.txt.gz"
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_BATCH = 8


def load_predictor():
    """EfficientSAM3 ships in thirdparty and is not installed on the worker by default."""
    root = str(_PKG)
    if root not in sys.path:
        sys.path.insert(0, root)
    psutil = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "ray" / "thirdparty_files"
    if psutil.is_dir() and str(psutil) not in sys.path:
        sys.path.insert(0, str(psutil))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_efficientsam3_image_model

    model = build_efficientsam3_image_model(
        checkpoint_path=str(_CKPT),
        bpe_path=str(_BPE),
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        load_from_HF=False,
        device="cuda",
    )
    return Sam3Processor(model, confidence_threshold=0.5)


def install_frame_batch(predictor, frames: int = 16) -> None:
    """The segment actor still calls this. The detection head batch is fixed."""
    return None


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
        """Segment one prefetched video. The detection head takes several frames at once."""
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


def _text(processor):
    cached = getattr(processor, "_person", None)
    if cached is None:
        cached = processor.model.backbone.forward_text(["person"], device=processor.device)
        processor._person = cached
    return cached


def _segment(processor, video: Path | None = None, frames: np.ndarray | None = None) -> np.ndarray:
    if frames is None:
        frames, _fps = read_rgb(video)
    if len(frames) == 0:
        raise RuntimeError("EfficientSAM3 returned no frames")
    text = _text(processor)
    masks = [_ground(processor, frames[start : start + _BATCH], text) for start in range(0, len(frames), _BATCH)]
    return np.concatenate(masks)


def _ground(processor, frames: np.ndarray, text: dict) -> np.ndarray:
    from PIL import Image

    from sam3.model.data_misc import FindStage, interpolate

    state = processor.set_image_batch([Image.fromarray(frame) for frame in frames])
    state["backbone_out"].update(text)
    count = len(frames)
    device = processor.device
    find = FindStage(
        img_ids=torch.arange(count, device=device),
        text_ids=torch.zeros(count, dtype=torch.long, device=device),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    out = processor.model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=find,
        find_target=None,
        geometric_prompt=processor.model._get_dummy_prompt(count),
    )
    probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
    low = out["pred_masks"]
    keep = probs > processor.confidence_threshold
    height, width = frames.shape[1], frames.shape[2]
    masks = []
    for index in range(count):
        chosen = low[index][keep[index]]
        if chosen.numel() == 0:
            masks.append(np.zeros((height, width), dtype=bool))
            continue
        up = interpolate(chosen.unsqueeze(1), (height, width), mode="bilinear", align_corners=False).sigmoid()
        merged = (up.squeeze(1) > 0.5).any(dim=0)
        masks.append(_close(merged.detach().cpu().numpy()))
    return np.stack(masks)


def _close(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL).astype(bool)

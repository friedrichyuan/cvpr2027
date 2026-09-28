"""ProPainter weights and batched inpainting. thirdparty stays unmodified."""

from __future__ import annotations

import os
import sys

import numpy as np

from egowhale.step import ROOT

_ROOT = ROOT / "thirdparty" / "propainter"
_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


def _libs():
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from core.utils import to_tensors
    from inference_propainter import get_ref_index, resize_frames
    from model.misc import get_device
    from model.modules.flow_comp_raft import RAFT_bi
    from model.propainter import InpaintGenerator
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from utils.download_util import load_file_from_url

    return {
        "to_tensors": to_tensors,
        "get_ref_index": get_ref_index,
        "resize_frames": resize_frames,
        "get_device": get_device,
        "RAFT_bi": RAFT_bi,
        "InpaintGenerator": InpaintGenerator,
        "RecurrentFlowCompleteNet": RecurrentFlowCompleteNet,
        "load_file_from_url": load_file_from_url,
    }


def load_models(weight_dir: str, device=None):
    """RAFT, flow completion, and the painter. They stay on the inpaint actor."""
    libs = _libs()
    device = device or libs["get_device"]()
    os.makedirs(weight_dir, exist_ok=True)
    raft_path = libs["load_file_from_url"](
        url=_URL + "raft-things.pth", model_dir=weight_dir, progress=False, file_name=None
    )
    raft = libs["RAFT_bi"](raft_path, device)
    flow_path = libs["load_file_from_url"](
        url=_URL + "recurrent_flow_completion.pth", model_dir=weight_dir, progress=False, file_name=None
    )
    flow = libs["RecurrentFlowCompleteNet"](flow_path)
    for parameter in flow.parameters():
        parameter.requires_grad = False
    flow.to(device).eval()
    paint_path = libs["load_file_from_url"](
        url=_URL + "ProPainter.pth", model_dir=weight_dir, progress=False, file_name=None
    )
    painter = libs["InpaintGenerator"](model_path=paint_path).to(device).eval()
    return {"raft": raft, "flow": flow, "painter": painter, "device": device, "half": False}


def prepare_clip(frames: np.ndarray, masks: np.ndarray, resize_ratio: float = 0.5, mask_dilation: int = 4) -> dict:
    """Resize and dilate on the CPU dataloader. Tensors stay on CPU."""
    import cv2
    import scipy.ndimage
    import torch
    from PIL import Image

    libs = _libs()
    height, width = frames.shape[1], frames.shape[2]
    size = (int(width * resize_ratio), int(height * resize_ratio))
    resized, process_size, out_size = libs["resize_frames"]([Image.fromarray(frame) for frame in frames], size)
    process_w, process_h = process_size
    flow_masks = []
    dilated = []
    for mask in masks:
        image = cv2.resize(mask.astype(np.uint8), (process_w, process_h), interpolation=cv2.INTER_NEAREST)
        if mask_dilation > 0:
            image = scipy.ndimage.binary_dilation(image, iterations=mask_dilation).astype(np.uint8)
        else:
            image = (image > 0).astype(np.uint8)
        painted = Image.fromarray(image * 255)
        flow_masks.append(painted)
        dilated.append(painted)
    convert = libs["to_tensors"]()
    return {
        "frames": (convert(resized).unsqueeze(0) * 2 - 1).cpu(),
        "flow": convert(flow_masks).unsqueeze(0).cpu(),
        "masks": convert(dilated).unsqueeze(0).cpu(),
        "original": np.stack([np.asarray(frame) for frame in resized]),
        "out_size": out_size,
        "key": (int(resized[0].size[0]), int(resized[0].size[1]), len(resized)),
    }


def paint_clips(models, clips: list[dict], ref_stride=10, neighbor_length=10, subvideo_length=80, raft_iter=20, fp16=True):
    """One forward per group of clips that share process size and length."""
    import torch

    buckets: dict[tuple, list] = {}
    for index, clip in enumerate(clips):
        buckets.setdefault(clip["key"], []).append(index)
    painted = [None] * len(clips)
    for indexes in buckets.values():
        group = [clips[index] for index in indexes]
        outputs = _forward(
            models,
            torch.cat([clip["frames"] for clip in group], dim=0),
            torch.cat([clip["flow"] for clip in group], dim=0),
            torch.cat([clip["masks"] for clip in group], dim=0),
            [clip["original"] for clip in group],
            [clip["out_size"] for clip in group],
            ref_stride=ref_stride,
            neighbor_length=neighbor_length,
            subvideo_length=subvideo_length,
            raft_iter=raft_iter,
            fp16=fp16,
        )
        for index, frames in zip(indexes, outputs):
            painted[index] = frames
    return painted


def _forward(models, frames, flow_masks, masks, originals, out_sizes, ref_stride, neighbor_length, subvideo_length, raft_iter, fp16):
    import cv2
    import torch

    device = models["device"]
    frames = frames.to(device)
    flow_masks = flow_masks.to(device)
    masks = masks.to(device)
    batch, video_length, _, height, width = frames.shape
    raft = models["raft"]
    flow = models["flow"]
    painter = models["painter"]
    get_ref_index = _libs()["get_ref_index"]

    def release() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with torch.no_grad():
        if width <= 640:
            short_clip = 12
        elif width <= 720:
            short_clip = 8
        elif width <= 1280:
            short_clip = 4
        else:
            short_clip = 2
        if video_length > short_clip:
            forward_flows, backward_flows = [], []
            for start in range(0, video_length, short_clip):
                end = min(video_length, start + short_clip)
                window = frames[:, start:end] if start == 0 else frames[:, start - 1 : end]
                flows_f, flows_b = raft(window, iters=raft_iter)
                forward_flows.append(flows_f)
                backward_flows.append(flows_b)
                release()
            flows = (torch.cat(forward_flows, dim=1), torch.cat(backward_flows, dim=1))
        else:
            flows = raft(frames, iters=raft_iter)
            release()

        use_half = bool(fp16) and getattr(device, "type", "") == "cuda"
        if use_half:
            frames = frames.half()
            flow_masks = flow_masks.half()
            masks = masks.half()
            flows = (flows[0].half(), flows[1].half())
            if not models["half"]:
                models["flow"] = flow.half()
                models["painter"] = painter.half()
                flow = models["flow"]
                painter = models["painter"]
                models["half"] = True

        flow_length = flows[0].size(1)
        if flow_length > subvideo_length:
            pred_f, pred_b = [], []
            pad = 5
            for start in range(0, flow_length, subvideo_length):
                left = max(0, start - pad)
                right = min(flow_length, start + subvideo_length + pad)
                pad_left = max(0, start) - left
                pad_right = right - min(flow_length, start + subvideo_length)
                pair = (flows[0][:, left:right], flows[1][:, left:right])
                completed, _edges = flow.forward_bidirect_flow(pair, flow_masks[:, left : right + 1])
                completed = flow.combine_flow(pair, completed, flow_masks[:, left : right + 1])
                pred_f.append(completed[0][:, pad_left : right - left - pad_right])
                pred_b.append(completed[1][:, pad_left : right - left - pad_right])
                release()
            completed_flows = (torch.cat(pred_f, dim=1), torch.cat(pred_b, dim=1))
        else:
            completed_flows, _edges = flow.forward_bidirect_flow(flows, flow_masks)
            completed_flows = flow.combine_flow(flows, completed_flows, flow_masks)
            release()

        masked = frames * (1 - masks)
        prop_length = min(100, subvideo_length)
        if video_length > prop_length:
            updated_frames, updated_masks = [], []
            pad = 10
            for start in range(0, video_length, prop_length):
                left = max(0, start - pad)
                right = min(video_length, start + prop_length + pad)
                pad_left = max(0, start) - left
                pad_right = right - min(video_length, start + prop_length)
                count = masks[:, left:right].size(1)
                pair = (completed_flows[0][:, left : right - 1], completed_flows[1][:, left : right - 1])
                propagated, local_masks = painter.img_propagation(masked[:, left:right], pair, masks[:, left:right], "nearest")
                filled = frames[:, left:right] * (1 - masks[:, left:right]) + propagated.view(batch, count, 3, height, width) * masks[:, left:right]
                updated_frames.append(filled[:, pad_left : right - left - pad_right])
                updated_masks.append(local_masks.view(batch, count, 1, height, width)[:, pad_left : right - left - pad_right])
                release()
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            propagated, local_masks = painter.img_propagation(masked, completed_flows, masks, "nearest")
            updated_frames = frames * (1 - masks) + propagated.view(batch, video_length, 3, height, width) * masks
            updated_masks = local_masks.view(batch, video_length, 1, height, width)
            release()

    stride = neighbor_length // 2
    ref_num = subvideo_length // ref_stride if video_length > subvideo_length else -1
    composite = [[None] * video_length for _ in range(batch)]
    for start in range(0, video_length, stride):
        neighbors = list(range(max(0, start - stride), min(video_length, start + stride + 1)))
        refs = get_ref_index(start, neighbors, video_length, ref_stride, ref_num)
        selected = neighbors + refs
        with torch.no_grad():
            predicted = painter(
                updated_frames[:, selected],
                (completed_flows[0][:, neighbors[:-1]], completed_flows[1][:, neighbors[:-1]]),
                masks[:, selected],
                updated_masks[:, selected],
                len(neighbors),
            )
            predicted = (predicted.float() + 1) / 2
            predicted = predicted.cpu().permute(0, 1, 3, 4, 2).numpy() * 255
            binary = masks[:, neighbors].float().cpu().permute(0, 1, 3, 4, 2).numpy()
            binary = (binary > 0.5).astype(np.uint8)
        for item in range(batch):
            for local, index in enumerate(neighbors):
                image = predicted[item, local] * binary[item, local] + originals[item][index] * (1 - binary[item, local])
                previous = composite[item][index]
                if previous is not None:
                    image = previous.astype(np.float32) * 0.5 + image.astype(np.float32) * 0.5
                composite[item][index] = image.astype(np.uint8)
        release()

    outputs = []
    for item, out_size in enumerate(out_sizes):
        frames_out = []
        for frame in composite[item]:
            if frame is None:
                raise RuntimeError("ProPainter left a frame empty")
            if (frame.shape[1], frame.shape[0]) != tuple(out_size):
                frame = cv2.resize(frame, tuple(out_size), interpolation=cv2.INTER_CUBIC)
            frames_out.append(frame)
        outputs.append(np.stack(frames_out))
    return outputs

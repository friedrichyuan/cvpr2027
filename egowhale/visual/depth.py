"""Scene depth from Depth Anything 3, scaled by the EgoDex camera trajectory."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

from egowhale.media import read_rgb
from egowhale.step import DEPTH, INPAINT, ROOT, Step

_SRC = ROOT / "thirdparty" / "da3" / "src"
_WEIGHTS = ROOT / "thirdparty" / "da3" / "weights" / "DA3-GIANT"


class Depth(Step):
    name = "depth"
    needs = (INPAINT,)
    makes = (DEPTH,)
    gpus = 1

    def run(self, src: Path, dst: Path) -> None:
        if not (_WEIGHTS / "model.safetensors").is_file():
            raise FileNotFoundError(_WEIGHTS)
        frames, _fps = read_rgb(Path(dst) / INPAINT)
        extrinsics, intrinsics = _cameras(Path(src), frames.shape[1], frames.shape[2], len(frames))
        depth = _estimate(frames[: len(extrinsics)], extrinsics, intrinsics, getattr(self, "_model", None))
        path = Path(dst) / DEPTH
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, depth=depth.astype(np.float32))
        print(f"  depth median {float(np.median(depth)):.3f} m")


def _cameras(episode: Path, height: int, width: int, frames: int):
    with h5py.File(episode, "r") as handle:
        c2w = np.asarray(handle["transforms/camera"], dtype=np.float64)
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
    count = min(frames, len(c2w))
    video = episode.with_suffix(".mp4")
    capture = cv2.VideoCapture(str(video))
    source_h = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
    source_w = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    capture.release()
    intrinsic = intrinsic.copy()
    intrinsic[0] *= width / source_w
    intrinsic[1] *= height / source_h
    extrinsics = np.stack([_invert(c2w[index]) for index in range(count)])
    return extrinsics, np.repeat(intrinsic[None], count, axis=0)


def load_model():
    """DA3-GIANT kept on a depth actor. Serial runs load and drop it per episode."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    _stub_evo()
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    model = create_object(load_config(MODEL_REGISTRY["da3-giant"])).cuda().eval()
    state = {key.removeprefix("model."): value for key, value in load_file(str(_WEIGHTS / "model.safetensors")).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    depth_missing = [key for key in missing if "output_conv2." in key and "aux" not in key]
    if unexpected or depth_missing:
        raise RuntimeError(f"DA3 weight mismatch, missing {depth_missing[:5]}, unexpected {list(unexpected)[:5]}")
    return model


def _estimate(frames: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, model=None) -> np.ndarray:
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    _stub_evo()
    import torch

    from depth_anything_3.utils.geometry import affine_inverse
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.utils.io.output_processor import OutputProcessor

    own = model is None
    if own:
        model = load_model()
    images, extrinsics_t, intrinsics_t = InputProcessor()(
        [frame for frame in frames], extrinsics, intrinsics, num_workers=1
    )
    try:
        with torch.inference_mode():
            raw = model(
                images[None].cuda().float(),
                _normalize_extrinsics(affine_inverse, extrinsics_t),
                intrinsics_t[None].cuda().float(),
                ref_view_strategy="middle",
            )
        prediction = OutputProcessor()(raw)
    finally:
        if own:
            del model
    depth = np.asarray(prediction.depth, dtype=np.float32) / _metric_scale(prediction.extrinsics, extrinsics_t.numpy())
    height, width = frames.shape[1], frames.shape[2]
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    if depth.shape[1:] != (height, width):
        depth = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth])
    return depth


def _stub_evo() -> None:
    """DA3 imports evo while building the unused Gaussian head."""
    import types

    if "evo.core.trajectory" in sys.modules:
        return
    evo = types.ModuleType("evo")
    core = types.ModuleType("evo.core")
    trajectory = types.ModuleType("evo.core.trajectory")
    trajectory.PosePath3D = type("PosePath3D", (), {})
    sys.modules["evo"] = evo
    sys.modules["evo.core"] = core
    sys.modules["evo.core.trajectory"] = trajectory


def _metric_scale(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Scale that takes predicted depth into the EgoDex camera trajectory's meters."""
    source = _centers(reference)
    target = _centers(predicted)
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)
    covariance = target_c.T @ source_c / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1
    variance = float((source_c ** 2).sum() / len(source))
    if variance < 1e-8:
        return 1.0
    return float(np.sum(singular * np.diag(sign)) / variance)


def _centers(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[-2:] == (3, 4):
        padded = np.repeat(np.eye(4)[None], len(extrinsics), axis=0)
        padded[:, :3, :4] = extrinsics
        extrinsics = padded
    rotation = extrinsics[:, :3, :3]
    return -np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), extrinsics[:, :3, 3])


def _normalize_extrinsics(affine_inverse, extrinsics):
    import torch

    poses = extrinsics[None].cuda().float()
    transform = affine_inverse(poses[:, :1])
    normalized = poses @ transform
    cameras = affine_inverse(normalized)
    median = torch.clamp(cameras[..., :3, 3].norm(dim=-1).median(), min=1e-1)
    normalized = normalized.clone()
    normalized[..., :3, 3] = normalized[..., :3, 3] / median
    return normalized


def _invert(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse

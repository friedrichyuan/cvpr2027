"""One pipeline stage. Serial today; the same class is a Ray actor later."""

from __future__ import annotations

import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GRIPPER = "action/gripper.npz"
MASKS = "visual/masks.npz"
INPAINT = "visual/inpaint.mp4"
DEPTH = "visual/depth.npz"
BASE = "action/base.json"
IK = "action/ik.npz"
COMPOSITE = "visual/composite.mp4"
QUALITY = "action/quality.json"


class Step:
    """Subclass `run`. `gpus` is the only scheduling hint a later launcher needs."""

    name: str
    needs: tuple[str, ...] = ()
    makes: tuple[str, ...] = ()
    gpus: float = 0

    def run(self, src: Path, dst: Path) -> None:
        raise NotImplementedError(self.name)

    @classmethod
    def remote(cls):
        """Ray actor class. Call `run` with string paths."""
        import ray

        cached = getattr(cls, "_ray", None)
        if cached is None:
            cached = ray.remote(num_gpus=cls.gpus)(cls)
            cls._ray = cached
        return cached


def run(src: Path | str, dst: Path | str, steps: tuple[type[Step], ...]) -> None:
    """Single process, one GPU stage at a time."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for cls in steps:
        if _done(dst, cls):
            print(f"skip {cls.name}")
            continue
        missing = [name for name in cls.needs if not (dst / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{cls.name} missing {missing}")
        print(f"run {cls.name}")
        cls().run(src, dst)
        missing = [name for name in cls.makes if not (dst / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{cls.name} did not write {missing}")
        if cls.gpus:
            _release_gpu()


def run_remote(src: Path | str, dst: Path | str, steps: tuple[type[Step], ...]) -> None:
    """Same order on one GPU via Ray. Overlap and multi-GPU stay a scheduler change."""
    import ray

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not ray.is_initialized():
        ray.init(num_gpus=1, ignore_reinit_error=True)
    for cls in steps:
        if _done(dst, cls):
            print(f"skip {cls.name}")
            continue
        print(f"run {cls.name}")
        if cls.gpus:
            actor = cls.remote().options(num_gpus=cls.gpus, max_concurrency=1).remote()
            ray.get(actor.run.remote(str(src), str(dst)))
        else:
            cls().run(src, dst)


def _done(dst: Path, cls: type[Step]) -> bool:
    return all((dst / name).is_file() for name in cls.makes)


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_dtype(torch.float32)
    except ImportError:
        pass

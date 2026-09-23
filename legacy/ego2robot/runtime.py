"""Laptop default: one GPU slot, stages never overlap. Optional Ray uses the same slot."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from .processor import EpisodeContext, Processor

if TYPE_CHECKING:
    from .manager import Registry
    from .store import ArtifactStore


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class LocalRuntime:
    def run(self, processor: Processor, ctx: EpisodeContext) -> None:
        processor.run(ctx)
        if processor.resources.kind == "gpu":
            _release_gpu()


class RayRuntime:
    """One `num_gpus=1` actor. CPU processors stay in this process."""

    def __init__(self, registry: Registry, store: ArtifactStore) -> None:
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self._store = store
        actor_cls = _gpu_slot_actor()
        self._gpu = actor_cls.options(num_gpus=1, max_concurrency=1).remote(list(registry.names()))

    def run(self, processor: Processor, ctx: EpisodeContext) -> None:
        if processor.resources.kind != "gpu":
            processor.run(ctx)
            return
        import ray

        ray.get(
            self._gpu.run.remote(
                processor.name,
                {
                    "split": ctx.split,
                    "task": ctx.task,
                    "episode_id": ctx.episode_id,
                    "hdf5": str(ctx.hdf5_path),
                    "video": str(ctx.video_path) if ctx.video_path else None,
                    "out_dir": str(ctx.out_dir),
                    "store_root": str(self._store.root),
                },
            )
        )


def _gpu_slot_actor():
    import ray

    @ray.remote
    class GpuSlotActor:
        def __init__(self, processor_names: list[str]) -> None:
            from ego2robot.registry import build_registry

            registry = build_registry()
            self._processors = {name: registry.get(name) for name in processor_names}

        def run(self, name: str, payload: dict) -> None:
            from pathlib import Path

            from ego2robot.processor import EpisodeContext

            ctx = EpisodeContext(
                split=payload["split"],
                task=payload["task"],
                episode_id=payload["episode_id"],
                hdf5_path=Path(payload["hdf5"]),
                video_path=Path(payload["video"]) if payload["video"] else None,
                out_dir=Path(payload["out_dir"]),
            )
            self._processors[name].run(ctx)
            _release_gpu()

    return GpuSlotActor

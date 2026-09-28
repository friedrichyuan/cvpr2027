"""Ray pools for the episode DAG. One heavy model stays on each GPU."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from egowhale.action.approach import Approach
from egowhale.action.base_ik import BaseIK
from egowhale.action.curate import Curate
from egowhale.action.retarget import Retarget
from egowhale.step import ROOT, _done
from egowhale.visual.composite import Composite
from egowhale.visual.depth import Depth
from egowhale.visual.inpaint import Inpaint
from egowhale.visual.segment import Segment

# Later stages wait on these. The two branches meet at composite.
DEPS = {
    "retarget": (),
    "segment": (),
    "base_ik": ("retarget",),
    "inpaint": ("segment",),
    "approach": ("base_ik",),
    "depth": ("inpaint",),
    "composite": ("retarget", "segment", "base_ik", "inpaint", "approach", "depth"),
    "curate": ("composite",),
}
POOL = {
    "retarget": "action",
    "base_ik": "action",
    "approach": "action",
    "segment": "segment",
    "inpaint": "inpaint",
    "depth": "depth",
    "composite": "cpu",
    "curate": "cpu",
}
STEPS = {
    "retarget": Retarget,
    "segment": Segment,
    "inpaint": Inpaint,
    "depth": Depth,
    "base_ik": BaseIK,
    "approach": Approach,
    "composite": Composite,
    "curate": Curate,
}


class Job:
    def __init__(self, src: Path, dst: Path):
        self.src = src
        self.dst = dst
        self.done = {name for name, cls in STEPS.items() if _done(dst, cls)}
        self.running: set[str] = set()
        self.failed = False
        self.error = ""
        self.started = time.perf_counter()

    def ready(self) -> list[str]:
        if self.failed:
            return []
        return [
            name
            for name, deps in DEPS.items()
            if name not in self.done and name not in self.running and all(dep in self.done for dep in deps)
        ]

    def finished(self) -> bool:
        return self.failed or set(DEPS) <= self.done


def execute(step, src, dst) -> str:
    os.environ["EGOWHALE_QUIET"] = "1"
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if _done(dst, type(step)):
        return ""
    missing = [name for name in step.needs if not (dst / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{step.name} missing {missing}")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        step.run(src, dst)
    missing = [name for name in step.makes if not (dst / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{step.name} did not write {missing}")
    return " ".join(line.strip() for line in buffer.getvalue().splitlines() if line.strip())


def _device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "cpu"
    return "cuda:" + visible.split(",")[0]


def _actors(pools: dict[str, int]):
    import ray

    gpu = ray.remote(num_gpus=1, max_concurrency=1)
    cpu = ray.remote(num_cpus=1, max_concurrency=1)

    @gpu
    class SegmentActor:
        def __init__(self):
            from egowhale.visual.segment import load_predictor

            self.step = Segment()
            self.step._held = load_predictor()

        def ready(self):
            return _device()

        def run(self, src, dst):
            return execute(self.step, src, dst)

    @gpu
    class InpaintActor:
        def __init__(self):
            self.step = Inpaint()
            self.step.load()

        def ready(self):
            return _device()

        def run(self, src, dst):
            return execute(self.step, src, dst)

    @gpu
    class DepthActor:
        def __init__(self):
            from egowhale.visual.depth import load_model

            self.step = Depth()
            self.step._model = load_model()

        def ready(self):
            return _device()

        def run(self, src, dst):
            return execute(self.step, src, dst)

    @gpu
    class ActionActor:
        def __init__(self):
            self.steps = {"retarget": Retarget(), "base_ik": BaseIK(), "approach": Approach()}

        def ready(self):
            return _device()

        def run(self, stage, src, dst):
            return execute(self.steps[stage], src, dst)

    @cpu
    class CpuActor:
        def __init__(self):
            os.environ.pop("EGOWHALE_VLM_API_KEY", None)
            self.steps = {"composite": Composite(), "curate": Curate()}

        def ready(self):
            return _device()

        def run(self, stage, src, dst):
            return execute(self.steps[stage], src, dst)

    classes = {
        "segment": SegmentActor,
        "inpaint": InpaintActor,
        "depth": DepthActor,
        "action": ActionActor,
        "cpu": CpuActor,
    }
    return {
        name: [(f"{name}{index}", cls.remote()) for index in range(pools[name])]
        for name, cls in classes.items()
    }


_PLACED = (
    ("segment", "segment", "SAM 3"),
    ("inpaint", "inpaint", "ProPainter"),
    ("depth", "depth", "DA3-GIANT"),
    ("action", "action", "cuRobo"),
    ("cpu", "composite", "MuJoCo"),
    ("cpu", "curate", "checks"),
)


def _logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> │ <level>{level:<7}</level> │ {message}", colorize=True)
    return logger


def _row(log, status: str, node: str, device: str, model: str) -> None:
    line = f"{status:<10} {node:<22} {device:<16} {model}"
    if status == "Running":
        log.success(line)
    else:
        log.info(line)


def _boot(free: dict, log) -> None:
    import ray

    _row(log, "Status", "Node", "Device", "Model")
    seen = set()
    for pool, _node, _model in _PLACED:
        if pool in seen or pool not in free:
            continue
        seen.add(pool)
        names = ", ".join(name for owner, name, _item in _PLACED if owner == pool)
        models = ", ".join(item for owner, _name, item in _PLACED if owner == pool)
        _row(log, "Starting", names, "", models)
    pending = []
    left = {pool: len(slots) for pool, slots in free.items()}
    found: dict[str, list[str]] = {pool: [] for pool in free}
    for pool, slots in free.items():
        for _label, actor in slots:
            pending.append((pool, actor.ready.remote()))
    while pending:
        done, _rest = ray.wait([item[1] for item in pending], num_returns=1)
        ref = done[0]
        pool, handle = next(item for item in pending if item[1] is ref)
        pending.remove((pool, ref))
        found[pool].append(ray.get(handle))
        left[pool] -= 1
        if left[pool] == 0:
            names = ", ".join(name for owner, name, _item in _PLACED if owner == pool)
            models = ", ".join(item for owner, _name, item in _PLACED if owner == pool)
            devices = ", ".join(dict.fromkeys(found[pool]))
            _row(log, "Running", names, devices, models)


def serve(jobs: list[Job], pools: dict[str, int], inflight: int, log: Path, board: Path) -> None:
    import ray
    from torch.utils.tensorboard import SummaryWriter

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR)
    have = int(ray.cluster_resources().get("GPU", 0))
    need = pools["segment"] + pools["inpaint"] + pools["depth"] + pools["action"]
    if need > have:
        raise SystemExit(f"GPU pools ask for {need} devices, cluster has {have}")
    journal = _logger()
    free = _actors(pools)
    _boot(free, journal)
    watch = _watch(len(jobs))
    journal.info(f"{len(jobs)} episodes    tensorboard {board}")
    journal.info(f"tensorboard --logdir {board}")
    writer = SummaryWriter(log_dir=str(board))
    clock = time.perf_counter()
    frames = 0
    ok = fail = 0
    waiting = list(jobs)
    active: list[Job] = []
    pending = {}

    def submit() -> None:
        while len(active) < inflight and waiting:
            active.append(waiting.pop(0))
        for job in active:
            for stage in job.ready():
                pool = POOL[stage]
                if not free[pool]:
                    continue
                label, actor = free[pool].pop()
                method = actor.run.remote(stage, str(job.src), str(job.dst)) if pool in ("action", "cpu") else actor.run.remote(str(job.src), str(job.dst))
                job.running.add(stage)
                pending[method] = (job, stage, pool, label, actor)

    while waiting or active or pending:
        submit()
        if not pending:
            for job in list(active):
                if job.finished():
                    ok, fail, frames = _record(job, log, writer, clock, ok, fail, frames)
                    active.remove(job)
            if not waiting and not pending:
                break
            continue
        done, _rest = ray.wait(list(pending), num_returns=1)
        ref = done[0]
        job, stage, pool, label, actor = pending.pop(ref)
        try:
            ray.get(ref)
        except Exception:
            job.failed = True
            job.error = f"{stage}: {traceback.format_exc().strip().splitlines()[-1]}"
        else:
            job.done.add(stage)
        job.running.discard(stage)
        free[pool].append((label, actor))
        if job.finished() and job in active:
            ok, fail, frames = _record(job, log, writer, clock, ok, fail, frames)
            active.remove(job)
    writer.flush()
    writer.close()
    elapsed = max(time.perf_counter() - clock, 1e-6)
    journal.info(f"done  {ok} ok  {fail} fail  {frames / elapsed:.2f} fps")
    ray.shutdown()


def _episode(job: Job) -> str:
    return f"{job.src.parent.name}/{job.src.stem}"


def _watch(count: int) -> set[int]:
    if count <= 0:
        return set()
    keep = min(count, max(1, int(round(count * 0.1))))
    indexes = np.unique(np.round(np.linspace(0, count - 1, keep)).astype(int))
    return set(int(index) for index in indexes)


def _record(job: Job, log: Path, writer, clock: float, ok: int, fail: int, frames: int):
    row = {
        "episode": _episode(job),
        "ok": not job.failed,
        "error": job.error,
        "seconds": round(time.perf_counter() - job.started, 2),
    }
    if row["ok"]:
        ok += 1
        frames += job.frames
    else:
        fail += 1
    elapsed = max(time.perf_counter() - clock, 1e-6)
    step = ok + fail
    writer.add_scalar("throughput/fps", frames / elapsed, step)
    writer.add_scalar("progress/ok", ok, step)
    writer.add_scalar("progress/fail", fail, step)
    if row["ok"] and job.watch:
        _compare(job, writer)
    writer.flush()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    return ok, fail, frames


def _compare(job: Job, writer) -> None:
    import cv2

    from egowhale.media import read_rgb
    from egowhale.step import COMPOSITE, INPAINT, PREFIX

    before, _fps = read_rgb(job.dst / INPAINT)
    after, _fps = read_rgb(job.dst / COMPOSITE)
    prefix = len(np.load(job.dst / PREFIX)["qpos"])
    after = after[max(prefix - 1, 0) :]
    count = min(len(before), len(after))
    if count == 0:
        return
    pick = np.unique(np.round(np.linspace(0, count - 1, min(32, count))).astype(int))
    frames = [_pair(_fit(before[index]), _fit(after[index])) for index in pick]
    video = np.stack(frames).transpose(0, 3, 1, 2)[None]
    writer.add_video("compare/" + _episode(job).replace("/", "_"), video, job.index, fps=4)
    writer.flush()


def _fit(frame: np.ndarray) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    if height <= 360:
        return np.ascontiguousarray(frame)
    width = int(width * 360 / height) // 2 * 2
    return cv2.resize(frame, (width, 360), interpolation=cv2.INTER_AREA)


def _pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    import cv2

    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return np.ascontiguousarray(np.concatenate([left, right], axis=1))


def _frame_count(src: Path) -> int:
    import h5py

    with h5py.File(src) as handle:
        return int(handle["transforms/camera"].shape[0])


def simulate(count: int, pools: dict[str, int], inflight: int, fail: dict[tuple[int, str], str] | None = None) -> list[tuple]:
    """Single-threaded stand-in: each pool has N slots and finishes the oldest task first."""
    fail = fail or {}
    jobs = [Job(Path(f"task/{index}.hdf5"), Path(f"out/{index}")) for index in range(count)]
    for job in jobs:
        job.done.clear()
    free = {name: size for name, size in pools.items()}
    waiting = list(jobs)
    active: list[Job] = []
    running: list[tuple] = []
    trace = []
    guard = 0
    while waiting or active or running:
        guard += 1
        if guard > 10000:
            raise RuntimeError("scheduler did not finish")
        while len(active) < inflight and waiting:
            active.append(waiting.pop(0))
        for job in active:
            index = int(job.src.stem)
            for stage in job.ready():
                pool = POOL[stage]
                if free[pool] <= 0:
                    continue
                free[pool] -= 1
                job.running.add(stage)
                running.append((index, stage, pool))
                trace.append((index, stage, "start"))
        if not running:
            for job in list(active):
                if job.finished():
                    active.remove(job)
            if not running and all(job.finished() for job in active):
                break
            if not running:
                raise RuntimeError("no ready stage and nothing running")
            continue
        index, stage, pool = running.pop(0)
        job = jobs[index]
        job.running.discard(stage)
        free[pool] += 1
        reason = fail.get((index, stage))
        if reason:
            job.failed = True
            job.error = reason
            trace.append((index, stage, "fail"))
        else:
            job.done.add(stage)
            trace.append((index, stage, "end"))
        if job.finished() and job in active:
            active.remove(job)
    return trace


def episodes_from(paths: list[Path], limit: int) -> list[Path]:
    found = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(item for item in path.rglob("*.hdf5") if item.with_suffix(".mp4").is_file()))
        else:
            found.append(path)
    if limit:
        found = found[:limit]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the episode DAG on Ray pools.")
    parser.add_argument("paths", nargs="+", type=Path, help="HDF5 files or a dataset directory")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--inflight", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--inpaint", type=int, default=1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--action", type=int, default=1)
    args = parser.parse_args()
    sources = episodes_from([path.expanduser().resolve() for path in args.paths], args.limit)
    if not sources:
        raise SystemExit("no episodes")
    out = args.out.expanduser().resolve()
    jobs = [Job(src, out / src.parent.name / src.stem) for src in sources]
    pools = {
        "segment": args.segment,
        "inpaint": args.inpaint,
        "depth": args.depth,
        "action": args.action,
        "cpu": args.inflight,
    }
    watch = _watch(len(jobs))
    for index, job in enumerate(jobs):
        job.index = index
        job.frames = _frame_count(job.src)
        job.watch = index in watch
    serve(jobs, pools, args.inflight, out / "batch.jsonl", out / "tb")


if __name__ == "__main__":
    main()

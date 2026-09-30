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
import warnings
from pathlib import Path

# moviepy 1.0.3 predates Python 3.12 and warns on its own regex literals.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"moviepy(\.|$)")

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
from egowhale.visual.render import Render
from egowhale.visual.segment import Segment

# Later stages wait on these. Only inpaint waits inside the visual branch; everything meets at composite.
DEPS = {
    "retarget": (),
    "segment": (),
    "base_ik": ("retarget",),
    "inpaint": ("segment",),
    "approach": ("base_ik",),
    "depth": (),
    "render": ("retarget", "base_ik", "approach"),
    "composite": ("segment", "inpaint", "depth", "approach", "render"),
    "curate": ("composite",),
}
POOL = {
    "retarget": "action",
    "base_ik": "action",
    "approach": "action",
    "segment": "segment",
    "inpaint": "inpaint",
    "depth": "depth",
    "render": "render",
    "composite": "cpu",
    "curate": "cpu",
}
VISUAL = ("segment", "inpaint", "depth")
# Frames packed into one forward. A single longer episode still runs alone. OOM halves the budget.
FRAME_BUDGET = {"inpaint": 320, "depth": 192}
STEPS = {
    "retarget": Retarget,
    "segment": Segment,
    "inpaint": Inpaint,
    "depth": Depth,
    "base_ik": BaseIK,
    "approach": Approach,
    "render": Render,
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
        self.ran = False
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


def execute(step, src, dst) -> tuple[bool, float]:
    """Run one stage. Return whether it computed, and how long that took. Skips are (False, 0)."""
    os.environ["EGOWHALE_QUIET"] = "1"
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if _done(dst, type(step)):
        return False, 0.0
    missing = [name for name in step.needs if not (dst / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{step.name} missing {missing}")
    buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        step.run(src, dst)
    missing = [name for name in step.makes if not (dst / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{step.name} did not write {missing}")
    return True, time.perf_counter() - started


def _device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "cpu"
    return "cuda:" + visible.split(",")[0]


def _egl(device: str) -> None:
    """MuJoCo EGL ignores CUDA_VISIBLE_DEVICES and would render on physical GPU 0."""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = device


def _actors(pools: dict[str, int], share: dict[str, float]):
    """Every actor lives for the whole run. `share` is the GPU fraction for pools that stack on one card."""
    import ray

    gpu = ray.remote(num_gpus=1, max_concurrency=1)
    # The next episode waits on the compute lock while this one copies its result out.
    pipe = ray.remote(num_gpus=1, max_concurrency=2)
    cpu = ray.remote(num_cpus=1, max_concurrency=1)

    @pipe
    class SegmentActor:
        def __init__(self):
            from egowhale.visual.segment import load_tracker

            self.step = Segment()
            self.step._processor, self.step._cutie = load_tracker()

        def ready(self):
            return _device()

        def run(self, src, dst):
            return execute(self.step, src, dst)

        def consume(self, payload):
            return self.step.consume(payload)

    @pipe
    class InpaintActor:
        def __init__(self):
            self.step = Inpaint()
            self.step.load()

        def ready(self):
            return _device()

        def run(self, src, dst):
            return execute(self.step, src, dst)

        def consume(self, payload, budget):
            return self.step.consume(payload, budget)

    @pipe
    class DepthActor:
        def __init__(self):
            from egowhale.visual.depth import load_model

            self.step = Depth()
            self.step._model = load_model()

        def ready(self):
            return _device()

        def run(self, src, dst):
            return execute(self.step, src, dst)

        def consume(self, payload, budget):
            return self.step.consume(payload, budget)

    @gpu
    class ActionActor:
        def __init__(self):
            self.steps = {"retarget": Retarget(), "base_ik": BaseIK(), "approach": Approach()}

        def ready(self):
            return _device()

        def run(self, stage, src, dst):
            return execute(self.steps[stage], src, dst)

        def compute(self, stage, dst, data):
            import ray

            started = time.perf_counter()
            try:
                out = self.steps[stage].compute(data)
            except Exception as exc:
                return {"ok": False, "error": str(exc).splitlines()[-1], "seconds": 0.0, "save_ref": None}
            packed = {"stage": stage, "dst": dst, "out": out}
            return {"ok": True, "seconds": time.perf_counter() - started, "save_ref": ray.put(packed)}

    @cpu
    class CpuActor:
        def __init__(self):
            os.environ.pop("EGOWHALE_VLM_API_KEY", None)
            self.steps = {"composite": Composite(), "curate": Curate()}

        def ready(self):
            return "cpu"

        def egl(self, device):
            _egl(device)

        def run(self, stage, src, dst):
            return execute(self.steps[stage], src, dst)

    @cpu
    class RenderActor:
        def __init__(self):
            self.step = Render()

        def ready(self):
            return "cpu"

        def egl(self, device):
            _egl(device)

        def run(self, stage, src, dst):
            return execute(self.step, src, dst)

    @ray.remote(num_cpus=2, max_concurrency=8)
    class FeedActor:
        def ready(self):
            return "cpu"

        def prepare(self, stage, pairs):
            from egowhale.visual.feed import prepare

            return prepare(stage, pairs)

    @ray.remote(num_cpus=1, max_concurrency=1)
    class StoreActor:
        def ready(self):
            return "cpu"

        def commit(self, saves):
            from egowhale.visual.store import commit

            commit(saves)
            return True

    @ray.remote(num_cpus=1, max_concurrency=1)
    class ActionReadActor:
        def ready(self):
            return "cpu"

        def load(self, stage, src, dst):
            src, dst = Path(src), Path(dst)
            if stage == "retarget":
                from egowhale.action.retarget import read_episode

                return read_episode(src)
            if stage == "base_ik":
                from egowhale.action.base_ik import read_gripper

                return read_gripper(dst)
            if stage == "approach":
                from egowhale.action.approach import read_ik

                return read_ik(dst)
            raise ValueError(stage)

    @ray.remote(num_cpus=1, max_concurrency=1)
    class ActionWriteActor:
        def ready(self):
            return "cpu"

        def save(self, packed):
            dst = Path(packed["dst"])
            stage = packed["stage"]
            out = packed["out"]
            if stage == "retarget":
                from egowhale.action.retarget import save_gripper

                save_gripper(dst, out)
            elif stage == "base_ik":
                from egowhale.action.base_ik import save_base

                save_base(dst, out)
            elif stage == "approach":
                from egowhale.action.approach import save_prefix

                save_prefix(dst, out)
            else:
                raise ValueError(stage)
            return True

    classes = {
        "segment": SegmentActor,
        "inpaint": InpaintActor,
        "depth": DepthActor,
        "action": ActionActor,
        "cpu": CpuActor,
        "render": RenderActor,
        "feed": FeedActor,
        "write": StoreActor,
        "action_read": ActionReadActor,
        "action_write": ActionWriteActor,
    }
    made = {}

    def make(names) -> None:
        for name in names:
            cls = classes[name].options(num_gpus=share[name]) if name in share else classes[name]
            made[name] = [(f"{name}{index}", cls.remote()) for index in range(pools[name])]

    # Whole cards are placed before fractional ones so a shared card cannot take a slot a whole actor needs.
    make([name for name in classes if name not in share])
    ray.get([actor.ready.remote() for name in ("inpaint", "depth") for _label, actor in made.get(name, [])])
    for name in share:
        make([name])
        ray.get([actor.ready.remote() for _label, actor in made[name]])
    return {name: made[name] for name in classes}


_PLACED = (
    ("segment", "segment", "SAM 3 + Cutie"),
    ("inpaint", "inpaint", "ProPainter"),
    ("depth", "depth", "DA3-GIANT"),
    ("action", "action", "cuRobo"),
    ("render", "render", "MuJoCo"),
    ("cpu", "composite", "paste"),
    ("cpu", "curate", "checks"),
    ("feed", "feed", "prefetch"),
    ("write", "write", "store"),
    ("action_read", "action-read", "prefetch"),
    ("action_write", "action-write", "store"),
)


def _logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> │ <level>{level:<7}</level> │ {message}", colorize=True)
    return logger


def _row(log, status: str, node: str, device: str, model: str) -> None:
    line = f"{status:<10} │ {node:<22} │ {device:<16} │ {model}"
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


def _take(active: list[Job], stage: str, limit: int) -> list[Job]:
    """Whole episodes, oldest first. A node then adds later episodes of the same length until it is full."""
    ready = [job for job in active if stage in job.ready()]
    if not ready or stage in ("segment", "inpaint"):
        return ready[:1]
    length = ready[0].frames
    chosen: list[Job] = []
    total = 0
    for job in ready:
        if job.frames != length:
            continue
        if chosen and total + job.frames > limit:
            break
        chosen.append(job)
        total += job.frames
    return chosen


def _apply(jobs, rows, stage, meter, writer, log, active) -> None:
    noted = False
    for job, row in zip(jobs, rows):
        ran, seconds = row[0], row[1]
        error = row[2] if len(row) > 2 else ""
        if error:
            job.failed = True
            job.error = f"{stage}: {error}"
        else:
            job.done.add(stage)
            if ran:
                job.ran = True
                meter.note(stage, job.frames, seconds)
                noted = True
        job.running.discard(stage)
        if job.finished() and job in active:
            _record(job, log, writer, meter)
            active.remove(job)
    if noted:
        meter.write(writer)


def serve(jobs: list[Job], pools: dict[str, int], share: dict[str, float], inflight: int, log: Path, board: Path) -> None:
    import ray
    from torch.utils.tensorboard import SummaryWriter

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR)
    have = int(ray.cluster_resources().get("GPU", 0))
    need = round(sum(pools[name] * share.get(name, 1.0) for name in ("segment", "inpaint", "depth", "action")))
    if need > have:
        raise SystemExit(f"GPU pools ask for {need} devices, cluster has {have}")
    journal = _logger()
    free = _actors(pools, share)
    _boot(free, journal)
    # Render and curate draw with EGL on the action cards, which cuRobo leaves mostly idle.
    cards = sorted({device.split(":")[-1] for device in ray.get([actor.ready.remote() for _label, actor in free["action"]])})
    ray.get([actor.egl.remote(cards[index % len(cards)]) for pool in ("render", "cpu") for index, (_label, actor) in enumerate(free[pool])])
    journal.info(f"{len(jobs)} episodes    tensorboard {board}")
    journal.info(f"tensorboard --logdir {board}")
    writer = SummaryWriter(log_dir=str(board))
    meter = _Meter()
    waiting = list(jobs)
    active: list[Job] = []
    pending = {}
    budget = dict(FRAME_BUDGET)
    readers = [actor for _label, actor in free.pop("feed")]
    writers = [actor for _label, actor in free.pop("write")]
    lanes = []
    slot = 0
    for stage in VISUAL:
        for label, actor in free.pop(POOL[stage]):
            lanes.append(
                {
                    "stage": stage,
                    "label": label,
                    "actor": actor,
                    "readers": readers[slot : slot + 2],
                    "writer": writers[slot // 2],
                    "busy": {},
                    "ready": [],
                    "inflight": [],
                }
            )
            slot += 2
    action_lanes = []
    for (_label, actor), (_rl, reader), (_wl, store) in zip(free.pop("action"), free.pop("action_read"), free.pop("action_write")):
        action_lanes.append(
            {
                "actor": actor,
                "reader": reader,
                "writer": store,
                "read": None,
                "gpu": None,
                "job": None,
                "stage": None,
                "next_read": None,
                "next_job": None,
                "next_stage": None,
                "next_data": None,
            }
        )

    def consume(lane, data, chosen) -> None:
        if lane["stage"] == "segment":
            launched = lane["actor"].consume.remote(data)
        else:
            launched = lane["actor"].consume.remote(data, budget[lane["stage"]])
        lane["inflight"].append(launched)
        pending[launched] = ("gpu", lane, chosen)

    def load(lane) -> int:
        return len(lane["busy"]) + len(lane["ready"]) + len(lane["inflight"])

    def fill_one(lane) -> bool:
        """Start decoding one more episode for this lane, if it has room and work exists."""
        if load(lane) >= 4:
            return False
        reader = next((item for item in lane["readers"] if item not in lane["busy"].values()), None)
        if reader is None:
            return False
        stage = lane["stage"]
        chosen = _take(active, stage, budget.get(stage, 10**9))
        if not chosen:
            return False
        for job in chosen:
            job.running.add(stage)
        ref = reader.prepare.remote(stage, [(str(job.src), str(job.dst)) for job in chosen])
        lane["busy"][ref] = reader
        pending[ref] = ("feed", lane, chosen)
        return True

    def fill() -> None:
        """One episode on the GPU, one queued, both readers decoding. Scarce work goes to the least loaded lane first."""
        progress = True
        while progress:
            progress = False
            for lane in sorted(lanes, key=load):
                progress |= fill_one(lane)

    def launch(lane) -> None:
        while len(lane["inflight"]) < 2 and lane["ready"]:
            data, chosen = lane["ready"].pop(0)
            consume(lane, data, chosen)

    def promote_action(lane) -> None:
        job = lane["next_job"]
        stage = lane["next_stage"]
        data = lane["next_data"]
        lane["next_job"] = None
        lane["next_stage"] = None
        lane["next_data"] = None
        lane["job"] = job
        lane["stage"] = stage
        launched = lane["actor"].compute.remote(stage, str(job.dst), data)
        lane["gpu"] = launched
        pending[launched] = ("action", lane)

    def pump() -> None:
        while len(active) < inflight and waiting:
            active.append(waiting.pop(0))
        fill()
        for lane in lanes:
            launch(lane)
        held = set()
        for lane in action_lanes:
            if lane["job"] is not None:
                held.add(id(lane["job"]))
            if lane["next_job"] is not None:
                held.add(id(lane["next_job"]))
        for lane in action_lanes:
            idle = lane["read"] is None and lane["gpu"] is None and lane["job"] is None and lane["next_read"] is None and lane["next_data"] is None
            want_next = lane["gpu"] is not None and lane["next_read"] is None and lane["next_data"] is None
            if not idle and not want_next:
                continue
            chosen = None
            for job in active:
                if id(job) in held:
                    continue
                for stage in job.ready():
                    if POOL[stage] != "action":
                        continue
                    chosen = (job, stage)
                    break
                if chosen:
                    break
            if chosen is None:
                continue
            job, stage = chosen
            job.running.add(stage)
            held.add(id(job))
            ref = lane["reader"].load.remote(stage, str(job.src), str(job.dst))
            if idle:
                lane["read"] = ref
                lane["job"] = job
                lane["stage"] = stage
                pending[ref] = ("aread", lane)
            else:
                lane["next_read"] = ref
                lane["next_job"] = job
                lane["next_stage"] = stage
                pending[ref] = ("anext", lane)
        for job in active:
            for stage in job.ready():
                pool = POOL[stage]
                if stage in VISUAL or pool == "action":
                    continue
                if not free[pool]:
                    continue
                label, actor = free[pool].pop()
                method = actor.run.remote(stage, str(job.src), str(job.dst))
                job.running.add(stage)
                pending[method] = ("run", job, stage, pool, label, actor)

    while waiting or active or pending:
        pump()
        if not pending:
            for job in list(active):
                if job.finished():
                    _record(job, log, writer, meter)
                    active.remove(job)
            if not waiting and not pending:
                break
            continue
        ready_refs, _rest = ray.wait(list(pending), num_returns=1, fetch_local=False)
        ref = ready_refs[0]
        kind = pending.pop(ref)
        if kind[0] == "feed":
            lane, chosen = kind[1], kind[2]
            lane["busy"].pop(ref, None)
            lane["ready"].append((ref, chosen))
            launch(lane)
            fill()
            continue
        if kind[0] == "aread":
            lane = kind[1]
            lane["read"] = None
            launched = lane["actor"].compute.remote(lane["stage"], str(lane["job"].dst), ref)
            lane["gpu"] = launched
            pending[launched] = ("action", lane)
            continue
        if kind[0] == "anext":
            lane = kind[1]
            lane["next_read"] = None
            lane["next_data"] = ref
            if lane["gpu"] is None:
                promote_action(lane)
            continue
        if kind[0] == "run":
            _job, stage, pool, label, actor = kind[1:]
            try:
                ran, seconds = ray.get(ref)
                rows = [(ran, seconds, "")]
            except Exception:
                message = traceback.format_exc().strip().splitlines()[-1]
                rows = [(False, 0.0, message)]
            _apply([_job], rows, stage, meter, writer, log, active)
            free[pool].append((label, actor))
            continue
        if kind[0] in ("write", "awrite"):
            stage, chosen, rows = kind[1:]
            try:
                ray.get(ref)
            except Exception:
                message = traceback.format_exc().strip().splitlines()[-1]
                rows = [(False, 0.0, message)] * len(chosen)
            _apply(chosen, rows, stage, meter, writer, log, active)
            continue
        if kind[0] == "action":
            lane = kind[1]
            job = lane["job"]
            stage = lane["stage"]
            lane["gpu"] = None
            lane["job"] = None
            lane["stage"] = None
            if lane["next_data"] is not None:
                promote_action(lane)
            try:
                payload = ray.get(ref)
            except Exception:
                message = traceback.format_exc().strip().splitlines()[-1]
                _apply([job], [(False, 0.0, message)], stage, meter, writer, log, active)
                continue
            if not payload["ok"]:
                _apply([job], [(False, 0.0, payload["error"])], stage, meter, writer, log, active)
                continue
            rows = [(True, payload["seconds"], "")]
            if payload["save_ref"] is None:
                _apply([job], rows, stage, meter, writer, log, active)
                continue
            stored = lane["writer"].save.remote(payload["save_ref"])
            pending[stored] = ("awrite", stage, [job], rows)
            continue
        lane, chosen = kind[1], list(kind[2])
        stage = lane["stage"]
        lane["inflight"] = [item for item in lane["inflight"] if item != ref]
        launch(lane)
        fill()
        try:
            payload = ray.get(ref)
        except Exception:
            message = traceback.format_exc().strip().splitlines()[-1]
            _apply(chosen, [(False, 0.0, message)] * len(chosen), stage, meter, writer, log, active)
            continue
        rows = payload["results"]
        if len(rows) != len(chosen):
            rows = [(False, 0.0, "batch result mismatch")] * len(chosen)
        elif stage in budget and "budget" in payload:
            budget[stage] = min(budget[stage], int(payload["budget"]))
        save_ref = payload.get("save_ref")
        if save_ref is None:
            _apply(chosen, rows, stage, meter, writer, log, active)
            continue
        stored = lane["writer"].commit.remote(save_ref)
        pending[stored] = ("write", stage, chosen, rows)
    writer.flush()
    writer.close()
    elapsed = max(time.perf_counter() - meter.wall0, 1e-6)
    journal.info(f"done  {meter.ok} ok  {meter.fail} fail  {meter.frames / elapsed:.2f} fps")
    ray.shutdown()


def _episode(job: Job) -> str:
    return f"{job.src.parent.name}/{job.src.stem}"


def _watch(count: int) -> set[int]:
    if count <= 0:
        return set()
    keep = min(count, max(1, int(round(count * 0.1))))
    indexes = np.unique(np.round(np.linspace(0, count - 1, keep)).astype(int))
    return set(int(index) for index in indexes)


class _Meter:
    """system/* counts this run only. node/* is each stage's own speed. Skips add episodes, not frames."""

    def __init__(self):
        self.wall0 = time.perf_counter()
        self.episodes = 0
        self.ok = 0
        self.fail = 0
        self.frames = 0
        self.nodes = {name: {"frames": 0, "seconds": 0.0, "episodes": 0} for name in STEPS}
        self._last = -1

    def note(self, stage: str, frames: int, seconds: float) -> None:
        node = self.nodes[stage]
        node["frames"] += frames
        node["seconds"] += seconds
        node["episodes"] += 1

    def finish(self, job: Job) -> None:
        self.episodes += 1
        if job.failed:
            self.fail += 1
            return
        self.ok += 1
        if job.ran:
            self.frames += job.frames

    def write(self, writer) -> None:
        step = self._step()
        writer.add_scalar("system/episodes", self.episodes, step)
        writer.add_scalar("system/frames", self.frames, step)
        elapsed = time.perf_counter() - self.wall0
        if self.frames and elapsed >= 1:
            writer.add_scalar("system/fps", self.frames / elapsed, step)
        for name, node in self.nodes.items():
            if not node["episodes"]:
                continue
            writer.add_scalar(f"node/{name}/episodes", node["episodes"], step)
            writer.add_scalar(f"node/{name}/frames", node["frames"], step)
            if node["seconds"] >= 0.05:
                writer.add_scalar(f"node/{name}/fps", node["frames"] / node["seconds"], step)

    def _step(self) -> int:
        """TensorBoard step is wall seconds since boot. Milliseconds made 100s look like 100000."""
        step = int(time.perf_counter() - self.wall0)
        if step <= self._last:
            step = self._last + 1
        self._last = step
        return step


def _record(job: Job, log: Path, writer, meter: _Meter) -> None:
    row = {
        "episode": _episode(job),
        "ok": not job.failed,
        "error": job.error,
        "seconds": round(time.perf_counter() - job.started, 2),
    }
    meter.finish(job)
    meter.write(writer)
    if row["ok"] and job.watch:
        _compare(job, writer)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


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
    parser.add_argument("--inflight", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--inpaint", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--action", type=int, default=1)
    parser.add_argument("--segment-actors", type=int, default=2, help="segment actors sharing each segment card")
    parser.add_argument("--action-actors", type=int, default=4, help="action actors sharing each action card")
    parser.add_argument("--render", type=int, default=4, help="MuJoCo render actors")
    args = parser.parse_args()
    sources = episodes_from([path.expanduser().resolve() for path in args.paths], args.limit)
    if not sources:
        raise SystemExit("no episodes")
    out = args.out.expanduser().resolve()
    jobs = [Job(src, out / src.parent.name / src.stem) for src in sources]
    segment = args.segment * args.segment_actors
    action = args.action * args.action_actors
    visual = segment + args.inpaint + args.depth
    pools = {
        "segment": segment,
        "inpaint": args.inpaint,
        "depth": args.depth,
        "action": action,
        "cpu": min(args.inflight, 8),
        "render": args.render,
        "feed": visual * 2,
        "write": visual,
        "action_read": action,
        "action_write": action,
    }
    share = {"segment": 1.0 / args.segment_actors, "action": 1.0 / args.action_actors}
    watch = _watch(len(jobs))
    for index, job in enumerate(jobs):
        job.index = index
        job.frames = _frame_count(job.src)
        job.watch = index in watch
    serve(jobs, pools, share, args.inflight, out / "batch.jsonl", out / "tb")


if __name__ == "__main__":
    main()

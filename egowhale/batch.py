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

        def run(self, src, dst):
            return execute(self.step, src, dst)

    @gpu
    class InpaintActor:
        def __init__(self):
            self.step = Inpaint()

        def run(self, src, dst):
            return execute(self.step, src, dst)

    @gpu
    class DepthActor:
        def __init__(self):
            from egowhale.visual.depth import load_model

            self.step = Depth()
            self.step._model = load_model()

        def run(self, src, dst):
            return execute(self.step, src, dst)

    @gpu
    class ActionActor:
        def __init__(self):
            self.steps = {"retarget": Retarget(), "base_ik": BaseIK(), "approach": Approach()}

        def run(self, stage, src, dst):
            return execute(self.steps[stage], src, dst)

    @cpu
    class CpuActor:
        def __init__(self):
            os.environ.pop("EGOWHALE_VLM_API_KEY", None)
            self.steps = {"composite": Composite(), "curate": Curate()}

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


_LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> │ {extra[progress]} │ <level>{extra[verb]}</level>"
    " │ {extra[actor]} │ {extra[stage]} │ {extra[episode]} │ {extra[elapsed]} │ {message}"
)
_LEVEL = {"START": "INFO", "DONE": "SUCCESS", "FAIL": "ERROR", "OK": "SUCCESS"}


def _col(text: str, width: int) -> str:
    text = text or ""
    if len(text) <= width:
        return text.ljust(width)
    return text[: width - 2] + ".."


class Log:
    """Fixed columns: progress, verb, actor, stage, episode, elapsed, detail."""

    def __init__(self, path: Path, total: int):
        from loguru import logger

        self.total = total
        self.ok = 0
        self.fail = 0
        self._logger = logger
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.add(sys.stderr, format=_LOG_FORMAT, colorize=True)
        logger.add(path, format=_LOG_FORMAT, colorize=False, encoding="utf-8")

    def event(self, kind: str, label: str, stage: str, episode: str, extra: str = "") -> None:
        if kind == "start":
            verb, elapsed, detail = "START", "", ""
        elif kind == "done":
            verb = "DONE"
            elapsed, separated, detail = (extra or "").partition("  ")
            if not separated:
                elapsed, detail = extra or "", ""
        elif kind == "ok":
            verb, elapsed, detail = "OK", extra, ""
        else:
            verb = "FAIL"
            elapsed, separated, detail = (extra or "").partition("  ")
            if not separated or not elapsed.endswith("s"):
                elapsed, detail = "", extra
        done = f"{self.ok + self.fail}/{self.total}"
        self._logger.bind(
            progress=_col(done, 9),
            verb=_col(verb, 5),
            actor=_col("" if label == "-" else label, 9),
            stage=_col("episode" if stage == "-" else stage, 10),
            episode=_col(episode, 28),
            elapsed=_col(elapsed, 8),
        ).log(_LEVEL[verb], detail)


def serve(jobs: list[Job], pools: dict[str, int], inflight: int, log: Path) -> None:
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR)
    have = int(ray.cluster_resources().get("GPU", 0))
    need = pools["segment"] + pools["inpaint"] + pools["depth"] + pools["action"]
    if need > have:
        raise SystemExit(f"GPU pools ask for {need} devices, cluster has {have}")
    free = _actors(pools)
    journal = Log(log.with_suffix(".log"), len(jobs))
    waiting = list(jobs)
    active: list[Job] = []
    pending = {}

    def submit() -> None:
        while len(active) < inflight and waiting:
            active.append(waiting.pop(0))
        for job in active:
            episode = _episode(job)
            for stage in job.ready():
                pool = POOL[stage]
                if not free[pool]:
                    continue
                label, actor = free[pool].pop()
                method = actor.run.remote(stage, str(job.src), str(job.dst)) if pool in ("action", "cpu") else actor.run.remote(str(job.src), str(job.dst))
                job.running.add(stage)
                pending[method] = (job, stage, pool, label, actor, time.perf_counter())
                journal.event("start", label, stage, episode)

    while waiting or active or pending:
        submit()
        if not pending:
            for job in list(active):
                if job.finished():
                    _record(job, log, journal)
                    active.remove(job)
            if not waiting and not pending:
                break
            continue
        done, _rest = ray.wait(list(pending), num_returns=1)
        ref = done[0]
        job, stage, pool, label, actor, started = pending.pop(ref)
        episode = _episode(job)
        try:
            summary = ray.get(ref) or ""
        except Exception:
            job.failed = True
            job.error = f"{stage}: {traceback.format_exc().strip().splitlines()[-1]}"
            journal.event("fail", label, stage, episode, job.error)
        else:
            job.done.add(stage)
            elapsed = f"{time.perf_counter() - started:.1f}s"
            journal.event("done", label, stage, episode, f"{elapsed}  {summary}".rstrip())
        job.running.discard(stage)
        free[pool].append((label, actor))
        if job.finished() and job in active:
            _record(job, log, journal)
            active.remove(job)
    ray.shutdown()


def _episode(job: Job) -> str:
    return f"{job.src.parent.name}/{job.src.stem}"


def _record(job: Job, log: Path, journal: Log) -> None:
    row = {
        "episode": _episode(job),
        "ok": not job.failed,
        "error": job.error,
        "seconds": round(time.perf_counter() - job.started, 2),
    }
    if row["ok"]:
        journal.ok += 1
    else:
        journal.fail += 1
    if row["ok"]:
        journal.event("ok", "", "episode", row["episode"], f"{row['seconds']:.1f}s")
    else:
        journal.event("fail", "", "episode", row["episode"], f"{row['seconds']:.1f}s  {row['error']}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


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
    print(f"{len(jobs)} episodes  inflight {args.inflight}  pools {pools}", flush=True)
    serve(jobs, pools, args.inflight, out / "batch.jsonl")


if __name__ == "__main__":
    main()

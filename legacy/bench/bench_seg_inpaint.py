"""Segment then inpaint only, on the production scheduler. Finds the card split that keeps both busy.

    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python legacy/bench/bench_seg_inpaint.py \
        --segment 1 --segment-actors 2 --inpaint 5 --out outputs/bench_si_1x2_5 <dataset dirs...>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from egowhale import batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--segment-actors", type=int, default=2)
    parser.add_argument("--inpaint", type=int, default=4)
    parser.add_argument("--inflight", type=int, default=80)
    args = parser.parse_args()

    batch.DEPS = {"segment": (), "inpaint": ("segment",)}
    batch.STEPS = {"segment": batch.STEPS["segment"], "inpaint": batch.STEPS["inpaint"]}
    batch.VISUAL = ("segment", "inpaint")

    sources = batch.episodes_from([path.resolve() for path in args.paths], 0)
    out = args.out.resolve()
    jobs = [batch.Job(src, out / src.parent.name / src.stem) for src in sources]
    for index, job in enumerate(jobs):
        job.index = index
        job.frames = batch._frame_count(job.src)
        job.watch = False
    segment = args.segment * args.segment_actors
    visual = segment + args.inpaint
    pools = {
        "segment": segment,
        "inpaint": args.inpaint,
        "depth": 0,
        # The scheduler expects one action actor; it gets a sliver of a card and no work.
        "action": 1,
        "cpu": 1,
        "render": 0,
        "feed": visual * 2,
        "write": visual,
        "action_read": 1,
        "action_write": 1,
    }
    # Leave 0.01 of the segment card so the idle action actor has somewhere to sit.
    share = {"segment": 0.99 / args.segment_actors, "action": 0.01}
    batch.serve(jobs, pools, share, args.inflight, out / "batch.jsonl", out / "tb")


if __name__ == "__main__":
    main()

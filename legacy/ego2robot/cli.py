"""Catalog EgoDex episodes and run a processor profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import read_index, scan_egodex, to_context, write_index
from .manager import PipelineManager
from .registry import PROFILES, build_registry
from .runtime import LocalRuntime, RayRuntime
from .store import ArtifactStore

DEFAULT_INPUT = Path("/home/ymq/code/EGODEX_DATASET")
DEFAULT_OUTPUT = Path("outputs/ego2robot/egodex")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog", help="Write an episode index jsonl")
    catalog.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    catalog.add_argument("--index", type=Path, default=DEFAULT_OUTPUT / "index.jsonl")
    run = sub.add_parser("run", help="Run enabled processors on indexed episodes")
    run.add_argument("--index", type=Path, default=DEFAULT_OUTPUT / "index.jsonl")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--profile", default="path_a_egodex", choices=sorted(PROFILES))
    run.add_argument("--processors", nargs="+", default=None, help="Override profile")
    run.add_argument("--limit", type=int, default=1)
    run.add_argument("--task", default=None, help="Only this EgoDex task directory")
    run.add_argument("--episode", default=None, help="Only this episode id")
    run.add_argument("--backend", choices=("local", "ray"), default="local")
    args = parser.parse_args(argv)

    if args.command == "catalog":
        records = scan_egodex(args.input)
        write_index(records, args.index)
        print(f"Wrote {len(records)} episodes to {args.index}")
        return

    records = read_index(args.index)
    if args.task:
        records = [record for record in records if record.task == args.task]
    if args.episode:
        records = [record for record in records if record.episode_id == args.episode]
    records = records[: args.limit]
    if not records:
        raise SystemExit(f"Index is empty: {args.index}")
    registry = build_registry()
    store = ArtifactStore(args.output)
    runtime = RayRuntime(registry, store) if args.backend == "ray" else LocalRuntime()
    manager = PipelineManager(registry, store, runtime)
    manager.enable(list(args.processors or PROFILES[args.profile]))
    print("schedule:", " -> ".join(processor.name for processor in manager.schedule()))
    for record in records:
        ctx = to_context(record, args.output)
        print(f"run {ctx.key}")
        manager.run_episode(ctx)
        print(f"done {ctx.out_dir}")


if __name__ == "__main__":
    main()

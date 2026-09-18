"""Batch-export deterministic ARX IK references from EgoDex HDF5 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .defaults import DEFAULT_SCENE
from .geometry import DEFAULT_SCENE_ANCHOR
from .reference import build_reference, reference_summary, save_reference

DEFAULT_INPUT_ROOT = Path("/home/ymq/code/EGODEX_DATASET/test")
DEFAULT_OUTPUT_ROOT = Path("references/arx5_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--task", default="stack", help="Task directory below --input-root")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of episodes")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--scene-anchor",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=tuple(DEFAULT_SCENE_ANCHOR),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = (args.input_root / args.task).expanduser().resolve()
    output_dir = (args.output_root / args.task).expanduser().resolve()
    episodes = sorted(input_dir.glob("*.hdf5"))
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise FileNotFoundError(f"No HDF5 episodes found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for index, episode_path in enumerate(episodes, start=1):
            output_path = output_dir / f"{episode_path.stem}.npz"
            if output_path.exists() and args.skip_existing:
                print(f"[{index}/{len(episodes)}] skip {episode_path.name}")
                continue
            reference = build_reference(
                episode_path,
                scene_path=args.scene,
                scene_anchor=np.asarray(args.scene_anchor, dtype=np.float64),
            )
            save_reference(reference, output_path)
            record = {
                "source": str(episode_path),
                "reference": str(output_path),
                **reference_summary(reference),
            }
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
            manifest.flush()
            print(
                f"[{index}/{len(episodes)}] {episode_path.name}: "
                f"{record['ik_converged_fraction']:.1%} strict IK convergence"
            )


if __name__ == "__main__":
    main()

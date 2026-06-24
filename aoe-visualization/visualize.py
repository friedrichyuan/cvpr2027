# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python3
"""AoE-Visualization command-line entry point.

Render the end-to-end visualization (ego video + annotation panel + camera-frame
MANO mesh & keypoint overlay + future wrist trails + world-frame 3D panel +
unified header bar + bottom timeline scrubber) for one sample or a whole delivery
directory. Each sample produces a single video, ``AoE_output_vis.mp4``.

All rendering options use sensible baked-in defaults (full-resolution row height,
every frame, world + trails + mesh + keypoints on, H.264 transcode on), so the
CLI only takes an input and an output location.

Examples
--------
    # single sample
    python visualize.py --sample /path/to/delivery/data/<sample_name>

    # whole delivery directory
    python visualize.py --data_dir /path/to/delivery/data --output_dir ./output

The hand mesh uses a high-quality offscreen OpenGL renderer when its GL stack is
available, and transparently falls back to a pure-NumPy shaded mesh otherwise.
This tool imports only from ``aoe_vis`` and loads the vendored MANO model from
``assets/mano``; it does not import any external pipeline code at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from aoe_vis import render as render_mod          # noqa: E402
from aoe_vis.sample import SamplePaths            # noqa: E402


def find_samples(data_dir: str) -> list:
    out = []
    for child in sorted(Path(data_dir).iterdir()):
        if child.is_dir() and (child / "ego_annotation" /
                               "ego_action_annotation.json").exists():
            out.append(str(child))
    return out


def process(sample_dir: str, output_root: str) -> None:
    paths = SamplePaths(sample_dir)
    out_dir = os.path.join(output_root, paths.name)
    print(f"[sample] {paths.name}")
    render_mod.render_sample(paths, out_dir)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Open-AoE end-to-end visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", help="single sample directory")
    g.add_argument("--data_dir", help="directory containing many sample dirs")
    p.add_argument("--output_dir", default=os.path.join(THIS_DIR, "output"),
                   help="output root (default: ./output)")
    args = p.parse_args()

    samples = [args.sample] if args.sample else find_samples(args.data_dir)
    if not samples:
        print("No samples with annotations found.")
        return 1

    print(f"Processing {len(samples)} sample(s) -> {args.output_dir}")
    for i, s in enumerate(samples, 1):
        print(f"\n=== [{i}/{len(samples)}] ===")
        try:
            process(s, args.output_dir)
        except Exception as exc:  # keep batch alive
            import traceback
            print(f"  [error] {s}: {exc}")
            traceback.print_exc()
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe_retarget_lab.artifacts import cell_artifacts
from aoe_retarget_lab.matrix import build_cells, default_run_dir


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    cells = build_cells(config)
    run_dir = default_run_dir(config)
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": str(args.config),
        "run_dir": str(run_dir),
        "scene": config["scene"],
        "cells": [],
    }

    for cell in cells:
        paths = cell_artifacts(config, cell.key, cell)
        status = {
            "key": cell.key,
            "object_pipeline": cell.object_pipeline,
            "hand_source": cell.hand_source,
            "retargeter": cell.retargeter,
            "artifacts": {k: str(v) if v else None for k, v in paths.items()},
            "exists": {k: bool(v and Path(v).exists()) for k, v in paths.items()},
            "composed_video": None,
        }
        if args.dry_run:
            print(json.dumps(status, indent=2))
        elif args.collect_existing and args.compose:
            required = ["overlay", "depth", "robot"]
            if all(status["exists"].get(k) for k in required):
                out = run_dir / "videos" / f"{cell.key}.mp4"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "compose_triptych.py"),
                    "--overlay", str(paths["overlay"]),
                    "--depth", str(paths["depth"]),
                    "--robot", str(paths["robot"]),
                    "--output", str(out),
                ]
                if args.duration > 0:
                    cmd += ["--duration", str(args.duration)]
                subprocess.run(cmd, check=True)
                status["composed_video"] = str(out)
            else:
                missing = [k for k in required if not status["exists"].get(k)]
                status["missing_reason"] = "missing " + ", ".join(missing)
        report["cells"].append(status)

    report_path = run_dir / "matrix_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "cells": len(cells)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

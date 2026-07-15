#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe_retarget_lab.artifacts import cell_artifacts
from aoe_retarget_lab.matrix import MatrixCell, build_cells


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def package_for_role(role: str, cell: MatrixCell) -> str:
    if role in {"overlay", "depth"}:
        return cell.object_pipeline
    if role == "robot":
        if cell.retargeter == "egoinfinity_g1":
            return "egoinfinity"
        if cell.retargeter == "do_as_i_do_sharpa":
            return "do_as_i_do"
        if cell.retargeter == "spider_mjwp":
            return "spider"
    return "shared"


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"unknown materialize mode: {mode}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_demo_dirs(config: dict, output_root: Path | None, run_name: str | None) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    name = run_name or config["run_name"]
    root = output_root or (repo_root / "experiments" / name)
    return root, root / "intermediates", root / "cells"


def prefer_experiment_artifacts(config: dict, intermediates_dir: Path) -> dict:
    config = dict(config)
    artifacts = dict(config.get("artifacts", {}))
    artifacts.setdefault(
        "do_as_i_do_retargeting_output_root",
        str(intermediates_dir / "do_as_i_do" / "retargeting_outputs"),
    )
    artifacts.setdefault(
        "egoinfinity_run",
        str(intermediates_dir / "egoinfinity" / "runs" / config["run_name"]),
    )
    config["artifacts"] = artifacts
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect per-package demo artifacts and compose vertical overlay/depth/robot videos."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    demo_root, intermediates_dir, cells_dir = build_demo_dirs(config, args.output_root, args.run_name)
    config = prefer_experiment_artifacts(config, intermediates_dir)
    videos_dir = demo_root / "videos"
    cells = build_cells(config)
    report = {
        "config": str(args.config),
        "demo_root": str(demo_root),
        "intermediates_dir": str(intermediates_dir),
        "cells_dir": str(cells_dir),
        "videos_dir": str(videos_dir),
        "mode": args.mode,
        "scene": config["scene"],
        "cells": [],
    }

    for cell in cells:
        artifacts = cell_artifacts(config, cell.key, cell)
        cell_dir = cells_dir / cell.key
        cell_manifest = {
            "key": cell.key,
            "object_pipeline": cell.object_pipeline,
            "hand_source": cell.hand_source,
            "retargeter": cell.retargeter,
            "roles": {},
            "composed_video": None,
            "missing": [],
        }

        role_cell_paths: dict[str, Path] = {}
        for role in ["overlay", "depth", "robot"]:
            src = artifacts.get(role)
            role_record = {
                "source": str(src) if src else None,
                "source_exists": bool(src and Path(src).exists()),
                "package": package_for_role(role, cell),
                "package_path": None,
                "cell_path": None,
            }
            if src and Path(src).exists():
                package = role_record["package"]
                package_dst = intermediates_dir / package / cell.key / f"{role}.mp4"
                cell_dst = cell_dir / f"{role}.mp4"
                if not args.dry_run:
                    materialize(Path(src), package_dst, args.mode)
                    materialize(package_dst, cell_dst, args.mode)
                role_record["package_path"] = str(package_dst)
                role_record["cell_path"] = str(cell_dst)
                role_cell_paths[role] = cell_dst
            else:
                cell_manifest["missing"].append(role)
            cell_manifest["roles"][role] = role_record

        if args.compose and not cell_manifest["missing"]:
            out = videos_dir / f"{cell.key}.mp4"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "compose_triptych.py"),
                "--overlay", str(role_cell_paths["overlay"]),
                "--depth", str(role_cell_paths["depth"]),
                "--robot", str(role_cell_paths["robot"]),
                "--output", str(out),
                "--layout", "vertical",
                "--labels",
                "reconstruction overlay|depth / input geometry|retargeting robot",
            ]
            if args.duration > 0:
                cmd += ["--duration", str(args.duration)]
            if not args.dry_run:
                subprocess.run(cmd, check=True)
            cell_manifest["composed_video"] = str(out)

        if not args.dry_run:
            write_json(cell_dir / "manifest.json", cell_manifest)
        report["cells"].append(cell_manifest)

    composed = sum(1 for cell in report["cells"] if cell.get("composed_video"))
    missing = {
        cell["key"]: cell["missing"]
        for cell in report["cells"]
        if cell.get("missing")
    }

    if not args.dry_run:
        write_json(demo_root / "visual_demo_report.json", report)
    print(
        json.dumps(
            {
                "demo_root": str(demo_root),
                "cells": len(cells),
                "composed": composed,
                "missing_cells": missing,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

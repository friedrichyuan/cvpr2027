#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        if mode == "symlink":
            dst.symlink_to(src, target_is_directory=True)
        else:
            shutil.copytree(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize AoE demo inputs into experiments/<run>/intermediates before package-specific runs."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = args.run_name or config["run_name"]
    demo_root = args.output_root or (REPO_ROOT / "experiments" / run_name)
    inter = demo_root / "intermediates"
    art = config["artifacts"]
    task = art.get("do_as_i_do_task", "foundation_jar_bimanual_leftscale")

    manifest: dict[str, str] = {
        "demo_root": str(demo_root),
        "mode": args.mode,
    }

    clip = Path(art["do_as_i_do_clip"])
    recon_dst = inter / "do_as_i_do" / "reconstruction"
    for name in [
        "raw.mp4",
        "metadata.json",
        "output_tapir_foundation_jar.mp4",
        "moge_depth.mp4",
        "pointmap_depth.mp4",
    ]:
        link_or_copy(clip / name, recon_dst / name, args.mode)
    link_or_copy(clip / "all_frames", recon_dst / "all_frames", args.mode)
    link_or_copy(clip / "obj_tracking_out", recon_dst / "obj_tracking_out", args.mode)

    retarget_root = inter / "do_as_i_do" / "retargeting_outputs"
    link_or_copy(
        Path(art["do_as_i_do_object_mesh"]),
        retarget_root / "assets" / "objects" / task / "visual.obj",
        args.mode,
    )
    link_or_copy(
        Path(art["do_as_i_do_spider_task"]),
        retarget_root / "mano" / "bimanual" / task / "task_info.json",
        args.mode,
    )
    stage1_root = Path(art["do_as_i_do_spider_task"]).parent
    link_or_copy(
        stage1_root / "0" / "trajectory_keypoints.npz",
        retarget_root / "mano" / "bimanual" / task / "0" / "trajectory_keypoints.npz",
        args.mode,
    )

    egoinfinity_dst = inter / "egoinfinity"
    link_or_copy(Path(art["egoinfinity_clip"]), egoinfinity_dst / "clip", args.mode)

    manifest.update(
        {
            "do_as_i_do_reconstruction": str(recon_dst),
            "do_as_i_do_retargeting_output_root": str(retarget_root),
            "egoinfinity_intermediate": str(egoinfinity_dst),
        }
    )
    manifest_path = demo_root / "materialized_inputs.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

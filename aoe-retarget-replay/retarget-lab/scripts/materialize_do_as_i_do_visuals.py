#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def object_id_from_task(task: str) -> str:
    for marker in ["_bimanual", "_right", "_left"]:
        if marker in task:
            return task.split(marker, 1)[0]
    return task


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def ffmpeg_writer(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_depth(pointmaps: list[Path], output: Path, fps: float) -> None:
    if not pointmaps:
        raise FileNotFoundError("no *_pointmap.npy files found")
    sample = np.load(pointmaps[0])
    if sample.ndim != 3 or sample.shape[-1] < 3:
        raise ValueError(f"expected HxWx3 pointmap, got {sample.shape}")
    height, width = sample.shape[:2]
    proc = ffmpeg_writer(output, width, height, fps)
    assert proc.stdin is not None
    try:
        for path in pointmaps:
            arr = np.load(path)
            z = arr[..., 2].astype(np.float32)
            finite = np.isfinite(z)
            if finite.any():
                lo, hi = np.nanpercentile(z[finite], [2.0, 98.0])
                if hi <= lo:
                    hi = lo + 1e-6
                img = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
            else:
                img = np.zeros_like(z, dtype=np.float32)
            gray = (img * 255.0).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=-1)
            proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Do-as-I-Do overlay/depth videos under experiments/.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    clip_dir = args.clip_dir.expanduser().resolve()
    if not clip_dir.is_dir():
        raise FileNotFoundError(f"clip dir does not exist: {clip_dir}")

    object_id = object_id_from_task(args.task)
    exp = REPO_ROOT / "experiments" / args.run_name
    out_dir = exp / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction"
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_candidates = [
        clip_dir / f"output_tapir_{object_id}_overlay.mp4",
        clip_dir / f"output_tapir_{object_id}.mp4",
        clip_dir / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
        clip_dir / "obj_tracking_out" / object_id / "combined_visualization" / "projected" / "video.mp4",
        clip_dir / "video_segmentation" / f"tracked_{object_id}_bootstrap.mp4",
        clip_dir / "raw.mp4",
    ]
    overlay = next((p for p in overlay_candidates if p.exists()), None)
    if overlay is not None:
        link_or_copy(overlay, out_dir / "overlay.mp4", args.copy)

    mask_candidates = sorted((clip_dir / "video_segmentation").glob("tracked_*.mp4"))
    if mask_candidates:
        link_or_copy(mask_candidates[0], out_dir / "mask_overlay.mp4", args.copy)

    frame_dir = clip_dir / "all_frames"
    pointmaps = sorted(frame_dir.glob("*_pointmap.npy"))
    if args.max_frames > 0:
        pointmaps = pointmaps[: args.max_frames]
    if pointmaps:
        render_depth(pointmaps, out_dir / "depth.mp4", args.fps)

    indexed_raw_dir = out_dir / "raw_dir"
    raw_dir = indexed_raw_dir if indexed_raw_dir.exists() else clip_dir
    if raw_dir.exists():
        mesh_overlay_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_do_as_i_do_mesh_overlay.py"),
            "--clip-dir",
            str(clip_dir),
            "--raw-dir",
            str(raw_dir),
            "--task",
            args.task,
            "--output",
            str(out_dir / "mesh_overlay.mp4"),
            "--fps",
            str(args.fps),
        ]
        if args.max_frames > 0:
            mesh_overlay_cmd += ["--max-frames", str(args.max_frames)]
        subprocess.run(mesh_overlay_cmd, check=True)

        mesh_pure_cmd = list(mesh_overlay_cmd)
        output_idx = mesh_pure_cmd.index("--output") + 1
        mesh_pure_cmd[output_idx] = str(out_dir / "mesh_pure_camera.mp4")
        mesh_pure_cmd += ["--background", "black"]
        subprocess.run(mesh_pure_cmd, check=True)

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

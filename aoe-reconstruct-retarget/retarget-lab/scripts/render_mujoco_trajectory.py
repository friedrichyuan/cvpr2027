#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np


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


def load_qpos(path: Path, nq: int) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "qpos" not in data:
        raise KeyError(f"{path} does not contain qpos")
    qpos = np.asarray(data["qpos"], dtype=np.float64).reshape(-1, data["qpos"].shape[-1])
    if qpos.shape[1] < nq:
        raise ValueError(f"qpos dim {qpos.shape[1]} is smaller than model.nq {nq}")
    if qpos.shape[1] > nq:
        qpos = qpos[:, :nq]
    return qpos


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a MuJoCo qpos trajectory to mp4.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--camera", default=None)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    qpos = load_qpos(args.trajectory, model.nq)

    camera = args.camera
    if camera is None and model.ncam > 0:
        camera = 0

    proc = ffmpeg_writer(args.output, args.width, args.height, args.fps)
    assert proc.stdin is not None
    try:
        for frame_qpos in qpos[:: max(args.stride, 1)]:
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            if camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
        renderer.close()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

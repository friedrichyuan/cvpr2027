"""Replay saved MjLab rollout qpos in the original ARX table GLFW scene."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

from .defaults import DEFAULT_SCENE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    rollout = np.load(args.rollout)
    qpos = rollout["qpos"]
    fps = float(rollout.get("fps", 50.0))
    model = mujoco.MjModel.from_xml_path(str(args.scene.expanduser().resolve()))
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Expected rollout qpos (T, {model.nq}), got {qpos.shape}")
    data = mujoco.MjData(model)
    frame = 0
    next_time = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            frame += 1
            if frame == len(qpos):
                if not args.loop:
                    break
                frame = 0
            next_time += 1.0 / fps
            time.sleep(max(0.0, next_time - time.monotonic()))


if __name__ == "__main__":
    main()

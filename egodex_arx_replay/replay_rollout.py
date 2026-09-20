"""Replay saved MjLab rollout qpos in the original ARX table GLFW scene."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
from xml.etree import ElementTree

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

from .defaults import DEFAULT_SCENE


def _comparison_model(scene_path: Path, reference_y: float) -> tuple[mujoco.MjModel, np.ndarray]:
    """Build a scene with a passive, colored duplicate ARX for IK comparison."""
    root = ElementTree.parse(scene_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"Scene has no compiler element: {scene_path}")
    compiler.set("meshdir", str((scene_path.parent / "meshes").resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Scene has no worldbody: {scene_path}")
    source_body = next(
        (body for body in worldbody.findall("body") if body.get("name") == "base_link"),
        None,
    )
    if source_body is None:
        raise ValueError("Scene has no ARX base_link body")

    duplicate = copy.deepcopy(source_body)
    duplicate.set("pos", f"0 {reference_y:.4f} 0")
    _prefix_element_names(duplicate, "ik_")
    for geom in duplicate.iter("geom"):
        # Blue identifies the unoptimized IK reference robot.
        geom.set("rgba", "0.12 0.45 0.95 1")
    worldbody.append(duplicate)

    model = mujoco.MjModel.from_xml_string(ElementTree.tostring(root, encoding="unicode"))
    qpos_indices = np.array(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ik_{name}")
            ]
            for name in _ARX_JOINT_NAMES
        ],
        dtype=int,
    )
    return model, qpos_indices


def _prefix_element_names(element: ElementTree.Element, prefix: str) -> None:
    """Prefix body/joint/geom/site names in a duplicated kinematic subtree."""
    for node in element.iter():
        if "name" in node.attrib:
            node.set("name", prefix + node.attrib["name"])


_ARX_JOINT_NAMES = (
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "left_joint8",
    "right_joint11",
    "right_joint12",
    "right_joint13",
    "right_joint14",
    "right_joint15",
    "right_joint16",
    "right_joint17",
    "right_joint18",
)


class RolloutController:
    """Responsive, pauseable frame clock that stops at the final rollout frame."""

    def __init__(self, frame_count: int, fps: float, loop: bool) -> None:
        self.frame_count = frame_count
        self.fps = fps
        self.loop = loop
        self.frame = 0
        self.playing = True
        self.speed = 1.0
        self._credit = 0.0
        self._last_time = time.monotonic()

    def on_key(self, keycode: int) -> None:
        if keycode == 32:  # Space
            self.playing = not self.playing
        elif keycode in (74, 263):  # J or left arrow
            self.playing = False
            self.frame = max(0, self.frame - 1)
        elif keycode in (76, 262):  # L or right arrow
            self.playing = False
            self.frame = min(self.frame_count - 1, self.frame + 1)
        elif keycode in (82, 114):  # R or r
            self.frame = 0
            self._credit = 0.0
        elif keycode in (45, 95):  # - or _
            self.speed = max(0.125, self.speed / 2.0)
        elif keycode in (61, 43):  # = or +
            self.speed = min(8.0, self.speed * 2.0)

    def advance(self) -> int:
        now = time.monotonic()
        if not self.playing:
            self._last_time = now
            return self.frame
        self._credit += (now - self._last_time) * self.fps * self.speed
        self._last_time = now
        step_count = int(self._credit)
        self._credit -= step_count
        if step_count == 0:
            return self.frame
        next_frame = self.frame + step_count
        if next_frame < self.frame_count:
            self.frame = next_frame
        elif self.loop:
            self.frame = next_frame % self.frame_count
        else:
            self.frame = self.frame_count - 1
            self.playing = False
            self._credit = 0.0
        return self.frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Exported reference NPZ; when set, show an adjacent blue IK robot",
    )
    parser.add_argument(
        "--reference-y",
        type=float,
        default=-0.80,
        help="Lateral offset of the blue IK comparison robot in metres",
    )
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    with np.load(args.rollout) as rollout:
        qpos = np.asarray(rollout["qpos"], dtype=np.float64)
        fps = float(rollout["fps"]) if "fps" in rollout else 50.0
    scene_path = args.scene.expanduser().resolve()
    primary_model = mujoco.MjModel.from_xml_path(str(scene_path))
    primary_nq = primary_model.nq
    if args.reference is None:
        model = primary_model
        ik_qpos_indices = None
        ik_qpos = None
        ik_fps = None
    else:
        with np.load(args.reference) as reference:
            ik_qpos = np.asarray(reference["qpos_ref"], dtype=np.float64)
            metadata = reference["metadata"].item()
            import json

            ik_fps = float(json.loads(str(metadata))["fps"])
        model, ik_qpos_indices = _comparison_model(scene_path, args.reference_y)
    # With an IK comparison robot, ``model.nq`` contains both ARX instances.
    # The rollout always contains only the primary robot's qpos sequence.
    if qpos.ndim != 2 or qpos.shape[1] != primary_nq:
        raise ValueError(f"Expected rollout qpos (T, {primary_nq}), got {qpos.shape}")
    data = mujoco.MjData(model)
    controller = RolloutController(len(qpos), fps, args.loop)
    print("Controls: space play/pause | J/L or arrows step | R restart | -/+ speed")
    if ik_qpos is not None:
        print("Gray robot: policy rollout | blue robot: unoptimized IK reference")
    with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
        while viewer.is_running():
            frame = controller.advance()
            data.qpos[: len(qpos[frame])] = qpos[frame]
            if ik_qpos is not None and ik_qpos_indices is not None and ik_fps is not None:
                ik_frame = min(int(frame * ik_fps / fps), len(ik_qpos) - 1)
                data.qpos[ik_qpos_indices] = ik_qpos[ik_frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            # Keep GLFW responsive while paused and avoid a long blocking sleep.
            time.sleep(0.005)


if __name__ == "__main__":
    main()

"""Render the ARX arm and composite it. The arm is pasted; the gripper uses depth."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egowhale.media import load_masks, read_rgb, write_rgb
from egowhale.step import BASE, COMPOSITE, DEPTH, GRIPPER, IK, INPAINT, MASKS, ROOT, Step

_SCENE = ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"
_HIDE = ("floor", "table", "camera", "workspace", "front_workspace", "base_plus", "base_minus", "robot_front", "humanego")
_GRIPPER_GEOMS = ("left_link7", "left_link8", "right_link17", "right_link18")
_HAND_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class Composite(Step):
    name = "composite"
    needs = (INPAINT, IK, BASE, GRIPPER, DEPTH, MASKS)
    makes = (COMPOSITE,)

    def run(self, src: Path, dst: Path) -> None:
        dst = Path(dst)
        background, fps = read_rgb(dst / INPAINT)
        scale = background.shape[1] / _video_height(Path(src).with_suffix(".mp4"))
        with np.load(dst / IK) as data:
            qpos = np.asarray(data["qpos"])
        with np.load(dst / GRIPPER) as data:
            intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
        intrinsic = intrinsic.copy()
        intrinsic[:2] *= scale
        base = np.asarray(json.loads((dst / BASE).read_text())["matrix"], dtype=np.float64)
        frames = min(len(background), len(qpos))
        background, qpos = background[:frames], qpos[:frames]
        height, width = background.shape[1], background.shape[2]
        robot, robot_mask, gripper_mask, robot_depth = _render(qpos, base, intrinsic, height, width)
        scene_depth = _match_depth(np.load(dst / DEPTH)["depth"], height, width)
        hand = _match_mask(load_masks(dst / MASKS), height, width)
        image = _composite(background, robot, robot_mask, gripper_mask, robot_depth, scene_depth, hand)
        write_rgb(dst / COMPOSITE, image, fps)


def _video_height(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    capture.release()
    if height <= 0:
        raise FileNotFoundError(path)
    return height


def _render(qpos, base, intrinsic, height, width):
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    _hide(model)
    _place_base(model, base)
    _place_camera(model, intrinsic, height)
    gripper_ids = np.array([
        geom_id
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(_GRIPPER_GEOMS)
    ])
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    count = len(qpos)
    rgb = np.zeros((count, height, width, 3), dtype=np.uint8)
    robot_mask = np.zeros((count, height, width), dtype=bool)
    gripper_mask = np.zeros((count, height, width), dtype=bool)
    depth = np.zeros((count, height, width), dtype=np.float32)
    for index in range(count):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="ego_camera_calibrated")
        rgb[index] = renderer.render()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="ego_camera_calibrated")
        geom = renderer.render()[:, :, 0]
        renderer.disable_segmentation_rendering()
        robot_mask[index] = geom >= 0
        gripper_mask[index] = np.isin(geom, gripper_ids)
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="ego_camera_calibrated")
        depth[index] = renderer.render()
        renderer.disable_depth_rendering()
    renderer.close()
    return rgb, robot_mask, gripper_mask, depth


def _composite(background, robot, robot_mask, gripper_mask, robot_depth, scene_depth, hand):
    count = min(len(background), len(robot), len(scene_depth), len(hand))
    image = background[:count].copy()
    for index in range(count):
        dilated = cv2.dilate(hand[index].astype(np.uint8), _HAND_KERNEL, iterations=1).astype(bool)
        arm = robot_mask[index] & ~gripper_mask[index]
        hidden = gripper_mask[index] & (scene_depth[index] < robot_depth[index]) & ~dilated
        visible = arm | (gripper_mask[index] & ~hidden)
        image[index][visible] = robot[index][visible]
    return image


def _match_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.shape[1:] == (height, width):
        return depth
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth])


def _match_mask(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    if masks.shape[1:] == (height, width):
        return masks
    return np.stack([
        cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        for mask in masks
    ])


def _hide(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(_HIDE):
            model.geom_rgba[geom_id, 3] = 0.0
    model.site_rgba[:, 3] = 0.0


def _place_base(model: mujoco.MjModel, transform: np.ndarray) -> None:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    model.body_pos[body] = transform[:3, 3]
    model.body_quat[body] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]


def _place_camera(model: mujoco.MjModel, intrinsic: np.ndarray, height: int) -> None:
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego_camera_calibrated")
    model.cam_pos[cam] = 0.0
    model.cam_quat[cam] = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0])).as_quat()[[3, 0, 1, 2]]
    model.cam_fovy[cam] = float(np.degrees(2.0 * np.arctan(height / (2.0 * float(intrinsic[1, 1])))))

from __future__ import annotations

import json
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..processor import CPU, Artifacts, EpisodeContext, ResourceSpec
from ..robots.arx import apply_base_pose, load_arx_model
from ..video import read_rgb_video, save_masks, write_rgb_video

_HIDE_PREFIXES = (
    "floor",
    "table",
    "camera",
    "workspace",
    "front_workspace",
    "base_plus",
    "base_minus",
    "robot_front",
    "humanego",
)


class RenderProcessor:
    name = "render"
    requires = frozenset({Artifacts.SOURCE, Artifacts.IK, Artifacts.BASE})
    produces = frozenset({Artifacts.ROBOT_RGB, Artifacts.ROBOT_MASK})
    optional_requires: frozenset[str] = frozenset()
    resources: ResourceSpec = CPU

    def run(self, ctx: EpisodeContext) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        source = json.loads((ctx.out_dir / Artifacts.SOURCE).read_text(encoding="utf-8"))
        base = json.loads((ctx.out_dir / Artifacts.BASE).read_text(encoding="utf-8"))
        with np.load(ctx.out_dir / Artifacts.IK) as data:
            qpos = np.asarray(data["qpos"])
        height, width = _frame_size(ctx, source)
        model = load_arx_model()
        _hide_scene(model)
        apply_base_pose(model, _base_transform(base))
        _set_ego_camera(model, np.asarray(source["intrinsic"]), height)
        rgb, masks = _render_episode(model, qpos, height, width)
        fps = float(source.get("fps", 30.0))
        write_rgb_video(ctx.out_dir / Artifacts.ROBOT_RGB, rgb, fps)
        save_masks(ctx.out_dir / Artifacts.ROBOT_MASK, masks)


def _frame_size(ctx: EpisodeContext, source: dict) -> tuple[int, int]:
    video = ctx.video_path
    if video is not None and video.is_file():
        frames, _ = read_rgb_video(video)
        return int(frames.shape[1]), int(frames.shape[2])
    return 720, 1280


def _base_transform(base: dict) -> np.ndarray:
    quat = np.asarray(base["quat_wxyz"], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = np.asarray(base["translation"], dtype=np.float64)
    return transform


def _hide_scene(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if any(name.startswith(prefix) for prefix in _HIDE_PREFIXES):
            model.geom_rgba[geom_id, 3] = 0.0
    model.site_rgba[:, 3] = 0.0


def _set_ego_camera(model: mujoco.MjModel, intrinsic: np.ndarray, height: int) -> None:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego_camera_calibrated")
    if cam_id < 0:
        raise ValueError("Scene is missing camera 'ego_camera_calibrated'")
    model.cam_pos[cam_id] = 0.0
    # OpenCV camera (X right, Y down, Z forward) → MuJoCo camera (looks along -Z, Y up).
    model.cam_quat[cam_id] = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0])).as_quat()[[3, 0, 1, 2]]
    fy = float(intrinsic[1, 1])
    model.cam_fovy[cam_id] = float(np.degrees(2.0 * np.arctan(height / (2.0 * fy))))


def _render_episode(
    model: mujoco.MjModel, qpos: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    rgb = np.zeros((qpos.shape[0], height, width, 3), dtype=np.uint8)
    masks = np.zeros((qpos.shape[0], height, width), dtype=bool)
    for index, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="ego_camera_calibrated")
        rgb[index] = renderer.render()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="ego_camera_calibrated")
        geom_ids = renderer.render()[:, :, 0]
        masks[index] = geom_ids >= 0
        renderer.disable_segmentation_rendering()
    renderer.close()
    return rgb, masks

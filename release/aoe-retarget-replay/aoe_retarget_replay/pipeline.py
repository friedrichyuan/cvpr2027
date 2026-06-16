"""AoE-Retarget-Replay pipeline: AoE data → robot action sequences.

Orchestrates: load → shoulder synthesis → coordinate transform →
arm IK → hand retarget → action assembly → LeRobot parquet + MuJoCo viz.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import mujoco
import numpy as np

from aoe_retarget_replay.aoe.episode import (
    AoEEpisode,
    load_aoe_episode,
)
from aoe_retarget_replay.aoe.shoulder import (
    SHOULDER_HALF_WIDTH,
    SHOULDER_OFFSET_FROM_CAMERA_CAM,
    refine_shoulder_offset_to_arm_length,
    synth_shoulders_cam,
)
from aoe_retarget_replay.aoe.transform import transform_aoe_wrist_poses
from aoe_retarget_replay.constants import G1_STANDING_HEIGHT
from aoe_retarget_replay.constants.aoe import DEX3_TIPS, INSPIRE_TIPS
from aoe_retarget_replay.retarget.arm_ik import G1ArmIKSolver
from aoe_retarget_replay.robots import RobotSpec, get_spec

logger = logging.getLogger(__name__)


def _gap_fill_confidence(
    data: np.ndarray,
    conf: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Forward-fill low-confidence frames with last good value."""
    out = data.copy()
    last_good = data[0].copy()
    for t in range(data.shape[0]):
        if conf[t] >= threshold:
            last_good = data[t].copy()
            out[t] = data[t]
        else:
            out[t] = last_good
    return out


def _fingertips_for_spec(spec: RobotSpec) -> tuple[int, ...]:
    hand_cls = spec.hand_retargeter_cls
    if hand_cls is None:
        raise ValueError(f"Spec {spec.name!r} has no hand_retargeter_cls")
    n_tips = hand_cls.N_TIPS
    if n_tips == 3:
        return DEX3_TIPS
    if n_tips == 5:
        return INSPIRE_TIPS
    raise ValueError(f"Unsupported N_TIPS={n_tips} for spec {spec.name!r}")


def _estimate_g1_shoulder_positions(model, data) -> dict:
    """Get G1 shoulder positions from MuJoCo model in standing pose."""
    lid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link"
    )
    rid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link"
    )
    return {
        "left_shoulder": data.xpos[lid].copy(),
        "right_shoulder": data.xpos[rid].copy(),
    }


@dataclasses.dataclass
class RetargetResult:
    """Output of a single episode retarget."""
    actions: np.ndarray            # (T, action_dim)
    scale: float
    spec_name: str
    meta: AoEEpisode
    task_description: str


class RetargetSession:
    """Reusable session carrying IK solver + hand retargeters for one spec."""

    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self.ik = G1ArmIKSolver(mjcf_path=spec.mjcf_path)
        cls = spec.hand_retargeter_cls
        self.left_hand = cls(side="left")
        self.right_hand = cls(side="right")

    def retarget(
        self,
        episode_dir: Path | str,
        refine_shoulder: bool = True,
        frame: str = "cam",
    ) -> RetargetResult:
        """Retarget one AoE episode to robot action sequence."""
        spec = self.spec
        tips_idx = _fingertips_for_spec(spec)

        load_dict, meta = load_aoe_episode(
            episode_dir,
            fingertips=tips_idx,
            frame=frame,
        )

        T = load_dict["n_frames"]
        n_tips = spec.hand_retargeter_cls.N_TIPS

        if refine_shoulder:
            lw = load_dict["left_wrist_poses"][:, :3, 3]
            rw = load_dict["right_wrist_poses"][:, :3, 3]
            l_valid = load_dict["left_wrist_conf"] > 0.5
            r_valid = load_dict["right_wrist_conf"] > 0.5
            new_offset, new_hw = refine_shoulder_offset_to_arm_length(
                lw, rw,
                init_offset=SHOULDER_OFFSET_FROM_CAMERA_CAM,
                init_half_width=SHOULDER_HALF_WIDTH,
                valid_left=l_valid, valid_right=r_valid,
            )
            left_sh, right_sh = synth_shoulders_cam(
                T, offset=new_offset, half_width=new_hw
            )
            load_dict["left_shoulder_pos"] = left_sh
            load_dict["right_shoulder_pos"] = right_sh
            logger.info(
                "Shoulder refined: offset=%s, half_width=%.3f",
                np.round(new_offset, 3).tolist(), new_hw,
            )

        left_wrist_poses = _gap_fill_confidence(
            load_dict["left_wrist_poses"].reshape(T, -1),
            load_dict["left_wrist_conf"]
        ).reshape(T, 4, 4)
        right_wrist_poses = _gap_fill_confidence(
            load_dict["right_wrist_poses"].reshape(T, -1),
            load_dict["right_wrist_conf"]
        ).reshape(T, 4, 4)
        left_fingertips = _gap_fill_confidence(
            load_dict["left_fingertips"].reshape(T, -1),
            load_dict["left_wrist_conf"]
        ).reshape(T, n_tips, 3)
        right_fingertips = _gap_fill_confidence(
            load_dict["right_fingertips"].reshape(T, -1),
            load_dict["right_wrist_conf"]
        ).reshape(T, n_tips, 3)

        model = self.ik.model
        data = self.ik.data
        data.qpos[:] = 0.0
        data.qpos[2] = G1_STANDING_HEIGHT
        mujoco.mj_forward(model, data)
        g1_shoulders = _estimate_g1_shoulder_positions(model, data)

        left_wrist_g1, right_wrist_g1, scale = transform_aoe_wrist_poses(
            left_wrist_poses, right_wrist_poses,
            load_dict["left_shoulder_pos"], load_dict["right_shoulder_pos"],
            g1_shoulders["left_shoulder"], g1_shoulders["right_shoulder"],
        )
        logger.info("Episode T=%d, scale=%.3f", T, scale)

        arm_qpos = self.ik.solve_episode(
            left_wrist_g1, right_wrist_g1,
            load_dict["left_wrist_conf"], load_dict["right_wrist_conf"],
        )

        left_wrist_pos = left_wrist_poses[:, :3, 3]
        right_wrist_pos = right_wrist_poses[:, :3, 3]
        left_hand_qpos = self.left_hand.retarget_episode(
            left_fingertips, left_wrist_pos, load_dict["left_tips_conf"],
        )
        right_hand_qpos = self.right_hand.retarget_episode(
            right_fingertips, right_wrist_pos, load_dict["right_tips_conf"],
        )

        left_hand_qpos = np.clip(
            left_hand_qpos,
            model.jnt_range[spec.left_hand_clamp_joint_ids, 0],
            model.jnt_range[spec.left_hand_clamp_joint_ids, 1],
        )
        right_hand_qpos = np.clip(
            right_hand_qpos,
            model.jnt_range[spec.right_hand_clamp_joint_ids, 0],
            model.jnt_range[spec.right_hand_clamp_joint_ids, 1],
        )

        actions = np.concatenate(
            [arm_qpos, left_hand_qpos, right_hand_qpos], axis=-1,
        ).astype(np.float32)
        assert actions.shape == (T, spec.action_dim), (actions.shape, spec.action_dim)

        return RetargetResult(
            actions=actions,
            scale=scale,
            spec_name=spec.name,
            meta=meta,
            task_description=load_dict.get("description", ""),
        )


def discover_episodes(data_root: Path) -> list[Path]:
    """Discover all episode directories under a poc_deliver-style root.

    An episode directory is one that contains ego_process/ or
    ego_hands_reconstruction/ with a hands.npz file.
    """
    episodes = []
    data_root = Path(data_root)

    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        for sub in ("ego_process/ego_hands_reconstruction",
                     "ego_hands_reconstruction"):
            hands_path = entry / sub / "hands.npz"
            if hands_path.exists():
                episodes.append(entry)
                break

    logger.info("Discovered %d episodes under %s", len(episodes), data_root)
    return episodes

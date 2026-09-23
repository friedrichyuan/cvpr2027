"""Manager-based ARX zero-to-EgoDex-initial-frame transition environment."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as builtin_mdp
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from .actions import ArmTransitionActionCfg
from .initial_reference_command import InitialReferenceCommandCfg
from .mdp import (
    action_l2,
    final_joint_tracking_exp,
    final_tcp_orientation_tracking_exp,
    final_tcp_position_tracking_exp,
    target_pose_command,
    tcp_orientation_error,
    tcp_position_error,
    transition_success_reward,
)
from .robot import get_arx5_robot_cfg


def make_transition_env_cfg(
    reference_dir: str,
    num_envs: int = 128,
    episode_length_s: float = 1.0,
) -> ManagerBasedRlEnvCfg:
    tcp_cfg = SceneEntityCfg("robot", site_names=("left_tcp", "right_tcp"))
    arm_joint_cfg = SceneEntityCfg(
        "robot",
        joint_names=(
            "left_joint[1-6]",
            "right_joint1[1-6]",
        ),
    )
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=builtin_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=builtin_mdp.joint_vel_rel),
        "pose_command": ObservationTermCfg(
            func=target_pose_command,
            params={"command_name": "reference"},
        ),
        "actions": ObservationTermCfg(func=builtin_mdp.last_action),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_arx5_robot_cfg()},
            num_envs=num_envs,
            env_spacing=1.5,
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, concatenate_terms=True),
            "critic": ObservationGroupCfg(actor_terms, concatenate_terms=True),
        },
        actions={
            "arm_transition": ArmTransitionActionCfg(
                entity_name="robot",
                actuator_names=(
                    "left_joint[1-6]",
                    "right_joint1[1-6]",
                ),
                scale=0.05,
                clip={".*": (-0.05, 0.05)},
                end_velocity_scale=0.2,
                max_end_velocity=0.5,
            )
        },
        commands={
            "reference": InitialReferenceCommandCfg(
                reference_dir=reference_dir,
                resampling_time_range=(1.0e9, 1.0e9),
            )
        },
        events={},
        rewards={
            "end_effector_position_tracking": RewardTermCfg(
                func=tcp_position_error,
                weight=-0.2,
                params={"command_name": "reference", "asset_cfg": tcp_cfg},
            ),
            "end_effector_orientation_tracking": RewardTermCfg(
                func=tcp_orientation_error,
                weight=-0.1,
                params={"command_name": "reference", "asset_cfg": tcp_cfg},
            ),
            "success": RewardTermCfg(
                func=transition_success_reward,
                weight=10.0,
                params={
                    "command_name": "reference",
                    "asset_cfg": tcp_cfg,
                    "position_threshold": 0.05,
                    "orientation_threshold": 0.2,
                },
            ),
            "final_joint": RewardTermCfg(
                func=final_joint_tracking_exp,
                weight=2.0,
                params={"command_name": "reference", "std": 0.10},
            ),
            "final_tcp_position": RewardTermCfg(
                func=final_tcp_position_tracking_exp,
                weight=5.0,
                params={"command_name": "reference", "asset_cfg": tcp_cfg, "std": 0.025},
            ),
            "final_tcp_orientation": RewardTermCfg(
                func=final_tcp_orientation_tracking_exp,
                weight=1.0,
                params={"command_name": "reference", "asset_cfg": tcp_cfg, "std": 0.20},
            ),
            "action_rate": RewardTermCfg(func=builtin_mdp.action_rate_l2, weight=-0.001),
            "action_magnitude": RewardTermCfg(func=action_l2, weight=-0.01),
            "joint_vel": RewardTermCfg(
                func=builtin_mdp.joint_vel_l2,
                weight=-0.001,
                params={"asset_cfg": arm_joint_cfg},
            ),
        },
        terminations={"time_out": TerminationTermCfg(func=builtin_mdp.time_out, time_out=True)},
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=episode_length_s,
        is_finite_horizon=True,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.5,
            elevation=-20.0,
            azimuth=120.0,
        ),
    )

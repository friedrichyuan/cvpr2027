"""Manager-based fixed-base ARX reference-tracking environment."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as builtin_mdp
from mjlab.managers import (
    EventTermCfg,
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

from .actions import ReferenceResidualActionCfg
from .mdp import (
    joint_tracking_exp,
    reference_command,
    reset_to_reference,
    tcp_orientation_tracking_exp,
    tcp_position_tracking_exp,
    tcp_target_error,
)
from .reference_bank import ReferenceBankCommandCfg
from .robot import get_arx5_robot_cfg


def make_env_cfg(reference_dir: str, num_envs: int = 128) -> ManagerBasedRlEnvCfg:
    tcp_cfg = SceneEntityCfg("robot", site_names=("left_tcp", "right_tcp"))
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=builtin_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=builtin_mdp.joint_vel_rel),
        "reference": ObservationTermCfg(
            func=reference_command, params={"command_name": "reference"}
        ),
        "tcp_target_error": ObservationTermCfg(
            func=tcp_target_error,
            params={"command_name": "reference", "asset_cfg": tcp_cfg},
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
            "residual_joint_pos": ReferenceResidualActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.03,
                clip={".*": (-0.05, 0.05)},
            )
        },
        commands={
            "reference": ReferenceBankCommandCfg(
                reference_dir=reference_dir,
                resampling_time_range=(1.0e9, 1.0e9),
            )
        },
        events={
            "reset_reference": EventTermCfg(
                func=reset_to_reference,
                mode="reset",
                params={"command_name": "reference"},
            )
        },
        rewards={
            "joint_tracking": RewardTermCfg(
                func=joint_tracking_exp,
                weight=0.50,
                params={"command_name": "reference", "std": 0.25},
            ),
            "tcp_position": RewardTermCfg(
                func=tcp_position_tracking_exp,
                weight=0.40,
                params={"command_name": "reference", "asset_cfg": tcp_cfg, "std": 0.04},
            ),
            "tcp_orientation": RewardTermCfg(
                func=tcp_orientation_tracking_exp,
                weight=0.10,
                params={"command_name": "reference", "asset_cfg": tcp_cfg, "std": 0.35},
            ),
            "action_rate": RewardTermCfg(func=builtin_mdp.action_rate_l2, weight=-0.05),
            "joint_limit": RewardTermCfg(
                func=builtin_mdp.joint_pos_limits,
                weight=-2.0,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
        },
        terminations={"time_out": TerminationTermCfg(func=builtin_mdp.time_out, time_out=True)},
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=1.0,
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

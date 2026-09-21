"""Initial-frame reference command for ARX zero-to-EgoDex transition tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class InitialReferenceCommandCfg(CommandTermCfg):
    reference_dir: str
    fixed_trajectory_id: int | None = None

    def build(self, env) -> "InitialReferenceCommand":
        return InitialReferenceCommand(self, env)


class InitialReferenceCommand(CommandTerm):
    """Samples one exported trajectory and exposes only its frame-zero target."""

    def __init__(self, cfg: InitialReferenceCommandCfg, env) -> None:
        super().__init__(cfg, env)
        files = sorted(Path(cfg.reference_dir).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No exported references in {cfg.reference_dir}")
        self.files = files

        refs = [np.load(path, allow_pickle=False) for path in files]
        nq = refs[0]["qpos_ref"].shape[1]
        self.qpos_bank = torch.empty(len(refs), nq, device=self.device)
        self.qvel_bank = torch.empty_like(self.qpos_bank)
        self.tcp_pos_bank = torch.empty(len(refs), 2, 3, device=self.device)
        self.tcp_quat_bank = torch.empty(len(refs), 2, 4, device=self.device)
        self.width_bank = torch.empty(len(refs), 2, device=self.device)
        for index, ref in enumerate(refs):
            self.qpos_bank[index] = torch.as_tensor(
                ref["qpos_ref"][0], dtype=torch.float32, device=self.device
            )
            self.qvel_bank[index] = torch.as_tensor(
                ref["qvel_ref"][0], dtype=torch.float32, device=self.device
            )
            self.tcp_pos_bank[index] = torch.as_tensor(
                ref["tcp_pos_ref"][0], dtype=torch.float32, device=self.device
            )
            self.tcp_quat_bank[index] = torch.as_tensor(
                ref["tcp_quat_wxyz_ref"][0], dtype=torch.float32, device=self.device
            )
            self.width_bank[index] = torch.as_tensor(
                ref["gripper_width_ref"][0], dtype=torch.float32, device=self.device
            )
            ref.close()

        self.trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["reference_id"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def qpos(self) -> torch.Tensor:
        return self.qpos_bank[self.trajectory_ids]

    @property
    def qvel(self) -> torch.Tensor:
        return self.qvel_bank[self.trajectory_ids]

    @property
    def tcp_pos(self) -> torch.Tensor:
        return self.tcp_pos_bank[self.trajectory_ids]

    @property
    def tcp_quat(self) -> torch.Tensor:
        return self.tcp_quat_bank[self.trajectory_ids]

    @property
    def width(self) -> torch.Tensor:
        return self.width_bank[self.trajectory_ids]

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            (
                self.qpos,
                self.qvel,
                self.tcp_pos.reshape(self.num_envs, -1),
                self.tcp_quat.reshape(self.num_envs, -1),
                self.width,
            ),
            dim=-1,
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """Sample a first-frame target, then start the robot from zero position."""
        assert isinstance(env_ids, torch.Tensor)
        extras = super().reset(env_ids)
        robot = self._env.scene["robot"]
        zero_qpos = torch.zeros_like(self.qpos[env_ids])
        zero_qvel = torch.zeros_like(self.qvel[env_ids])
        robot.write_joint_state_to_sim(zero_qpos, zero_qvel, env_ids=env_ids)
        robot.reset(env_ids=env_ids)
        return extras

    def _update_metrics(self) -> None:
        self.metrics["reference_id"] = self.trajectory_ids.float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.fixed_trajectory_id is None:
            self.trajectory_ids[env_ids] = torch.randint(
                len(self.files), (len(env_ids),), device=self.device
            )
            return
        if not 0 <= self.cfg.fixed_trajectory_id < len(self.files):
            raise ValueError(
                f"fixed_trajectory_id={self.cfg.fixed_trajectory_id} is outside "
                f"[0, {len(self.files)})"
            )
        self.trajectory_ids[env_ids] = self.cfg.fixed_trajectory_id

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        # Static command: the target is always frame zero of the sampled reference.
        return None

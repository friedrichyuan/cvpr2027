"""GPU-resident reference bank with one independently sampled clip per env."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class ReferenceBankCommandCfg(CommandTermCfg):
    reference_dir: str
    fps: float = 30.0
    fixed_trajectory_id: int | None = None
    random_start: bool = True

    def build(self, env) -> "ReferenceBankCommand":
        return ReferenceBankCommand(self, env)


class ReferenceBankCommand(CommandTerm):
    """Samples different exported trajectories for parallel fixed-base ARX envs."""

    def __init__(self, cfg: ReferenceBankCommandCfg, env) -> None:
        super().__init__(cfg, env)
        files = sorted(Path(cfg.reference_dir).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No exported references in {cfg.reference_dir}")
        refs = [np.load(path, allow_pickle=False) for path in files]
        lengths = torch.tensor([ref["qpos_ref"].shape[0] for ref in refs], device=self.device)
        self.lengths = lengths
        max_length = int(lengths.max())
        nq = refs[0]["qpos_ref"].shape[1]
        self.qpos_bank = torch.empty(len(refs), max_length, nq, device=self.device)
        self.qvel_bank = torch.empty_like(self.qpos_bank)
        self.tcp_pos_bank = torch.empty(len(refs), max_length, 2, 3, device=self.device)
        self.tcp_quat_bank = torch.empty(len(refs), max_length, 2, 4, device=self.device)
        self.width_bank = torch.empty(len(refs), max_length, 2, device=self.device)
        for index, ref in enumerate(refs):
            length = int(lengths[index])
            for key, bank in (
                ("qpos_ref", self.qpos_bank),
                ("qvel_ref", self.qvel_bank),
                ("tcp_pos_ref", self.tcp_pos_bank),
                ("tcp_quat_wxyz_ref", self.tcp_quat_bank),
                ("gripper_width_ref", self.width_bank),
            ):
                values = torch.as_tensor(ref[key], dtype=torch.float32, device=self.device)
                bank[index, :length] = values
                bank[index, length:] = values[-1]
            ref.close()
        self.trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_id"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def _indices(self) -> torch.Tensor:
        return torch.floor(self.phase).long()

    @property
    def qpos(self) -> torch.Tensor:
        return self.qpos_bank[self.trajectory_ids, self._indices]

    @property
    def qvel(self) -> torch.Tensor:
        return self.qvel_bank[self.trajectory_ids, self._indices]

    @property
    def tcp_pos(self) -> torch.Tensor:
        return self.tcp_pos_bank[self.trajectory_ids, self._indices]

    @property
    def tcp_quat(self) -> torch.Tensor:
        return self.tcp_quat_bank[self.trajectory_ids, self._indices]

    @property
    def width(self) -> torch.Tensor:
        return self.width_bank[self.trajectory_ids, self._indices]

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.qpos, self.qvel), dim=-1)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """Sample first, then reset the robot to that exact reference state.

        MjLab resets event terms before it resets command terms.  Writing the
        reference state from an event therefore observes the previous clip and
        causes a visible first-frame correction.  Doing it here guarantees the
        selected trajectory and phase are already current.
        """
        assert isinstance(env_ids, torch.Tensor)
        extras = super().reset(env_ids)
        robot = self._env.scene["robot"]
        robot.write_joint_state_to_sim(
            self.qpos[env_ids],
            # Do not inject the finite-difference reference velocity at reset:
            # a large first-frame qvel produces a visible impulse before the
            # first policy action.  qvel_ref remains available to the policy
            # through the reference command.
            torch.zeros_like(self.qvel[env_ids]),
            env_ids=env_ids,
        )
        robot.reset(env_ids=env_ids)
        return extras

    def _update_metrics(self) -> None:
        self.metrics["reference_id"] = self.trajectory_ids.float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.fixed_trajectory_id is None:
            self.trajectory_ids[env_ids] = torch.randint(
                len(self.lengths), (len(env_ids),), device=self.device
            )
        else:
            if not 0 <= self.cfg.fixed_trajectory_id < len(self.lengths):
                raise ValueError(
                    f"fixed_trajectory_id={self.cfg.fixed_trajectory_id} is outside "
                    f"[0, {len(self.lengths)})"
                )
            self.trajectory_ids[env_ids] = self.cfg.fixed_trajectory_id
        if self.cfg.random_start and self.cfg.fixed_trajectory_id is None:
            episode_frames = int(round(self._env.cfg.episode_length_s * self.cfg.fps))
            max_start = torch.clamp(
                self.lengths[self.trajectory_ids[env_ids]] - 1 - episode_frames,
                min=0,
            )
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device) * max_start
        else:
            self.phase[env_ids] = 0.0

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.phase[ids] = torch.minimum(
            self.phase[ids] + self._env.step_dt * self.cfg.fps,
            self.lengths[self.trajectory_ids[ids]].float() - 1.0,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Abstract base class for fingertip → robot-hand retargeting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

from phantom.constants import CONF_THRESHOLD

logger = logging.getLogger(__name__)


class HandRetargeter:
    """Fingertips → robot hand joint angles. Side-aware.

    Subclass contract:
        N_TIPS, N_HAND_DOF, CONFIG_FILENAME_TEMPLATE,
        canonical_joint_names(side), urdf_dir(), config_dir().
    """

    N_TIPS: int = 0
    N_HAND_DOF: int = 0
    CONFIG_FILENAME_TEMPLATE: str = ""

    @classmethod
    def canonical_joint_names(cls, side: str) -> list[str]:
        raise NotImplementedError

    @classmethod
    def urdf_dir(cls) -> Path:
        raise NotImplementedError

    @classmethod
    def config_dir(cls) -> Path:
        raise NotImplementedError

    def _validate_optimizer(self, side: str) -> None:
        pass

    def __init__(
        self,
        side: str,
        urdf_dir: Path | str | None = None,
        config_dir: Path | str | None = None,
    ):
        assert side in ("left", "right"), f"Invalid side: {side}"
        self.side = side

        urdf_dir = Path(urdf_dir) if urdf_dir is not None else self.urdf_dir()
        config_dir = Path(config_dir) if config_dir is not None else self.config_dir()

        RetargetingConfig.set_default_urdf_dir(str(urdf_dir))

        config_path = config_dir / self.CONFIG_FILENAME_TEMPLATE.format(side=side)
        if not config_path.exists():
            raise FileNotFoundError(
                f"{type(self).__name__} config not found: {config_path}"
            )

        cfg = RetargetingConfig.load_from_file(str(config_path))
        self.retargeting: SeqRetargeting = cfg.build()
        self._validate_optimizer(side)

        opt_joint_names = self.retargeting.optimizer.robot.dof_joint_names
        hand_joint_names = [n for n in opt_joint_names if not n.startswith("dummy_")]
        canonical = self.canonical_joint_names(side)
        self._reorder_idx = np.array(
            [hand_joint_names.index(n) for n in canonical], dtype=int,
        )
        self.n_hand_dof = self.N_HAND_DOF
        self.n_tips = self.N_TIPS

    def retarget_frame(self, fingertips_rel: np.ndarray) -> np.ndarray:
        """Retarget one frame of wrist-relative fingertip positions.

        Args:
            fingertips_rel: (N_TIPS, 3) tip positions relative to wrist.

        Returns:
            (N_HAND_DOF,) actuated joint angles in canonical order.
        """
        assert fingertips_rel.shape == (self.N_TIPS, 3)
        qpos_full = self.retargeting.retarget(fingertips_rel.astype(np.float64))
        qpos_hand = qpos_full[6:]
        return qpos_hand[self._reorder_idx].astype(np.float32)

    def retarget_episode(
        self,
        fingertips: np.ndarray,
        wrist_pos: np.ndarray,
        tips_conf: np.ndarray | None = None,
    ) -> np.ndarray:
        """Retarget an entire episode.

        Args:
            fingertips: (T, N_TIPS, 3) fingertip positions.
            wrist_pos:  (T, 3) wrist position.
            tips_conf:  (T, N_TIPS) per-fingertip confidence.

        Returns:
            (T, N_HAND_DOF) actuated joint angles in canonical order.
        """
        T = fingertips.shape[0]
        out = np.zeros((T, self.N_HAND_DOF), dtype=np.float32)
        if tips_conf is None:
            tips_conf = np.ones((T, self.N_TIPS), dtype=np.float32)

        self.retargeting.reset()
        last_good = np.zeros(self.N_HAND_DOF, dtype=np.float32)

        for t in range(T):
            if np.any(tips_conf[t] < CONF_THRESHOLD):
                out[t] = last_good
                continue
            tips_rel = fingertips[t] - wrist_pos[t]
            try:
                q = self.retarget_frame(tips_rel)
                out[t] = q
                last_good = q.copy()
            except Exception as e:
                logger.warning(
                    "%s retargeting failed at frame %d: %s",
                    type(self).__name__, t, e,
                )
                out[t] = last_good
        return out

    def reset(self) -> None:
        self.retargeting.reset()

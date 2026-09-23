"""Camera-frame base search + full-sequence IK. cuRobo if installed, else MuJoCo."""

from __future__ import annotations

import json

import numpy as np

from egodex_arx_replay.gripper import GripperTrajectory

from ..processor import GPU_CUROBO, Artifacts, EpisodeContext, ResourceSpec
from ..robots.arx import ARX_REACH, search_base_and_solve


class BaseIkProcessor:
    name = "base_ik"
    requires = frozenset({Artifacts.ACTION})
    produces = frozenset({Artifacts.BASE, Artifacts.IK})
    optional_requires: frozenset[str] = frozenset()
    resources: ResourceSpec = GPU_CUROBO

    def run(self, ctx: EpisodeContext) -> None:
        with np.load(ctx.out_dir / Artifacts.ACTION) as data:
            targets = GripperTrajectory(
                position=np.asarray(data["position"], dtype=np.float64),
                rotation=np.asarray(data["rotation"], dtype=np.float64),
                width=np.asarray(data["width"], dtype=np.float64),
                valid=np.asarray(data["valid"]),
            )
        result = search_base_and_solve(targets, reach=ARX_REACH)
        base_path = ctx.out_dir / Artifacts.BASE
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(result.base, indent=2, sort_keys=True), encoding="utf-8")
        np.savez_compressed(
            ctx.out_dir / Artifacts.IK,
            qpos=result.qpos.astype(np.float32),
            position_error=result.position_error.astype(np.float32),
            orientation_error=result.orientation_error.astype(np.float32),
            converged=result.converged,
            contacts=np.asarray(result.contacts, dtype=np.int32),
        )

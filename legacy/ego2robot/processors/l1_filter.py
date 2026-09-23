from __future__ import annotations

import json

import numpy as np

from ..processor import CPU, Artifacts, EpisodeContext, ResourceSpec
from ..video import load_masks


class L1FilterProcessor:
    name = "l1_filter"
    requires = frozenset({Artifacts.IK, Artifacts.ROBOT_MASK})
    produces = frozenset({Artifacts.L1})
    optional_requires: frozenset[str] = frozenset()
    resources: ResourceSpec = CPU

    def run(self, ctx: EpisodeContext) -> None:
        with np.load(ctx.out_dir / Artifacts.IK) as data:
            position_error = np.asarray(data["position_error"])
            converged = np.asarray(data["converged"])
            contacts = np.asarray(data["contacts"])
        masks = load_masks(ctx.out_dir / Artifacts.ROBOT_MASK)
        pixels = masks.reshape(masks.shape[0], -1).mean(axis=1)
        ik_ok = np.nanmean(position_error, axis=1) < 0.05
        visible = pixels > 0
        not_huge = pixels < 0.70
        no_contact = contacts <= 1
        valid = ik_ok & visible & not_huge & no_contact
        report = {
            "frames": int(masks.shape[0]),
            "valid_fraction": float(valid.mean()),
            "ik_ok_fraction": float(ik_ok.mean()),
            "visible_fraction": float(visible.mean()),
            "mean_robot_coverage": float(pixels.mean()),
            "converged_fraction": float(converged.mean()),
            "accepted": bool(valid.mean() >= 0.40),
        }
        (ctx.out_dir / Artifacts.L1).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

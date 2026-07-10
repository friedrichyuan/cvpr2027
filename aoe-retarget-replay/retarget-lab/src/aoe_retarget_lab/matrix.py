from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatrixCell:
    trajectory_6dof: str
    hand_source: str
    retargeting: str

    @property
    def key(self) -> str:
        return "__".join(
            [
                f"traj_{self.trajectory_6dof}",
                f"hand_{self.hand_source}",
                f"retarget_{self.retargeting}",
            ]
        )

    @property
    def object_pipeline(self) -> str:
        """Backward-compatible name used by older artifact collectors."""
        return self.trajectory_6dof

    @property
    def retargeter(self) -> str:
        """Backward-compatible retargeter id used by older artifact collectors."""
        return {
            "egoinfinity": "egoinfinity_g1",
            "do_as_i_do": "do_as_i_do_sharpa",
            "spider": "spider_mjwp",
        }.get(self.retargeting, self.retargeting)


def build_cells(config: dict) -> list[MatrixCell]:
    trajectory_6dof = config.get("trajectory_6dof_pipelines", config.get("object_pipelines", []))
    retargeting = config.get("retargeting_pipelines", config.get("retargeters", []))
    retargeting = [
        {
            "egoinfinity_g1": "egoinfinity",
            "do_as_i_do_sharpa": "do_as_i_do",
            "spider_mjwp": "spider",
        }.get(item, item)
        for item in retargeting
    ]
    return [
        MatrixCell(*items)
        for items in itertools.product(
            trajectory_6dof,
            config["hand_sources"],
            retargeting,
        )
    ]


def default_run_dir(config: dict) -> Path:
    return Path(__file__).resolve().parents[2] / "experiments" / config["run_name"]

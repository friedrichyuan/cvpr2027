from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path


TRAJECTORY_6DOF_OPTIONS = ("egoinfinity", "do_as_i_do")
HAND_SOURCE_OPTIONS = ("aoe", "estimated")
RETARGETING_OPTIONS = ("egoinfinity", "do_as_i_do", "spider")


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


def all_matrix_cells() -> tuple[MatrixCell, ...]:
    return tuple(
        MatrixCell(*items)
        for items in itertools.product(
            TRAJECTORY_6DOF_OPTIONS,
            HAND_SOURCE_OPTIONS,
            RETARGETING_OPTIONS,
        )
    )


def parse_cell_key(value: str) -> MatrixCell:
    parts = value.split("__")
    if len(parts) != 3:
        raise ValueError(f"invalid matrix cell key: {value}")
    prefixes = ("traj_", "hand_", "retarget_")
    if any(not part.startswith(prefix) for part, prefix in zip(parts, prefixes)):
        raise ValueError(f"invalid matrix cell key: {value}")
    return select_matrix_cells(
        trajectory_6dof=parts[0][len(prefixes[0]) :],
        hand_source=parts[1][len(prefixes[1]) :],
        retargeting=parts[2][len(prefixes[2]) :],
    )[0]


def select_matrix_cells(
    *,
    cell: str | None = None,
    trajectory_6dof: str | None = None,
    hand_source: str | None = None,
    retargeting: str | None = None,
) -> tuple[MatrixCell, ...]:
    axes = (trajectory_6dof, hand_source, retargeting)
    if cell is not None and any(value is not None for value in axes):
        raise ValueError("--cell cannot be combined with matrix axis options")
    if cell is not None:
        return (parse_cell_key(cell),)
    if not any(value is not None for value in axes):
        return all_matrix_cells()
    if not all(value is not None for value in axes):
        raise ValueError(
            "--trajectory-6dof, --hand-source, and --retargeting must be provided together"
        )
    if trajectory_6dof not in TRAJECTORY_6DOF_OPTIONS:
        raise ValueError(f"invalid trajectory_6dof: {trajectory_6dof}")
    if hand_source not in HAND_SOURCE_OPTIONS:
        raise ValueError(f"invalid hand_source: {hand_source}")
    if retargeting not in RETARGETING_OPTIONS:
        raise ValueError(f"invalid retargeting: {retargeting}")
    return (MatrixCell(trajectory_6dof, hand_source, retargeting),)


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

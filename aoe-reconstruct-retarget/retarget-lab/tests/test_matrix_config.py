from pathlib import Path
import json

import pytest

from aoe_retarget_lab.matrix import (
    all_matrix_cells,
    parse_cell_key,
    select_matrix_cells,
)


def test_foundation_matrix_has_12_cells():
    cfg = json.loads(Path("configs/foundation_jar_bimanual.json").read_text())
    assert len(cfg["object_pipelines"]) * len(cfg["hand_sources"]) * len(cfg["retargeters"]) == 12


def test_default_matrix_has_12_unique_cells():
    cells = all_matrix_cells()
    assert len(cells) == 12
    assert len({cell.key for cell in cells}) == 12


@pytest.mark.parametrize("cell", all_matrix_cells(), ids=lambda cell: cell.key)
def test_each_matrix_cell_can_be_selected_individually(cell):
    assert select_matrix_cells(cell=cell.key) == (cell,)
    assert parse_cell_key(cell.key) == cell
    assert select_matrix_cells(
        trajectory_6dof=cell.trajectory_6dof,
        hand_source=cell.hand_source,
        retargeting=cell.retargeting,
    ) == (cell,)


def test_partial_or_conflicting_cell_selection_is_rejected():
    with pytest.raises(ValueError, match="provided together"):
        select_matrix_cells(trajectory_6dof="egoinfinity")
    with pytest.raises(ValueError, match="cannot be combined"):
        select_matrix_cells(
            cell=all_matrix_cells()[0].key,
            trajectory_6dof="egoinfinity",
            hand_source="aoe",
            retargeting="spider",
        )

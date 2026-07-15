from pathlib import Path
import json


def test_foundation_matrix_has_12_cells():
    cfg = json.loads(Path("configs/foundation_jar_bimanual.json").read_text())
    assert len(cfg["object_pipelines"]) * len(cfg["hand_sources"]) * len(cfg["retargeters"]) == 12

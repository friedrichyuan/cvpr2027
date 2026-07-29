from __future__ import annotations

from pathlib import Path


def object_id_from_task(task: str) -> str:
    for marker in ["_bimanual", "_right", "_left"]:
        if marker in task:
            return task.split(marker, 1)[0]
    return task


def find_hand_mesh_npz(raw_dir: Path, task: str) -> Path:
    candidates = [
        raw_dir / task / "all_hand_meshes.npz",
        raw_dir / "raw" / "all_hand_meshes.npz",
        raw_dir / "all_hand_meshes.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "missing hand mesh file; checked: "
        + ", ".join(str(path) for path in candidates)
    )

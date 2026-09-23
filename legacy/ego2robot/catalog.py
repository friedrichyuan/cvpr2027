"""Scan EgoDex `{split}/{task}/{id}.hdf5` and optional sibling `.mp4`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .processor import EpisodeContext


@dataclass(frozen=True)
class EpisodeRecord:
    split: str
    task: str
    episode_id: str
    hdf5: str
    video: str | None


def scan_egodex(root: Path) -> list[EpisodeRecord]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EgoDex root does not exist: {root}")
    records: list[EpisodeRecord] = []
    for hdf5 in sorted(root.glob("*/*/*.hdf5")):
        split, task = hdf5.parent.parent.name, hdf5.parent.name
        video = hdf5.with_suffix(".mp4")
        records.append(
            EpisodeRecord(
                split=split,
                task=task,
                episode_id=hdf5.stem,
                hdf5=str(hdf5),
                video=str(video) if video.is_file() else None,
            )
        )
    return records


def write_index(records: list[EpisodeRecord], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return path


def read_index(path: Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(EpisodeRecord(**json.loads(line)))
    return records


def to_context(record: EpisodeRecord, output_root: Path) -> EpisodeContext:
    return EpisodeContext(
        split=record.split,
        task=record.task,
        episode_id=record.episode_id,
        hdf5_path=Path(record.hdf5),
        video_path=Path(record.video) if record.video else None,
        out_dir=Path(output_root) / record.split / record.task / record.episode_id,
    )

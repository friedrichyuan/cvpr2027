"""On-disk artifacts. A stage is done only after atomic rename of its .done file."""

from __future__ import annotations

import json
from pathlib import Path

from .processor import EpisodeContext


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def episode_dir(self, ctx: EpisodeContext) -> Path:
        return self.root / ctx.split / ctx.task / ctx.episode_id

    def path(self, ctx: EpisodeContext, artifact: str) -> Path:
        return self.episode_dir(ctx) / artifact

    def exists(self, ctx: EpisodeContext, artifact: str) -> bool:
        return self.path(ctx, artifact).is_file()

    def done_path(self, ctx: EpisodeContext, processor: str) -> Path:
        return self.episode_dir(ctx) / f".done.{processor}"

    def is_done(self, ctx: EpisodeContext, processor: str) -> bool:
        return self.done_path(ctx, processor).is_file()

    def mark_done(self, ctx: EpisodeContext, processor: str) -> None:
        path = self.done_path(ctx, processor)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(processor, encoding="utf-8")
        tmp.replace(path)

    def write_json(self, ctx: EpisodeContext, artifact: str, payload: object) -> Path:
        path = self.path(ctx, artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    def read_json(self, ctx: EpisodeContext, artifact: str) -> object:
        return json.loads(self.path(ctx, artifact).read_text(encoding="utf-8"))

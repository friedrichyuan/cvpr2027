"""CPU dataloader. Decodes the next visual batch while a GPU is busy."""

from __future__ import annotations

from pathlib import Path

from egowhale.media import load_masks, read_rgb
from egowhale.step import DEPTH, INPAINT, MASKS


def prepare(stage: str, pairs: list[tuple[str, str]]) -> dict:
    items = []
    for src, dst in pairs:
        try:
            items.append(_one(stage, Path(src), Path(dst)))
        except Exception as exc:
            items.append({"ok": False, "error": str(exc).splitlines()[-1]})
    return {"items": items}


def _one(stage: str, src: Path, dst: Path) -> dict:
    if stage == "segment":
        frames, _fps = read_rgb(src.with_suffix(".mp4"))
        return {"ok": True, "frames": frames, "dst": str(dst / MASKS)}
    if stage == "inpaint":
        from egowhale.visual.paint import prepare_clip

        frames, fps = read_rgb(src.with_suffix(".mp4"))
        masks = load_masks(dst / MASKS)
        if len(frames) != len(masks):
            raise ValueError(f"{len(frames)} frames vs {len(masks)} masks")
        clip = prepare_clip(frames, masks)
        clip["ok"] = True
        clip["fps"] = float(fps)
        clip["dst"] = str(dst / INPAINT)
        return clip
    if stage == "depth":
        from egowhale.visual.depth import prepare_view

        frames, _fps = read_rgb(dst / INPAINT)
        view = prepare_view(src, frames)
        view["ok"] = True
        view["dst"] = str(dst / DEPTH)
        return view
    raise ValueError(f"no feed for {stage}")

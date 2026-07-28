from __future__ import annotations

from os import PathLike
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_path(path: str | PathLike[str], root: str | PathLike[str]) -> str:
    """Return a path relative to *root* when possible, otherwise unchanged."""

    candidate = Path(path)
    try:
        return str(candidate.relative_to(Path(root)))
    except ValueError:
        return str(candidate)


def optional_relative_path(
    path: str | PathLike[str] | None,
    root: str | PathLike[str],
) -> str | None:
    if path is None:
        return None
    return relative_path(path, root)


def path_contains(path: str | PathLike[str] | None, needle: str) -> bool:
    """Search both lexical and resolved representations of a path."""

    if path is None:
        return False
    candidate = Path(path)
    text = str(candidate)
    try:
        text = f"{text} {candidate.resolve()}"
    except OSError:
        pass
    return needle in text

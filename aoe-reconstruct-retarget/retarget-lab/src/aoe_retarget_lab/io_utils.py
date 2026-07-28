from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterable


DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def file_sha256(
    path: str | PathLike[str],
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Return a streaming SHA256 digest for one file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(
    path: str | PathLike[str] | None,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str | None:
    """Hash an existing regular file, otherwise return ``None``."""

    if path is None or not Path(path).is_file():
        return None
    return file_sha256(path, chunk_size=chunk_size)


def ordered_files_sha256(paths: Iterable[str | PathLike[str]]) -> str:
    """Hash ordered basenames and raw per-file SHA256 bytes."""

    digest = hashlib.sha256()
    for value in paths:
        path = Path(value)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def file_sha256_binding(path: str | PathLike[str]) -> dict[str, object]:
    """Bind a resolved file path to its digest without raising for bad files."""

    resolved = Path(path).expanduser().resolve()
    binding: dict[str, object] = {"path": str(resolved), "sha256": None}
    if not resolved.is_file():
        binding["error"] = "missing"
        return binding
    try:
        binding["sha256"] = file_sha256(resolved)
    except OSError as exc:
        binding["error"] = f"unreadable:{type(exc).__name__}"
    return binding


def read_json(path: str | PathLike[str]) -> Any:
    """Read JSON while preserving the decoder's strict error behavior."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_if_exists(path: str | PathLike[str]) -> Any | None:
    """Read JSON, returning ``None`` only when the file is absent."""

    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def read_json_object(
    path: str | PathLike[str] | None,
) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` for missing or invalid input."""

    if path is None:
        return None
    try:
        payload = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_json_object_lenient(
    path: str | PathLike[str] | None,
) -> dict[str, Any] | None:
    """Preserve legacy best-effort manifest reads that suppress all errors."""

    if path is None or not Path(path).exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None

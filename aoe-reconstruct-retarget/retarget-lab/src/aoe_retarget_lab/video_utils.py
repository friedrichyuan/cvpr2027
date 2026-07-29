from __future__ import annotations

import json
import subprocess
from pathlib import Path


def ffmpeg_writer(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def probe_video(path: Path) -> dict[str, float | int] | None:
    """Return the minimum metadata needed to decide whether a video is reviewable."""

    if not path.is_file() or path.stat().st_size < 1024:
        return None
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        stream = json.loads(proc.stdout)["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        raw_frames = stream.get("nb_frames")
        if raw_frames in {None, "", "N/A"}:
            raw_frames = stream.get("nb_read_frames")
        frames = 0 if raw_frames in {None, "", "N/A"} else int(raw_frames)
        raw_duration = stream.get("duration")
        duration = 0.0 if raw_duration in {None, "", "N/A"} else float(raw_duration)
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if width <= 0 or height <= 0 or (frames <= 0 and duration <= 0.0):
        return None
    return {
        "width": width,
        "height": height,
        "frames": frames,
        "duration": duration,
    }


def is_reviewable_video(path: Path) -> bool:
    return probe_video(path) is not None

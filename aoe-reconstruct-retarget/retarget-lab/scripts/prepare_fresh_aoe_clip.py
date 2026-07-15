#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/mnt/nas/share/home/hjd/repro/datasets/openaoe-3hours/extracted/poc_deliver")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def object_id(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", name.strip().lower()).strip("_")
    if not value:
        raise ValueError(f"cannot derive object id from {name!r}")
    return value


def first_action(annotation: dict) -> dict:
    actions = annotation.get("atomic_action") or []
    if not actions:
        raise ValueError(f"annotation has no atomic_action: {annotation}")
    return actions[0]


def find_annotation(segment_dir: Path, annotation_id: int) -> dict:
    path = segment_dir / "ego_annotation" / "ego_action_annotation.json"
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"expected annotation list: {path}")
    for row in rows:
        if int(row.get("id")) == annotation_id:
            return row
    raise KeyError(f"annotation id {annotation_id} not found in {path}")


def ffprobe_fps(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if "/" in out:
        num, den = out.split("/", 1)
        return float(num) / float(den)
    return float(out)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean AoE RGB clip for a fresh 12-demo source run.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--annotation-id", required=True, type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--object-name")
    parser.add_argument("--anchor-hand", required=True, choices=["left", "right", "bimanual"])
    parser.add_argument("--duration-sec", type=float, default=0.0, help="Override annotation duration when > 0.")
    parser.add_argument("--extra-sec-after", type=float, default=0.0)
    parser.add_argument("--scale-width", type=int, default=1280, help="Set 0 to keep original width.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    segment_dir = dataset_root / args.segment
    source_video = segment_dir / "ego_process" / "ego_undistorted_video" / "raw_video_undistorted.mp4"
    hands_npz = segment_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    if not source_video.exists():
        raise FileNotFoundError(source_video)
    if not hands_npz.exists():
        raise FileNotFoundError(hands_npz)

    annotation = find_annotation(segment_dir, args.annotation_id)
    action = first_action(annotation)
    obj_name = args.object_name or str(action.get("object") or "")
    obj_id = object_id(obj_name)
    start_sec = float(annotation.get("start_ts", 0.0))
    end_sec = float(annotation.get("end_ts", start_sec))
    duration = args.duration_sec if args.duration_sec > 0 else max(0.0, end_sec - start_sec)
    duration += args.extra_sec_after
    if duration <= 0:
        raise ValueError(f"invalid duration for annotation {args.annotation_id}: {duration}")

    source_fps = ffprobe_fps(source_video)
    start_frame = int(round(start_sec * source_fps))
    num_frames = int(round(duration * source_fps))
    end_frame = start_frame + num_frames
    ref_frame_in_clip = max(0, min(num_frames - 1, num_frames // 2))

    out_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / args.run_name
        / "intermediates"
        / "trajectory_6dof"
        / "do_as_i_do"
        / "reconstruction"
        / "clip"
    )
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to recreate only for a fresh run")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_mp4 = out_dir / "raw.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.6f}",
    ]
    if args.scale_width > 0:
        ffmpeg_cmd += ["-vf", f"scale={args.scale_width}:-2"]
    ffmpeg_cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(raw_mp4)]
    run(ffmpeg_cmd)

    config = {
        "frame_number": ref_frame_in_clip,
        "object_names": [obj_id],
        "anchor_hand": args.anchor_hand,
        "prompt": obj_name,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "dataset_root": str(dataset_root),
        "segment": args.segment,
        "segment_dir": str(segment_dir),
        "annotation_id": args.annotation_id,
        "annotation": annotation,
        "action": action,
        "source_video": str(source_video),
        "source_fps": source_fps,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": start_sec,
        "duration": duration,
        "object_name": obj_name,
        "object_id": obj_id,
        "anchor_hand": args.anchor_hand,
        "ref_source_frame": start_frame + ref_frame_in_clip,
        "ref_frame_in_clip": ref_frame_in_clip,
        "hands_npz": str(hands_npz),
        "trimmed_video": str(raw_mp4),
        "ffmpeg_command": ffmpeg_cmd,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "fresh_input_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"clip_dir": str(out_dir), "raw_mp4": str(raw_mp4), "metadata": metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

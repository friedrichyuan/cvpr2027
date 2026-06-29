#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_INPUT_ROOT = Path("/PATH_TO/general_datasets/Open-AoE/poc_deliver")
DEFAULT_OUTPUT_ROOT = Path("/PATH_TO/dataset/Open_AoE/open_aoe_mano_nextstate_224_110d")
DEFAULT_REPO_ID = "local/open-aoe-mano-nextstate-224-110d"
FALLBACK_TASK = "egocentric hand manipulation"


HAND_POSE_DIMS = 45
HAND_VECTOR_DIMS = 1 + 3 + 3 + 3 + HAND_POSE_DIMS
STATE_DIMS = 2 * HAND_VECTOR_DIMS
ACTION_DIMS = 2 * HAND_VECTOR_DIMS


@dataclass
class EpisodeSummary:
    sample: str
    frames_written: int
    source_frames: int
    source_fps: float
    annotation_segments_used: int
    annotation_segments_ignored: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Open-AoE egocentric hand data into a LeRobot dataset. "
            "The generated dataset uses observation.images.front, observation.state, action, and task."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--robot-type", type=str, default="open_aoe_egocentric_mano")
    parser.add_argument("--dataset-fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vcodec", type=str, default="h264")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--skip-sample", action="append", default=[])
    parser.add_argument(
        "--only-source-fps",
        type=float,
        default=None,
        help="If set, keep only samples whose undistorted_video_info fps rounds to this value.",
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_dirs(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
        path.rmdir()
    path.parent.mkdir(parents=True, exist_ok=True)


def resize_letterbox_rgb(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return np.ascontiguousarray(canvas)


def hand_feature_names(prefix: str) -> list[str]:
    return (
        [f"{prefix}.valid"]
        + [f"{prefix}.wrist_trans_cam.{axis}" for axis in ("x", "y", "z")]
        + [f"{prefix}.wrist_rot_cam_axis_angle.{axis}" for axis in ("x", "y", "z")]
        + [f"{prefix}.wrist_vel_cam.{axis}" for axis in ("x", "y", "z")]
        + [f"{prefix}.mano_pose.{i}" for i in range(HAND_POSE_DIMS)]
    )


STATE_CORE_NAMES = hand_feature_names("state.left") + hand_feature_names("state.right")
STATE_NAMES = STATE_CORE_NAMES
ACTION_NAMES = hand_feature_names("action.left_next") + hand_feature_names("action.right_next")


def build_features(image_size: int) -> dict[str, dict]:
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIMS,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIMS,),
            "names": ACTION_NAMES,
        },
    }


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def format_action_task(scene: str, action: dict) -> str:
    verb = str(action.get("verb") or "manipulate").strip().replace("_", " ")
    obj = str(action.get("object") or "object").strip().replace("_", " ")
    hand = str(action.get("hand") or "hand").strip().replace("_", " ")
    scene = str(scene or "indoor").strip().replace("_", " ")
    return f"In {scene}, use {hand} hand to {verb} {obj}."


def load_annotation_tasks(sample_dir: Path, num_frames: int) -> tuple[list[tuple[int, int, str]], int]:
    ann_path = sample_dir / "ego_annotation" / "ego_action_annotation.json"
    if not ann_path.exists():
        return [], 0
    try:
        raw_segments = read_json(ann_path)
    except Exception:
        return [], 0
    if not isinstance(raw_segments, list):
        return [], 0

    segments: list[tuple[int, int, str]] = []
    ignored = 0
    for seg in raw_segments:
        if not isinstance(seg, dict):
            ignored += 1
            continue
        start = safe_int(seg.get("start_frame"), 0)
        end = safe_int(seg.get("end_frame"), start)
        if end <= start:
            ignored += 1
            continue
        start = max(0, min(start, num_frames))
        end = max(0, min(end, num_frames))
        if end <= start:
            ignored += 1
            continue
        actions = seg.get("atomic_action") or []
        if not isinstance(actions, list) or len(actions) == 0 or not isinstance(actions[0], dict):
            task = FALLBACK_TASK
        else:
            task = format_action_task(str(seg.get("scene") or "indoor"), actions[0])
        segments.append((start, end, task))
    return sorted(segments, key=lambda item: item[0]), ignored


def task_for_frame(segments: list[tuple[int, int, str]], frame_index: int, cursor: int) -> tuple[str, int]:
    while cursor + 1 < len(segments) and frame_index >= segments[cursor][1]:
        cursor += 1
    if segments and segments[cursor][0] <= frame_index < segments[cursor][1]:
        return segments[cursor][2], cursor
    return FALLBACK_TASK, cursor


def hand_vectors(hands: np.lib.npyio.NpzFile, fps: float) -> np.ndarray:
    valid = np.asarray(hands["pred_valid"], dtype=np.float32) > 0.5
    trans = np.asarray(hands["pred_trans_cam"], dtype=np.float32)
    rot = np.asarray(hands["pred_rot_cam"], dtype=np.float32)
    pose = np.asarray(hands["pred_hand_pose"], dtype=np.float32)[..., :HAND_POSE_DIMS]

    velocity = np.zeros_like(trans, dtype=np.float32)
    velocity[:, 1:] = (trans[:, 1:] - trans[:, :-1]) * float(fps)
    velocity[:, 1:] *= (valid[:, 1:] & valid[:, :-1])[..., None]

    out = np.zeros((2, valid.shape[1], HAND_VECTOR_DIMS), dtype=np.float32)
    for hand_idx in range(2):
        mask = valid[hand_idx]
        out[hand_idx, :, 0] = mask.astype(np.float32)
        out[hand_idx, mask, 1:4] = trans[hand_idx, mask]
        out[hand_idx, mask, 4:7] = rot[hand_idx, mask]
        out[hand_idx, mask, 7:10] = velocity[hand_idx, mask]
        out[hand_idx, mask, 10:] = pose[hand_idx, mask]
    return out


def state_from_vectors(vectors: np.ndarray, frame_index: int) -> np.ndarray:
    state = np.zeros((STATE_DIMS,), dtype=np.float32)
    state[:HAND_VECTOR_DIMS] = vectors[0, frame_index]
    state[HAND_VECTOR_DIMS : 2 * HAND_VECTOR_DIMS] = vectors[1, frame_index]
    return state


def action_from_vectors(vectors: np.ndarray, frame_index: int) -> np.ndarray:
    action = np.zeros((ACTION_DIMS,), dtype=np.float32)
    action[:HAND_VECTOR_DIMS] = vectors[0, frame_index]
    action[HAND_VECTOR_DIMS:] = vectors[1, frame_index]
    return action


def source_fps(sample_dir: Path) -> float:
    info_path = sample_dir / "ego_process" / "ego_undistorted_video" / "undistorted_video_info.json"
    if not info_path.exists():
        return 30.0
    info = read_json(info_path)
    return safe_float(info.get("fps"), 30.0)


def convert_episode(
    dataset: LeRobotDataset,
    sample_dir: Path,
    image_size: int,
    stride: int,
    max_frames_per_episode: int | None,
) -> EpisodeSummary:
    hands_path = sample_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    video_path = sample_dir / "ego_process" / "ego_undistorted_video" / "raw_video_undistorted.mp4"
    if not hands_path.exists():
        raise FileNotFoundError(f"Missing hands.npz: {hands_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing undistorted video: {video_path}")

    fps = source_fps(sample_dir)
    with np.load(hands_path, mmap_mode="r") as hands:
        num_frames = int(hands["pred_valid"].shape[1])
        vectors = hand_vectors(hands, fps=fps)

    segments, ignored_segments = load_annotation_tasks(sample_dir, num_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames_written = 0
    task_cursor = 0
    try:
        for frame_index in range(num_frames):
            ok, bgr = cap.read()
            if not ok:
                logging.warning("Video ended early at frame %d/%d for %s", frame_index, num_frames, sample_dir.name)
                break
            if frame_index % stride != 0:
                continue
            if max_frames_per_episode is not None and frames_written >= max_frames_per_episode:
                break
            target_index = min(frame_index + stride, num_frames - 1)
            task, task_cursor = task_for_frame(segments, frame_index, task_cursor)
            dataset.add_frame(
                {
                    "task": task,
                    "observation.images.front": resize_letterbox_rgb(bgr, image_size),
                    "observation.state": np.ascontiguousarray(state_from_vectors(vectors, frame_index)),
                    "action": np.ascontiguousarray(action_from_vectors(vectors, target_index)),
                }
            )
            frames_written += 1
    finally:
        cap.release()

    if frames_written == 0:
        dataset.clear_episode_buffer(delete_images=True)
    else:
        dataset.save_episode()

    return EpisodeSummary(
        sample=sample_dir.name,
        frames_written=frames_written,
        source_frames=num_frames,
        source_fps=fps,
        annotation_segments_used=len(segments),
        annotation_segments_ignored=ignored_segments,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    skipped = set(args.skip_sample or [])
    candidates = [p for p in sample_dirs(input_root) if p.name not in skipped]

    if args.only_source_fps is not None:
        target = float(args.only_source_fps)
        candidates = [p for p in candidates if round(source_fps(p)) == round(target)]

    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise FileNotFoundError(f"No samples selected under {input_root}")

    output_fps = max(1, int(round(args.dataset_fps / args.stride)))
    prepare_output_root(output_root, args.overwrite)
    logging.info("Selected %d samples from %s", len(candidates), input_root)
    logging.info("Writing %s repo_id=%s fps=%d image_size=%d", output_root, args.repo_id, output_fps, args.image_size)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        robot_type=args.robot_type,
        fps=output_fps,
        features=build_features(args.image_size),
        use_videos=True,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        image_writer_threads=args.image_writer_threads,
    )

    summaries: list[EpisodeSummary] = []
    try:
        for idx, sample_dir in enumerate(candidates, start=1):
            logging.info("[%d/%d] converting %s", idx, len(candidates), sample_dir.name)
            summary = convert_episode(
                dataset=dataset,
                sample_dir=sample_dir,
                image_size=args.image_size,
                stride=args.stride,
                max_frames_per_episode=args.max_frames_per_episode,
            )
            summaries.append(summary)
            logging.info(
                "[%d/%d] saved %s frames_written=%d source_frames=%d ann_used=%d ann_ignored=%d",
                idx,
                len(candidates),
                summary.sample,
                summary.frames_written,
                summary.source_frames,
                summary.annotation_segments_used,
                summary.annotation_segments_ignored,
            )
    finally:
        dataset.finalize()

    summary_path = output_root / "open_aoe_conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "repo_id": args.repo_id,
                "dataset_fps": output_fps,
                "image_size": args.image_size,
                "stride": args.stride,
                "state_dims": STATE_DIMS,
                "action_dims": ACTION_DIMS,
                "state_names": STATE_NAMES,
                "action_names": ACTION_NAMES,
                "episodes": [asdict(item) for item in summaries],
                "total_frames_written": int(sum(item.frames_written for item in summaries)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("Finished conversion. Summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

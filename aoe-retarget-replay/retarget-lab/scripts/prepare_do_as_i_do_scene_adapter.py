#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TIP_VERTICES = [745, 320, 443, 554, 671]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")


def remove_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    remove_existing(dst)
    dst.symlink_to(src)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    remove_existing(dst)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    remove_existing(dst)
    shutil.copytree(src, dst, symlinks=True)


def infer_object_id(source_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    config = load_json(source_dir / "config.json")
    objects = config.get("object_names") or []
    if not objects:
        raise RuntimeError(f"cannot infer object id from {source_dir / 'config.json'}")
    return str(objects[0])


def choose_hand_npz(source_dir: Path, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            source_dir / "raw" / "all_hand_meshes_46f.npz",
            source_dir / "raw" / "all_hand_meshes.npz",
            source_dir / "all_hand_meshes.npz",
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("missing hand mesh npz; checked " + ", ".join(str(p) for p in candidates))


def expand_16_to_21(joints: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints)
    vertices = np.asarray(vertices)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"expected joints as (N,J,3), got {joints.shape}")
    if joints.shape[1] == 21:
        return joints.astype(np.float32, copy=False)
    if joints.shape[1] != 16:
        raise ValueError(f"cannot adapt {joints.shape[1]} joints to OpenPose-21")
    if vertices.shape[0] != joints.shape[0] or vertices.shape[1] <= max(TIP_VERTICES):
        raise ValueError(f"vertices {vertices.shape} incompatible with joints {joints.shape}")

    out = np.empty((joints.shape[0], 21, 3), dtype=np.float32)
    out[:, 0] = joints[:, 0]
    out[:, 1], out[:, 2], out[:, 3], out[:, 4] = (
        joints[:, 13],
        joints[:, 14],
        joints[:, 15],
        vertices[:, TIP_VERTICES[0]],
    )
    out[:, 5], out[:, 6], out[:, 7], out[:, 8] = (
        joints[:, 1],
        joints[:, 2],
        joints[:, 3],
        vertices[:, TIP_VERTICES[1]],
    )
    out[:, 9], out[:, 10], out[:, 11], out[:, 12] = (
        joints[:, 4],
        joints[:, 5],
        joints[:, 6],
        vertices[:, TIP_VERTICES[2]],
    )
    out[:, 13], out[:, 14], out[:, 15], out[:, 16] = (
        joints[:, 10],
        joints[:, 11],
        joints[:, 12],
        vertices[:, TIP_VERTICES[3]],
    )
    out[:, 17], out[:, 18], out[:, 19], out[:, 20] = (
        joints[:, 7],
        joints[:, 8],
        joints[:, 9],
        vertices[:, TIP_VERTICES[4]],
    )
    return out


def write_adapted_hand_npz(src: Path, dst: Path) -> dict:
    data = np.load(src, allow_pickle=True)
    out = {key: data[key] for key in data.files}
    stats: dict[str, object] = {"source": str(src), "sides": {}}
    if "source_frame_ids" in data:
        source_frame_ids = np.asarray(data["source_frame_ids"]).astype(int)
        stats["source_frame_ids"] = {
            "count": int(source_frame_ids.size),
            "first": int(source_frame_ids[0]) if source_frame_ids.size else None,
            "last": int(source_frame_ids[-1]) if source_frame_ids.size else None,
            "sample": source_frame_ids[: min(8, source_frame_ids.size)].tolist(),
        }
    for side in ("left", "right"):
        joints_key = f"{side}_joints"
        vertices_key = f"{side}_vertices"
        if joints_key not in out or vertices_key not in out:
            continue
        before = tuple(np.asarray(out[joints_key]).shape)
        out[joints_key] = expand_16_to_21(np.asarray(out[joints_key]), np.asarray(out[vertices_key]))
        after = tuple(np.asarray(out[joints_key]).shape)
        stats["sides"][side] = {"joints_before": before, "joints_after": after}

    out["mano16_to_openpose21_adapter"] = np.array(
        "MANO-16 to OpenPose-21 using MANO fingertip vertices [745,320,443,554,671]",
        dtype="<U96",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    stats["output"] = str(dst)
    return stats


def hand_projection_qc(hand_npz: Path, frames_dir: Path, anchor_hand: str) -> dict:
    data = np.load(hand_npz, allow_pickle=False)
    vertices_key = f"{anchor_hand}_vertices"
    valid_key = f"{anchor_hand}_valid"
    if vertices_key not in data:
        return {"anchor_hand": anchor_hand, "available": False}
    vertices = np.asarray(data[vertices_key])
    valid_frames = np.asarray(data[valid_key]) if valid_key in data else np.ones(vertices.shape[0])
    frame_ids = sorted(set([0, vertices.shape[0] // 2, vertices.shape[0] - 1]))
    frames = []
    for frame_id in frame_ids:
        intrinsics_path = frames_dir / f"{frame_id:06d}_intrinsics.npy"
        shape = frame_shape(frames_dir, frame_id)
        if shape is None or not intrinsics_path.exists() or float(valid_frames[frame_id]) <= 0:
            frames.append({"frame": int(frame_id), "available": False})
            continue
        h, w = shape
        intrinsics = np.load(intrinsics_path)
        uv, valid = project_vertices(vertices[frame_id], intrinsics)
        finite = valid & np.isfinite(uv).all(axis=1)
        if not finite.any():
            frames.append({"frame": int(frame_id), "available": False})
            continue
        uv_valid = uv[finite]
        inside = (
            (uv_valid[:, 0] >= 0)
            & (uv_valid[:, 0] < w)
            & (uv_valid[:, 1] >= 0)
            & (uv_valid[:, 1] < h)
        )
        frames.append(
            {
                "frame": int(frame_id),
                "available": True,
                "bbox_xyxy": [
                    float(uv_valid[:, 0].min()),
                    float(uv_valid[:, 1].min()),
                    float(uv_valid[:, 0].max()),
                    float(uv_valid[:, 1].max()),
                ],
                "inside_fraction": float(inside.mean()),
                "image_wh": [int(w), int(h)],
            }
        )
    return {"anchor_hand": anchor_hand, "available": True, "frames": frames}


def frame_shape(frames_dir: Path, frame_id: int) -> tuple[int, int] | None:
    pointmap = frames_dir / f"{frame_id:06d}_pointmap.npy"
    if pointmap.exists():
        arr = np.load(pointmap, mmap_mode="r")
        return int(arr.shape[0]), int(arr.shape[1])
    rgb = cv2.imread(str(frames_dir / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
    if rgb is not None:
        h, w = rgb.shape[:2]
        return h, w
    return None


def project_vertices(vertices: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = vertices[:, 2]
    valid = z > 1e-6
    uv = np.zeros((vertices.shape[0], 2), dtype=np.float32)
    uv[:, 0] = intrinsics[0, 0] * vertices[:, 0] / np.maximum(z, 1e-6) + intrinsics[0, 2]
    uv[:, 1] = intrinsics[1, 1] * vertices[:, 1] / np.maximum(z, 1e-6) + intrinsics[1, 2]
    return uv, valid


def generate_hand_masks(hand_npz: Path, frames_dir: Path, masks_dir: Path) -> dict:
    data = np.load(hand_npz, allow_pickle=False)
    stats: dict[str, object] = {"output_dir": str(masks_dir), "sides": {}}
    for side in ("left", "right"):
        vertices_key = f"{side}_vertices"
        faces_key = f"{side}_faces"
        valid_key = f"{side}_valid"
        if vertices_key not in data or faces_key not in data:
            continue
        vertices = np.asarray(data[vertices_key])
        faces = np.asarray(data[faces_key], dtype=np.int32)
        valid_frames = np.asarray(data[valid_key]) if valid_key in data else np.ones(vertices.shape[0])
        written = 0
        for frame_id in range(vertices.shape[0]):
            if float(valid_frames[frame_id]) <= 0:
                continue
            intrinsics_path = frames_dir / f"{frame_id:06d}_intrinsics.npy"
            if not intrinsics_path.exists():
                continue
            shape = frame_shape(frames_dir, frame_id)
            if shape is None:
                continue
            h, w = shape
            intrinsics = np.load(intrinsics_path)
            uv, valid = project_vertices(vertices[frame_id], intrinsics)
            mask = np.zeros((h, w), dtype=np.uint8)
            for tri in faces:
                if np.any(tri < 0) or np.any(tri >= len(vertices[frame_id])) or not bool(valid[tri].all()):
                    continue
                pts = np.rint(uv[tri]).astype(np.int32)
                if pts[:, 0].max() < 0 or pts[:, 0].min() >= w or pts[:, 1].max() < 0 or pts[:, 1].min() >= h:
                    continue
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                cv2.fillConvexPoly(mask, pts, 255)
            mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            out_dir = masks_dir / f"frame_{frame_id:06d}_masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{side}_hand_0.png"), mask)
            written += 1
        stats["sides"][side] = {"frames_written": written, "frames_total": int(vertices.shape[0])}
    return stats


def link_tracking_assets(source_dir: Path, output_dir: Path, object_id: str, optimize_scale: bool) -> None:
    src_obj = source_dir / "obj_tracking_out" / object_id
    dst_obj = output_dir / "obj_tracking_out" / object_id
    dst_layout = dst_obj / "combined_visualization"
    dst_layout.mkdir(parents=True, exist_ok=True)
    require(src_obj / "combined_visualization" / "layout_camera_frame.json", "Do-as-I-Do layout_camera_frame.json")

    for path in sorted(src_obj.glob("frame_*_samples.pt")) + sorted(src_obj.glob("guided_poses.pt")):
        link_file(path.resolve(), dst_obj / path.name)

    for path in sorted((src_obj / "combined_visualization").iterdir()):
        if optimize_scale and path.name == "layout_camera_frame_optimized.json":
            continue
        link_file(path.resolve(), dst_layout / path.name)

    optimized = dst_layout / "layout_camera_frame_optimized.json"
    if not optimize_scale and not optimized.exists():
        copy_file(src_obj / "combined_visualization" / "layout_camera_frame.json", optimized)


def choose_ref_frame(config: dict, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    value = config.get("frame_number", 0)
    return int(value)


def choose_object_mesh(prepared_dir: Path, object_id: str, ref_frame: int) -> Path:
    preferred = prepared_dir / "video_segmentation" / "masks" / f"frame_{ref_frame:06d}_masks" / object_id / f"{object_id}.obj"
    if preferred.exists():
        return preferred
    candidates = sorted((prepared_dir / "video_segmentation" / "masks").glob(f"frame_*_masks/{object_id}/{object_id}.obj"))
    if not candidates:
        raise FileNotFoundError(f"missing object mesh under {prepared_dir / 'video_segmentation/masks'}")
    return candidates[0]


def run_scale_optimization(
    python_bin: str,
    prepared_dir: Path,
    object_id: str,
    anchor_hand: str,
    ref_frame: int,
    viz_dir: Path | None,
) -> None:
    script = REPO_ROOT / "third_party" / "do-as-i-do" / "reconstruction" / "scripts" / "optimize_translation_scale.py"
    require(script, "Do-as-I-Do optimize_translation_scale.py")
    layout_dir = prepared_dir / "obj_tracking_out" / object_id / "combined_visualization"
    cmd = [
        python_bin,
        str(script),
        "--layout-json",
        str(layout_dir / "layout_camera_frame.json"),
        "--mesh",
        str(choose_object_mesh(prepared_dir, object_id, ref_frame)),
        "--pointmap-dir",
        str(prepared_dir / "all_frames"),
        "--mask-dir",
        str(prepared_dir / "video_segmentation" / "masks"),
        "--mask-name",
        object_id,
        "--hand-meshes",
        str(prepared_dir / "raw" / "all_hand_meshes.npz"),
        "--anchor-hand",
        anchor_hand,
        "--ref-frame",
        str(ref_frame),
        "--output",
        str(layout_dir / "layout_camera_frame_optimized.json"),
        "--frames-dir",
        str(prepared_dir / "all_frames"),
    ]
    if viz_dir is not None:
        cmd += ["--viz-dir", str(viz_dir)]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an experiment-local Do-as-I-Do raw_dir from a reconstruction scene."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--hand-npz", type=Path)
    parser.add_argument("--fallback-gravity-json", type=Path)
    parser.add_argument("--anchor-hand", default="")
    parser.add_argument("--ref-frame", type=int)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--scale-viz-dir", type=Path)
    parser.add_argument("--no-optimize-scale", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require(source_dir / "config.json", "Do-as-I-Do config.json")
    require(source_dir / "raw.mp4", "Do-as-I-Do raw.mp4")
    require(source_dir / "all_frames", "Do-as-I-Do all_frames")
    require(source_dir / "video_segmentation", "Do-as-I-Do video_segmentation")

    if output_dir.exists() or output_dir.is_symlink():
        if not args.force:
            raise FileExistsError(f"output dir already exists: {output_dir}; pass --force to replace it")
        remove_existing(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(source_dir / "config.json")
    object_id = infer_object_id(source_dir, args.object_id)
    hand_npz = choose_hand_npz(source_dir, args.hand_npz)
    anchor_hand = args.anchor_hand or str(config.get("anchor_hand") or "left")
    ref_frame = choose_ref_frame(config, args.ref_frame)
    optimize_scale = not args.no_optimize_scale

    link_file((source_dir / "all_frames").resolve(), output_dir / "all_frames")
    link_file((source_dir / "raw.mp4").resolve(), output_dir / "raw.mp4")
    copy_tree(source_dir / "video_segmentation", output_dir / "video_segmentation")
    copy_file(source_dir / "config.json", output_dir / "config.json")

    gravity_src = source_dir / "gravity.json"
    if gravity_src.exists():
        copy_file(gravity_src, output_dir / "gravity.json")
    elif args.fallback_gravity_json is not None and args.fallback_gravity_json.exists():
        copy_file(args.fallback_gravity_json.resolve(), output_dir / "gravity.json")
    else:
        raise FileNotFoundError("missing gravity.json; pass --fallback-gravity-json")

    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    for path in sorted((source_dir / "raw").glob("*")):
        if path.name == "all_hand_meshes.npz":
            continue
        link_file(path.resolve(), output_dir / "raw" / path.name)

    hand_stats = write_adapted_hand_npz(hand_npz, output_dir / "raw" / "all_hand_meshes.npz")
    projection_qc = hand_projection_qc(output_dir / "raw" / "all_hand_meshes.npz", output_dir / "all_frames", anchor_hand)
    mask_stats = generate_hand_masks(
        output_dir / "raw" / "all_hand_meshes.npz",
        output_dir / "all_frames",
        output_dir / "video_segmentation" / "masks",
    )
    link_tracking_assets(source_dir, output_dir, object_id, optimize_scale)

    if optimize_scale:
        run_scale_optimization(
            args.python_bin,
            output_dir,
            object_id,
            anchor_hand,
            ref_frame,
            args.scale_viz_dir,
        )

    manifest = {
        "adapter": "prepare_do_as_i_do_scene_adapter.py",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "object_id": object_id,
        "anchor_hand": anchor_hand,
        "ref_frame": ref_frame,
        "optimize_scale": optimize_scale,
        "hand_npz": hand_stats,
        "source_frame_maps": [str(path) for path in sorted(source_dir.glob("frame_map*.json"))],
        "hand_projection_qc": projection_qc,
        "hand_masks": mask_stats,
        "layout": str(output_dir / "obj_tracking_out" / object_id / "combined_visualization" / "layout_camera_frame_optimized.json"),
    }
    (output_dir / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import trimesh
except Exception:  # pragma: no cover - optional dependency in some envs
    trimesh = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe_retarget_lab.matrix import (
    HAND_SOURCE_OPTIONS,
    RETARGETING_OPTIONS,
    TRAJECTORY_6DOF_OPTIONS,
    MatrixCell,
    all_matrix_cells,
    select_matrix_cells,
)
from aoe_retarget_lab.io_utils import (  # noqa: E402
    read_json_object_lenient as load_json_optional,
)
from aoe_retarget_lab.path_utils import (  # noqa: E402
    ensure_dir,
    optional_relative_path as rel,
    path_contains,
)
from aoe_retarget_lab.video_utils import probe_video  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = TRAJECTORY_6DOF_OPTIONS
HAND_SOURCES = HAND_SOURCE_OPTIONS
RETARGETERS = RETARGETING_OPTIONS


def cell_key(trajectory: str, hand_source: str, retargeting: str) -> str:
    return MatrixCell(trajectory, hand_source, retargeting).key


def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def materialize(src: Path | None, dst: Path, mode: str) -> Path | None:
    if src is None or not src.exists():
        remove_path(dst)
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        remove_path(dst)
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    elif mode == "hardlink" and src.is_file():
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    return dst


def egoinfinity_clip_dir(source: Path) -> Path:
    return first_existing([
        source / "intermediates" / "trajectory_6dof" / "egoinfinity" / "reconstruction" / "clip",
        source / "intermediates" / "egoinfinity" / "clip",
    ]) or source / "intermediates" / "trajectory_6dof" / "egoinfinity" / "reconstruction" / "clip"


def egoinfinity_mesh_from_manifest(manifest: dict, ego_clip: Path) -> Path | None:
    replacement = manifest.get("object_mesh_replacement") or {}
    ego_mesh = replacement.get("ego_mesh")
    if ego_mesh:
        path = Path(str(ego_mesh))
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.exists():
            return path
    ego_object_id = manifest.get("ego_object_id")
    if ego_object_id is not None:
        try:
            candidate = ego_clip / "sam3_meshes" / f"obj_{int(ego_object_id)}.ply"
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and candidate.exists():
            return candidate
    return None


def egoinfinity_visual_from_manifest(manifest: dict | None) -> Path | None:
    if manifest is None:
        return None
    replacement = manifest.get("object_mesh_replacement") or {}
    for item in replacement.get("targets") or []:
        path = Path(str(item))
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.exists():
            return path
    return None


def egoinfinity_adapter_quality(manifest: dict, mesh_path: Path | None) -> tuple[int, int]:
    if mesh_path is None:
        return (-10, 0)
    score = 0
    replacement = manifest.get("object_mesh_replacement") or {}
    object_name = " ".join(
        str(manifest.get(key) or "")
        for key in ("target_prompt", "ego_prompt", "dai_object_id")
    ).lower()
    if "box" in object_name:
        if replacement.get("boxlike_fallback"):
            score += 4
        stats = (replacement.get("stats") or [{}])[0]
        extents = np.asarray(stats.get("extents") or [], dtype=np.float64)
        if extents.size >= 3 and np.isfinite(extents[:3]).all() and extents[:3].max() > 1e-8:
            ratios = np.sort(extents[:3] / extents[:3].max())
            volumetric = min(1.0, float(ratios[1]) / 0.35) * min(1.0, float(ratios[0]) / 0.18)
            score += int(round(4.0 * volumetric))
            if float(ratios[0]) > 0.75 and float(ratios[1]) > 0.85 and not replacement.get("boxlike_fallback"):
                score -= 5
            if float(ratios[0]) < 0.16:
                score -= 2
    geom = manifest.get("geometry_interaction_hand") or {}
    geom_hand = geom.get("selected_hand")
    selected_hand = manifest.get("selected_hand")
    if selected_hand and geom_hand:
        if selected_hand == geom_hand or selected_hand == "bimanual":
            score += 3
        else:
            score -= 3
    if manifest.get("ego_object_id") is not None:
        score += 2
    if manifest.get("mesh_scale"):
        score += 1
    qc = manifest.get("object_selection_qc") or {}
    if qc.get("available") is True:
        score += 1
    return (score, 1)


def egoinfinity_object_binding_is_exact(manifest: dict) -> bool:
    return (
        manifest.get("object_geometry_source") == "ego"
        and manifest.get("object_track_source") == "egoinfinity"
        and manifest.get("object_mesh_source") == "egoinfinity"
        and manifest.get("retarget_object_source") == "egoinfinity"
    )


def adapter_route_binding(
    manifest_path: Path | None,
    *,
    trajectory: str,
    hand_source: str,
) -> dict[str, object]:
    manifest = load_json_optional(manifest_path)
    expected_object_source = "egoinfinity" if trajectory == "egoinfinity" else "dai_native"
    expected_geometry_source = "ego" if trajectory == "egoinfinity" else None
    errors: list[str] = []
    if not isinstance(manifest, dict):
        errors.append("missing_adapter_manifest")
        manifest = {}
    if manifest.get("hand_source") != hand_source:
        errors.append(
            f"hand_source={manifest.get('hand_source')!r}, expected {hand_source!r}"
        )
    for key in ("object_track_source", "object_mesh_source", "retarget_object_source"):
        if manifest.get(key) != expected_object_source:
            errors.append(
                f"{key}={manifest.get(key)!r}, expected {expected_object_source!r}"
            )
    if expected_geometry_source is not None and manifest.get("object_geometry_source") != expected_geometry_source:
        errors.append(
            f"object_geometry_source={manifest.get('object_geometry_source')!r}, "
            f"expected {expected_geometry_source!r}"
        )
    return {
        "status": "ok" if not errors else "invalid",
        "trajectory_6dof": trajectory,
        "hand_source": hand_source,
        "expected_object_source": expected_object_source,
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "actual": {
            key: manifest.get(key)
            for key in (
                "hand_source",
                "hand_geometry_source",
                "object_geometry_source",
                "object_track_source",
                "object_mesh_source",
                "retarget_object_source",
            )
        },
        "errors": errors,
    }


def load_latest_egoinfinity_adapter(source: Path, ego_clip: Path | None = None) -> dict | None:
    ego_clip = ego_clip or egoinfinity_clip_dir(source)
    manifests = sorted(
        (
            source
            / "intermediates"
            / "trajectory_6dof"
            / "egoinfinity"
        ).glob("do_as_i_do_raw_dir*/adapter_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    best: tuple[tuple[int, int, float], dict] | None = None
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not egoinfinity_object_binding_is_exact(manifest):
            continue
        manifest["_manifest_path"] = str(manifest_path)
        mesh_path = egoinfinity_mesh_from_manifest(manifest, ego_clip)
        manifest["_resolved_ego_mesh"] = str(mesh_path) if mesh_path is not None else None
        quality = egoinfinity_adapter_quality(manifest, mesh_path)
        rank = (quality[0], quality[1], manifest_path.stat().st_mtime)
        if best is None or rank > best[0]:
            best = (rank, manifest)
    return best[1] if best is not None else None


def selected_egoinfinity_mesh_ply(source: Path, ego_clip: Path) -> Path | None:
    manifest = load_latest_egoinfinity_adapter(source, ego_clip)
    if manifest is not None:
        mesh_path = egoinfinity_mesh_from_manifest(manifest, ego_clip)
        if mesh_path is not None:
            return mesh_path
    return first_existing([
        ego_clip / "sam3_meshes" / "obj_0.ply",
        ego_clip / "sam3_meshes" / "obj_1.ply",
    ])


def selected_egoinfinity_mesh_scale(source: Path, ego_clip: Path | None = None) -> float:
    manifest = load_latest_egoinfinity_adapter(source, ego_clip)
    if manifest is None:
        return 1.0
    try:
        scale = float(manifest.get("mesh_scale") or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_metadata() -> dict[str, object]:
    diff_hash = None
    try:
        proc = subprocess.run(["git", "diff", "--no-ext-diff", "--binary"], cwd=REPO_ROOT, check=False, capture_output=True)
        if proc.returncode == 0:
            diff_hash = hashlib.sha256(proc.stdout).hexdigest()
    except Exception:
        diff_hash = None
    status = run_text(["git", "status", "--short"])
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_diff_sha256": diff_hash,
        "status_short": status,
    }


def gpu_mapping_from_env() -> dict[str, str | None]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "MAIN_CUDA",
        "DAI_CUDA_VISIBLE_DEVICES",
        "SAM3_WORKER_CUDA",
        "SAM3D_WORKER_CUDA",
        "SPIDER_CUDA_VISIBLE_DEVICES",
        "SPIDER_DEVICE",
    ]
    return {key: os.environ.get(key) for key in keys}

def source_paths(
    source: Path,
    task: str,
    hand_type: str,
    dai_task: str,
    dai_hand_type: str,
    dai_source_task: str,
    dai_source_hand_type: str,
) -> dict[str, Path | None]:
    dai_key = cell_key("do_as_i_do", "estimated", "do_as_i_do")
    ego_dai_key = cell_key("egoinfinity", "estimated", "do_as_i_do")
    dai_root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / dai_key
        / "retargeting_outputs"
    )
    dai_robot = dai_root / "sharpa" / dai_source_hand_type / dai_source_task / "0"
    dai_mano = dai_root / "mano" / dai_source_hand_type / dai_source_task
    dai_obj = dai_root / "assets" / "objects" / dai_source_task
    ego_clip = egoinfinity_clip_dir(source)
    ego_dai_assets = source / "assets" / "cells" / ego_dai_key / "retargeting" / "do_as_i_do"
    ego_dai_root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / ego_dai_key
        / "retargeting_outputs"
    )
    ego_task_assets = source / "assets" / "trajectory_6dof" / "egoinfinity" / dai_task
    ego_display_assets = source / "assets" / "trajectory_6dof" / "egoinfinity" / task
    ego_selected_ply = selected_egoinfinity_mesh_ply(source, ego_clip)
    ego_selected_scale = selected_egoinfinity_mesh_scale(source, ego_clip)
    ego_adapter = load_latest_egoinfinity_adapter(source, ego_clip)
    ego_adapter_manifest = None
    ego_adapter_visual = egoinfinity_visual_from_manifest(ego_adapter)
    if ego_adapter is not None and ego_adapter.get("_manifest_path"):
        candidate_manifest = Path(str(ego_adapter["_manifest_path"]))
        if candidate_manifest.exists():
            ego_adapter_manifest = candidate_manifest
    ego_meshes = first_existing([
        ego_task_assets / "object_meshes",
        ego_display_assets / "object_meshes",
        ego_clip / "sam3_meshes",
        ego_clip / "sam3d_objects",
        ego_clip / "objects",
    ])
    return {
        "ego_overlay": first_existing([
            ego_clip / "rgb_mesh_overlay_selected.mp4",
            ego_clip / "rgb_mesh_overlay_fix02.mp4",
            ego_clip / "rgb_mesh_overlay.mp4",
            ego_clip / "retarget" / "g1" / "input_viz.mp4",
            ego_clip / "retarget_g1" / "input_viz.mp4",
        ]),
        "ego_depth": first_existing([
            ego_clip / "retarget_samples" / "depth.mp4",
            ego_clip / "retarget" / "g1" / "input_viz.mp4",
            ego_clip / "retarget_g1" / "input_viz.mp4",
        ]),
        "ego_robot": first_existing([ego_clip / "retarget" / "g1" / "robot_sim.mp4", ego_clip / "retarget_g1" / "robot_sim.mp4"]),
        "ego_pipeline": first_existing([
            ego_clip / "pipeline_result_selected.pkl.gz",
            ego_clip / "pipeline_result.pkl.gz",
        ]) or (ego_clip / "pipeline_result_selected.pkl.gz"),
        "ego_object_selection_report": first_existing([
            ego_clip / "object_selection_report.json",
            ego_clip / "object_filter_report.json",
        ]),
        "ego_meshes": ego_meshes,
        "ego_hands": first_existing([ego_clip / "retarget_samples" / "hand_joints.bin", ego_clip / "hand_joints.bin"]),
        "ego_spider_keypoints": first_existing([
            ego_dai_root / "mano" / dai_hand_type / dai_task / "0" / "trajectory_keypoints.npz",
            ego_dai_assets / "trajectories" / "trajectory_keypoints.npz",
        ]),
        "ego_spider_object_visual": None if ego_selected_ply is not None else first_existing([
            ego_adapter_visual,
            source / "assets" / "trajectory_6dof" / "egoinfinity" / dai_task / "object_meshes" / "visual.obj",
            source / "assets" / "trajectory_6dof" / "egoinfinity" / task / "object_meshes" / "visual.obj",
        ]),
        "ego_spider_adapter_visual": ego_adapter_visual,
        "ego_spider_object_ply": ego_selected_ply,
        "ego_spider_object_scale": ego_selected_scale,
        "ego_spider_object_convex": first_existing([
            source / "assets" / "trajectory_6dof" / "egoinfinity" / dai_task / "object_meshes" / "convex",
            source / "assets" / "trajectory_6dof" / "egoinfinity" / task / "object_meshes" / "convex",
        ]),
        "ego_adapter_manifest": ego_adapter_manifest,
        "dai_native_adapter_manifest": dai_adapter_manifest_for_robot(dai_robot),
        "dai_overlay": first_existing([
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "mesh_overlay.mp4",
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "overlay.mp4",
        ]),
        "dai_depth": first_existing([
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "depth.mp4",
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "moge_depth.mp4",
        ]),
        "dai_robot": first_existing([
            dai_robot / "visualization_mjwp_act_aligned.mp4",
        ]),
        "dai_robot_front": first_existing([
            dai_robot / "visualization_mjwp_act_aligned.mp4",
        ]),
        "dai_scene": dai_robot / "scene.xml",
        "dai_traj": first_existing([
            dai_robot / "trajectory_mjwp_aligned.npz",
            dai_robot / "trajectory_mjwp_act_aligned.npz",
        ]),
        "dai_keypoints": dai_mano / "0" / "trajectory_keypoints.npz",
        "dai_task_info": dai_mano / "task_info.json",
        "dai_object_visual": dai_obj / "visual.obj",
        "dai_object_convex": dai_obj / "convex",
    }


def load_ply_xyz(src: Path) -> np.ndarray | None:
    try:
        with src.open("rb") as handle:
            header_lines: list[str] = []
            while True:
                line = handle.readline()
                if not line:
                    return None
                text = line.decode("ascii", errors="replace").strip()
                header_lines.append(text)
                if text == "end_header":
                    break
            if not header_lines or header_lines[0] != "ply":
                return None
            fmt = next((line for line in header_lines if line.startswith("format ")), "")
            vertex_count = 0
            properties: list[str] = []
            in_vertex = False
            for line in header_lines:
                if line.startswith("element "):
                    parts = line.split()
                    in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                    if in_vertex:
                        vertex_count = int(parts[2])
                    continue
                if in_vertex and line.startswith("property "):
                    parts = line.split()
                    if len(parts) >= 3:
                        properties.append(parts[-1])
            if vertex_count <= 0 or not {"x", "y", "z"}.issubset(set(properties)):
                return None
            if "binary_little_endian" in fmt:
                dtype = np.dtype([(name, "<f4") for name in properties])
                vertices = np.frombuffer(handle.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
                xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
            elif "ascii" in fmt:
                x_idx = properties.index("x")
                y_idx = properties.index("y")
                z_idx = properties.index("z")
                rows = []
                for _ in range(vertex_count):
                    vals = handle.readline().decode("ascii", errors="replace").split()
                    if len(vals) >= len(properties):
                        rows.append([float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])])
                xyz = np.asarray(rows, dtype=np.float32)
            else:
                return None
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        return xyz if xyz.shape[0] >= 4 else None
    except Exception:
        return None


def write_bbox_obj_from_xyz(xyz: np.ndarray, src: Path, dst: Path, scale: float = 1.0) -> Path | None:
    if xyz.shape[0] < 8:
        return None
    xyz = np.asarray(xyz, dtype=np.float32) * float(scale)
    lo = np.percentile(xyz, 2.0, axis=0)
    hi = np.percentile(xyz, 98.0, axis=0)
    extent = np.maximum(hi - lo, 1e-4)
    pad = np.maximum(extent * 0.03, 1e-3)
    lo = lo - pad
    hi = hi + pad
    x0, y0, z0 = lo.tolist()
    x1, y1, z1 = hi.tolist()
    verts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out:
        out.write(f"# bbox converted from {src}\n")
        for x, y, z in verts:
            out.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            out.write(f"f {a} {b} {c}\n")
    return dst


def write_visual_obj_from_ply(src: Path | None, dst: Path, scale: float = 1.0) -> Path | None:
    if src is None or not src.exists():
        return None
    scale = float(scale) if np.isfinite(float(scale)) and float(scale) > 0 else 1.0
    try:
        xyz = load_ply_xyz(src)
        if xyz is None:
            return None
        if trimesh is not None:
            try:
                loaded = trimesh.load(src, process=False)
                if hasattr(loaded, "vertices") and len(getattr(loaded, "vertices", [])):
                    vertices = np.asarray(loaded.vertices, dtype=np.float64) * scale
                    faces = np.asarray(getattr(loaded, "faces", []), dtype=np.int64)
                    if faces.size:
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                    else:
                        mesh = trimesh.PointCloud(vertices).convex_hull
                elif hasattr(loaded, "geometry"):
                    chunks = []
                    face_chunks = []
                    offset = 0
                    for geom in loaded.geometry.values():
                        if not hasattr(geom, "vertices") or not len(geom.vertices):
                            continue
                        verts = np.asarray(geom.vertices, dtype=np.float64) * scale
                        chunks.append(verts)
                        geom_faces = np.asarray(getattr(geom, "faces", []), dtype=np.int64)
                        if geom_faces.size:
                            face_chunks.append(geom_faces + offset)
                        offset += len(verts)
                    if chunks:
                        vertices = np.concatenate(chunks, axis=0)
                        faces = np.concatenate(face_chunks, axis=0) if face_chunks else np.zeros((0, 3), dtype=np.int64)
                        mesh = (
                            trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                            if faces.size
                            else trimesh.PointCloud(vertices).convex_hull
                        )
                    else:
                        mesh = trimesh.PointCloud(xyz).convex_hull
                else:
                    mesh = trimesh.PointCloud(xyz).convex_hull
                dst.parent.mkdir(parents=True, exist_ok=True)
                mesh.export(dst)
                return dst
            except Exception as exc:
                print(f"[warn] trimesh EgoInfinity PLY->OBJ failed, falling back to bbox: {src}: {exc}")
        return write_bbox_obj_from_xyz(xyz, src, dst, scale)
    except Exception as exc:
        print(f"[warn] could not convert EgoInfinity PLY to OBJ for SPIDER: {src}: {exc}")
        return None


def write_visual_mesh_obj(src: Path | None, dst: Path, scale: float = 1.0) -> Path | None:
    if src is None or not src.exists():
        return None
    if src.suffix.lower() == ".ply":
        return write_visual_obj_from_ply(src, dst, scale)
    if trimesh is None:
        if src.suffix.lower() == ".obj" and float(scale) != 1.0:
            dst.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            with src.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("v "):
                        parts = line.rstrip("\n").split()
                        xyz = [float(parts[i]) * float(scale) for i in range(1, 4)]
                        suffix = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                        lines.append(f"v {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}{suffix}\n")
                    else:
                        lines.append(line)
            dst.write_text("".join(lines), encoding="utf-8")
            return dst
        return materialize(src, dst, "copy")
    try:
        loaded = trimesh.load(src, process=False)
        if hasattr(loaded, "geometry"):
            loaded = trimesh.util.concatenate([geom for geom in loaded.geometry.values() if hasattr(geom, "vertices")])
        if not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
            return materialize(src, dst, "copy")
        mesh = loaded.copy()
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(dst)
        return dst
    except Exception as exc:
        print(f"[warn] could not convert visual mesh to OBJ for SPIDER: {src}: {exc}", file=sys.stderr)
        return materialize(src, dst, "copy")


def write_dai_hand_ego_object_keypoints(dai_src: Path | None, ego_src: Path | None, dst: Path) -> Path | None:
    if dai_src is None or ego_src is None or not dai_src.exists() or not ego_src.exists():
        return None
    try:
        dai = np.load(dai_src, allow_pickle=True)
        ego = np.load(ego_src, allow_pickle=True)
        arrays = {key: dai[key] for key in dai.files}
        replaced: list[str] = []
        for side in ("left", "right"):
            object_key = f"qpos_obj_{side}"
            if object_key in arrays and object_key in ego and np.asarray(arrays[object_key]).shape == np.asarray(ego[object_key]).shape:
                arrays[object_key] = np.asarray(ego[object_key])
                replaced.append(object_key)
            contact_key = f"contact_{side}"
            if contact_key in arrays:
                arrays[contact_key] = np.zeros_like(np.asarray(arrays[contact_key]))
            contact_pos_key = f"contact_pos_{side}"
            if contact_pos_key in arrays:
                arrays[contact_pos_key] = np.zeros_like(np.asarray(arrays[contact_pos_key]))
        if not replaced:
            return None
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **arrays)
        return dst
    except Exception as exc:
        print(f"[warn] failed to fuse DAI hand with Ego object keypoints: {exc}", file=sys.stderr)
        return None


def summarize_keypoint_quality(path: Path | None) -> dict[str, object]:
    summary: dict[str, object] = {"available": False}
    if path is None or not path.exists():
        summary["reason"] = "missing"
        return summary
    try:
        with np.load(path, allow_pickle=False) as data:
            sides: dict[str, dict[str, object]] = {}
            best_side = None
            best_min = float("inf")
            for side in ("left", "right"):
                finger_key = f"qpos_finger_{side}"
                object_key = f"qpos_obj_{side}"
                if finger_key not in data or object_key not in data:
                    continue
                fingers = np.asarray(data[finger_key], dtype=np.float64)
                obj = np.asarray(data[object_key], dtype=np.float64)
                if fingers.size == 0 or obj.size == 0 or fingers.shape[0] == 0 or obj.shape[0] == 0:
                    continue
                frames = min(fingers.shape[0], obj.shape[0])
                tips = fingers[:frames, :, :3].reshape(frames, -1, 3)
                centers = obj[:frames, :3]
                if not np.isfinite(tips).any() or not np.isfinite(centers).any():
                    continue
                dist = np.linalg.norm(tips - centers[:, None, :], axis=-1)
                if float(np.nanmax(np.abs(tips))) < 1e-8 and float(np.nanmax(np.abs(centers))) < 1e-8:
                    continue
                contact_key = f"contact_{side}"
                contact_active = 0
                contact_mean = 0.0
                if contact_key in data:
                    contact = np.asarray(data[contact_key], dtype=np.float64)
                    contact_active = int(np.count_nonzero(contact > 0.5))
                    contact_mean = float(np.nanmean(contact)) if contact.size else 0.0
                wrist_key = f"qpos_wrist_{side}"
                wrist_motion = None
                if wrist_key in data:
                    wrist = np.asarray(data[wrist_key], dtype=np.float64)
                    if wrist.size:
                        wrist_motion = float(np.nanmax(np.linalg.norm(wrist[:, :3] - wrist[:1, :3], axis=-1)))
                min_dist = float(np.nanmin(dist))
                sides[side] = {
                    "frames": int(frames),
                    "finger_object_distance_median": float(np.nanmedian(dist)),
                    "finger_object_distance_min": min_dist,
                    "finger_object_distance_p10": float(np.nanpercentile(dist, 10)),
                    "contact_active": contact_active,
                    "contact_mean": contact_mean,
                    "wrist_motion": wrist_motion,
                }
                if min_dist < best_min:
                    best_min = min_dist
                    best_side = side
            summary.update({"available": bool(sides), "sides": sides, "best_side": best_side})
            if not sides:
                summary["reason"] = "no_hand_object_keys"
            return summary
    except Exception as exc:
        return {"available": False, "reason": f"load_failed: {exc}"}


def quality_for_hand(summary: dict[str, object], hand_type: str) -> dict[str, object]:
    sides = summary.get("sides")
    if not isinstance(sides, dict):
        return {}
    if hand_type in sides and isinstance(sides[hand_type], dict):
        return sides[hand_type]
    best_side = summary.get("best_side")
    if isinstance(best_side, str) and best_side in sides and isinstance(sides[best_side], dict):
        return sides[best_side]
    return {}


def spider_single_object_hand_selection(
    requested_hand_type: str,
    source_hand_type: str,
    quality: dict[str, object],
    adapter_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    if requested_hand_type in {"left", "right", "bimanual"}:
        return {
            "requested_hand_type": requested_hand_type,
            "resolved_hand_type": requested_hand_type,
            "reason": "explicit_spider_hand_type",
            "quality": quality,
        }
    if source_hand_type in {"left", "right"}:
        return {
            "requested_hand_type": requested_hand_type,
            "resolved_hand_type": source_hand_type,
            "reason": "explicit_source_task_side",
            "quality": quality,
        }
    adapter_candidates: list[tuple[str, object]] = []
    if isinstance(adapter_manifest, dict):
        for key in ("surface_contact_geometry", "geometry_interaction_hand"):
            nested = adapter_manifest.get(key)
            if isinstance(nested, dict):
                adapter_candidates.append((f"{key}.selected_hand", nested.get("selected_hand")))
        for key in ("anchor_hand", "selected_hand", "interaction_hand"):
            adapter_candidates.append((key, adapter_manifest.get(key)))
    for evidence, value in adapter_candidates:
        if value in {"left", "right"}:
            return {
                "requested_hand_type": requested_hand_type,
                "resolved_hand_type": value,
                "reason": "adapter_manifest_interaction_hand",
                "adapter_evidence": evidence,
                "quality": quality,
            }
    sides = quality.get("sides") if isinstance(quality, dict) else None
    if not isinstance(sides, dict) or not sides:
        return {
            "requested_hand_type": requested_hand_type,
            "resolved_hand_type": "auto",
            "reason": "insufficient_keypoint_quality",
            "quality": quality,
        }

    def rank(side: str) -> tuple[float, float, float, float, float]:
        row = sides.get(side) if isinstance(sides.get(side), dict) else {}
        contact_active = float(row.get("contact_active", 0) or 0)
        contact_mean = float(row.get("contact_mean", 0.0) or 0.0)
        median = float(row.get("finger_object_distance_median", float("inf")))
        minimum = float(row.get("finger_object_distance_min", float("inf")))
        return (
            1.0 if contact_active > 0 else 0.0,
            contact_active,
            contact_mean,
            -median,
            -minimum,
        )

    available = [side for side in ("left", "right") if side in sides]
    selected = max(available, key=rank)
    return {
        "requested_hand_type": requested_hand_type,
        "resolved_hand_type": selected,
        "reason": "single_object_best_interaction_hand",
        "quality": quality,
        "ranks": {side: list(rank(side)) for side in available},
    }


def dai_interaction_hand_types(hand_type: str, *, auto_select: bool) -> list[str]:
    hand_types = [hand_type]
    if auto_select:
        for candidate in ("left", "right", "bimanual"):
            if candidate and candidate not in hand_types:
                hand_types.append(candidate)
    return hand_types


def dai_quality_score(
    quality: dict[str, object],
    hand_type: str,
    requested_hand_type: str,
    *,
    trajectory: str,
) -> tuple[float, float, float, float, float]:
    hand_quality = quality_for_hand(quality, hand_type)
    median = float(hand_quality.get("finger_object_distance_median", float("inf")))
    min_dist = float(hand_quality.get("finger_object_distance_min", float("inf")))
    contact_active = int(hand_quality.get("contact_active", 0))
    return (
        1.0 if contact_active > 0 else 0.0,
        float(contact_active),
        -median - 0.1 * min_dist,
        1.0 if trajectory == "do_as_i_do" else 0.0,
        1.0 if hand_type == requested_hand_type else 0.0,
    )


def find_dai_review_video(robot_root: Path) -> Path | None:
    for video in [
        robot_root / "visualization_mjwp_act_aligned.mp4",
        robot_root / "visualization_mjwp_aligned.mp4",
        robot_root / "visualization_mjwp_act.mp4",
        robot_root / "visualization_mjwp.mp4",
    ]:
        if probe_video(video) is not None:
            return video
    return None


def find_dai_robot(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
) -> Path | None:
    key = cell_key(trajectory, hand_source, "do_as_i_do")
    root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / key
        / "retargeting_outputs"
        / "sharpa"
        / hand_type
        / task
        / "0"
    )
    return find_dai_review_video(root)


def find_dai_robot_candidates(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    hand_types = dai_interaction_hand_types(
        hand_type,
        auto_select=bool(args.dai_auto_select_interaction_hand),
    )
    hand_sources = [hand_source]
    if hand_source != "estimated" and args.allow_dai_hand_fallback:
        hand_sources.append("estimated")

    for search_hand_source in hand_sources:
        for search_hand_type in hand_types:
            bundles = find_dai_asset_bundles(
                source,
                trajectory,
                search_hand_source,
                task,
                search_hand_type,
                allow_glob=False,
            )
            if not bundles:
                continue
            for bundle in bundles:
                route_binding = adapter_route_binding(
                    Path(str(bundle["adapter_manifest"])),
                    trajectory=trajectory,
                    hand_source=search_hand_source,
                )
                if route_binding["status"] != "ok":
                    continue
                robot = find_dai_robot(
                    source,
                    trajectory,
                    search_hand_source,
                    str(bundle["task"]),
                    search_hand_type,
                )
                if robot is None:
                    continue
                quality = summarize_keypoint_quality(Path(str(bundle["keypoints"])))
                retarget_quality = load_json_optional(Path(str(bundle["retarget_quality"])))
                valid = probe_video(robot) is not None
                candidates.append({
                    "robot": robot,
                    "task": str(bundle["task"]),
                    "trajectory": trajectory,
                    "hand_source": search_hand_source,
                    "hand_type": search_hand_type,
                    "quality": quality,
                    "review_video_valid": valid,
                    "retarget_quality": retarget_quality,
                    "route_binding": route_binding,
                    "quality_diagnostics_are_advisory": True,
                    "score": dai_quality_score(
                        quality,
                        search_hand_type,
                        hand_type,
                        trajectory=trajectory,
                    ),
                })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def find_dai_robot_in_overrides(
    overrides: list[Path],
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    args: argparse.Namespace,
) -> tuple[Path | None, str | None]:
    all_candidates: list[dict[str, object]] = []
    search_specs: list[tuple[str, str, str]] = [(trajectory, task, hand_type)]
    if args.allow_cross_trajectory_dai_robot_fallback:
        alt_trajectory = "egoinfinity" if trajectory == "do_as_i_do" else "do_as_i_do"
        alt_task = args.dai_task if alt_trajectory == "egoinfinity" else args.dai_source_task
        alt_hand_type = args.dai_hand_type if alt_trajectory == "egoinfinity" else args.dai_source_hand_type
        spec = (alt_trajectory, alt_task, alt_hand_type)
        if spec not in search_specs:
            search_specs.append(spec)
    for override in overrides:
        for search_trajectory, search_task, search_hand_type in search_specs:
            candidates = find_dai_robot_candidates(
                override,
                search_trajectory,
                hand_source,
                search_task,
                search_hand_type,
                args,
            )
            for item in candidates:
                item = dict(item)
                item["override_name"] = override.name
                item["requested_trajectory"] = trajectory
                item["is_requested_trajectory"] = search_trajectory == trajectory
                all_candidates.append(item)
    if all_candidates:
        requested_exact = [
            item
            for item in all_candidates
            if item.get("is_requested_trajectory") and item.get("hand_source") == hand_source
        ]
        fallback_candidates = [item for item in all_candidates if item not in requested_exact]
        candidate_pool = requested_exact or fallback_candidates
        ranked_candidates = [item for item in candidate_pool if item.get("review_video_valid")]
        if not ranked_candidates:
            return None, None
        ranked_candidates.sort(key=lambda item: item["score"], reverse=True)
        item = ranked_candidates[0]
        detail = "exact" if item["hand_source"] == hand_source else "estimated-hand fallback"
        trajectory_detail = (
            f"{item['trajectory']} trajectory"
            if item["trajectory"] == item.get("requested_trajectory")
            else f"{item['trajectory']} trajectory fallback for {item.get('requested_trajectory')}"
        )
        hand_detail = (
            f"{item['hand_type']} interaction-hand"
            if item["hand_type"] != hand_type
            else f"{hand_type} hand"
        )
        valid_detail = "decodable review video"
        return (
            Path(str(item["robot"])),
            (
                f"{detail} {trajectory_detail} {hand_detail} task {item['task']} Do-as-I-Do/Sharpa cell "
                f"from override run {item['override_name']} ({valid_detail})"
            ),
        )
    return None, None


def find_quality_dai_robot_in_source(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    args: argparse.Namespace,
) -> tuple[Path | None, str | None]:
    candidates = find_dai_robot_candidates(
        source,
        trajectory,
        hand_source,
        task,
        hand_type,
        args,
    )
    if not candidates:
        return None, None
    exact_candidates = [item for item in candidates if item.get("hand_source") == hand_source]
    fallback_candidates = [item for item in candidates if item.get("hand_source") != hand_source]
    candidate_pool = exact_candidates or fallback_candidates
    ranked_candidates = [item for item in candidate_pool if item.get("review_video_valid")]
    if not ranked_candidates:
        return None, None
    ranked_candidates.sort(key=lambda item: item["score"], reverse=True)
    item = ranked_candidates[0]
    hand_detail = (
        f"{item['hand_type']} interaction-hand"
        if item["hand_type"] != hand_type
        else f"{hand_type} hand"
    )
    source_detail = "exact" if item["hand_source"] == hand_source else f"{item['hand_source']} hand-source fallback"
    valid_detail = "decodable review video"
    return (
        Path(str(item["robot"])),
        (
            f"{source_detail} source {trajectory} trajectory {hand_detail} task {item['task']} "
            f"Do-as-I-Do/Sharpa cell ({valid_detail})"
        ),
    )


def find_exact_spider_input_bundle(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    args: argparse.Namespace,
) -> dict[str, Path | str] | None:
    candidates: list[dict[str, object]] = []
    for search_hand_type in dai_interaction_hand_types(
        hand_type,
        auto_select=bool(args.dai_auto_select_interaction_hand),
    ):
        for bundle in find_dai_asset_bundles(
            source,
            trajectory,
            hand_source,
            task,
            search_hand_type,
            allow_glob=False,
        ):
            route_binding = adapter_route_binding(
                Path(str(bundle["adapter_manifest"])),
                trajectory=trajectory,
                hand_source=hand_source,
            )
            if route_binding["status"] != "ok":
                continue
            quality = summarize_keypoint_quality(Path(str(bundle["keypoints"])))
            candidates.append({
                "bundle": bundle,
                "route_binding": route_binding,
                "score": dai_quality_score(
                    quality,
                    search_hand_type,
                    hand_type,
                    trajectory=trajectory,
                ),
            })
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0]["bundle"]  # type: ignore[return-value]


def stage_exact_spider_input_bundle(
    bundle: dict[str, Path | str],
    *,
    output_root: Path,
    resolved_hand_type: str,
    data_id: int,
) -> dict[str, object]:
    """Materialize one exact, self-contained SPIDER input without symlinks.

    DAI may emit a bimanual task while the single-object SPIDER route selects
    one interaction hand. The source keypoints and object geometry stay byte
    identical; only the experiment-local task-info binding is rewritten.
    """
    if resolved_hand_type not in {"left", "right", "bimanual"}:
        raise ValueError(f"invalid resolved SPIDER hand type: {resolved_hand_type}")

    source_keypoints = Path(str(bundle["keypoints"])).resolve(strict=True)
    source_task_info = Path(str(bundle["task_info"])).resolve(strict=True)
    source_visual = Path(str(bundle["object_visual"])).resolve(strict=True)
    source_convex = Path(str(bundle["object_convex"])).resolve(strict=True)
    task = str(bundle["task"])
    source_payload = json.loads(source_task_info.read_text(encoding="utf-8"))

    remove_path(output_root)
    object_target = output_root / "assets" / "objects" / task
    shutil.copytree(source_visual.parent, object_target, symlinks=False)

    target_task_dir = output_root / "mano" / resolved_hand_type / task
    target_keypoints = target_task_dir / str(data_id) / "trajectory_keypoints.npz"
    target_keypoints.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_keypoints, target_keypoints)

    payload = dict(source_payload)
    payload.update(
        {
            "task": task,
            "dataset_name": "do_as_i_do",
            "robot_type": "mano",
            "embodiment_type": resolved_hand_type,
            "data_id": int(data_id),
        }
    )
    object_mesh_dir = f"assets/objects/{task}"
    object_convex_dir = f"{object_mesh_dir}/convex"
    if resolved_hand_type == "bimanual":
        # Preserve the source-side assignments for a true bimanual task.
        pass
    else:
        other = "right" if resolved_hand_type == "left" else "left"
        payload[f"{resolved_hand_type}_object_mesh_dir"] = object_mesh_dir
        payload[f"{resolved_hand_type}_object_convex_dir"] = object_convex_dir
        payload[f"{other}_object_mesh_dir"] = None
        payload[f"{other}_object_convex_dir"] = None
    target_task_info = target_task_dir / "task_info.json"
    target_task_info.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if any(path.is_symlink() for path in output_root.rglob("*")):
        raise RuntimeError("SPIDER staged input unexpectedly contains a symlink")
    source_keypoints_sha = sha256_file(source_keypoints)
    target_keypoints_sha = sha256_file(target_keypoints)
    if source_keypoints_sha != target_keypoints_sha:
        raise RuntimeError("SPIDER staged keypoints are not byte-identical")
    if not (object_target / source_visual.name).is_file():
        raise RuntimeError("SPIDER staged object visual is missing")
    staged_convex = object_target / source_convex.name
    if not staged_convex.exists():
        raise RuntimeError("SPIDER staged object convex geometry is missing")

    manifest = {
        "schema_version": 1,
        "scope": "retarget_lab_spider_input_binding",
        "source_hand_type": str(bundle["hand_type"]),
        "resolved_hand_type": resolved_hand_type,
        "task": task,
        "data_id": int(data_id),
        "source_task_info": str(source_task_info),
        "staged_task_info": str(target_task_info),
        "task_info_rewritten": True,
        "backend_parameters_modified": False,
        "trajectory_keypoints": {
            "source": str(source_keypoints),
            "staged": str(target_keypoints),
            "source_sha256": source_keypoints_sha,
            "staged_sha256": target_keypoints_sha,
            "byte_identical": True,
        },
        "object_assets": {
            "source": str(source_visual.parent),
            "staged": str(object_target),
            "visual_sha256": sha256_file(object_target / source_visual.name),
        },
    }
    write_json(output_root / "spider_input_staging_manifest.json", manifest)
    return {
        "input_root": output_root,
        "task_info": target_task_info,
        "keypoints": target_keypoints,
        "manifest": manifest,
        "manifest_path": output_root / "spider_input_staging_manifest.json",
    }


def find_dai_asset_bundles(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str | None,
    hand_type: str,
    allow_glob: bool = False,
) -> list[dict[str, Path | str]]:
    key = cell_key(trajectory, hand_source, "do_as_i_do")
    root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / key
        / "retargeting_outputs"
    )
    task_names = [task] if task else []
    if allow_glob:
        mano_root = root / "mano" / hand_type
        task_names += sorted(p.parents[1].name for p in mano_root.glob("*/0/trajectory_keypoints.npz"))

    seen: set[str] = set()
    bundles: list[dict[str, Path | str]] = []
    for task_name in task_names:
        if not task_name or task_name in seen:
            continue
        seen.add(task_name)
        keypoints = root / "mano" / hand_type / task_name / "0" / "trajectory_keypoints.npz"
        if not keypoints.exists():
            continue
        bundles.append({
            "task": task_name,
            "trajectory": trajectory,
            "hand_source": hand_source,
            "hand_type": hand_type,
            "keypoints": keypoints,
            "task_info": root / "mano" / hand_type / task_name / "task_info.json",
            "object_visual": root / "assets" / "objects" / task_name / "visual.obj",
            "object_convex": root / "assets" / "objects" / task_name / "convex",
            "robot_root": root / "sharpa" / hand_type / task_name / "0",
            "retarget_quality": root / "sharpa" / hand_type / task_name / "0" / "mjwp_object_tracking_quality.json",
            "adapter_manifest": root.parent / "raw_dir" / "adapter_manifest.json",
        })
    return bundles


def apply_dai_override_trajectory_assets(
    srcs: dict[str, Path | None],
    overrides: list[Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    report: dict[str, object] = {}
    candidates: list[dict[str, object]] = []
    for override in overrides:
        for trajectory, task, requested_hand_type in (
            ("do_as_i_do", args.dai_source_task, args.dai_source_hand_type),
            ("egoinfinity", args.dai_task, args.dai_hand_type),
        ):
            for hand_type in dai_interaction_hand_types(
                requested_hand_type,
                auto_select=bool(args.dai_auto_select_interaction_hand),
            ):
                for bundle in find_dai_asset_bundles(
                    override,
                    trajectory,
                    "estimated",
                    task,
                    hand_type,
                    allow_glob=True,
                ):
                    quality = summarize_keypoint_quality(Path(str(bundle["keypoints"])))
                    retarget_quality = load_json_optional(Path(str(bundle["retarget_quality"])))
                    candidates.append({
                        "override": override,
                        "bundle": bundle,
                        "quality": quality,
                        "retarget_quality": retarget_quality,
                        "score": dai_quality_score(
                            quality,
                            hand_type,
                            requested_hand_type,
                            trajectory=trajectory,
                        ),
                    })
    if candidates:
        candidates.sort(key=lambda item: item["score"], reverse=True)
        report["dai_asset_candidates"] = [
            {
                "source": Path(str(item["override"])).name,
                "trajectory": str(item["bundle"]["trajectory"]),
                "hand_source": str(item["bundle"]["hand_source"]),
                "hand_type": str(item["bundle"]["hand_type"]),
                "task": str(item["bundle"]["task"]),
                "keypoints": str(item["bundle"]["keypoints"]),
                "retarget_quality": item.get("retarget_quality"),
                "quality": item["quality"],
                "numerical_diagnostics_are_advisory": True,
            }
            for item in candidates
        ]
    if args.dai_override_ego_spider_keypoints:
        ego_candidates = [
            item
            for item in candidates
            if str(item["bundle"]["trajectory"]) == "egoinfinity"
        ]
        if ego_candidates:
            item = ego_candidates[0]
            bundle = item["bundle"]
            srcs["ego_spider_keypoints"] = bundle["keypoints"]
            report["ego_spider_keypoints"] = str(bundle["keypoints"])
            report["ego_spider_keypoints_source"] = Path(str(item["override"])).name
            report["ego_spider_keypoints_hand_type"] = str(bundle["hand_type"])
            report["ego_spider_keypoints_task"] = str(bundle["task"])
            report["ego_spider_keypoints_quality"] = item["quality"]
        else:
            report["ego_spider_keypoints"] = None
            report["ego_spider_keypoints_reason"] = "no_egoinfinity_override_bundle"
    for item in candidates:
        bundle = item["bundle"]
        if str(bundle["trajectory"]) != "do_as_i_do":
            continue
        srcs["dai_keypoints"] = bundle["keypoints"]
        srcs["dai_task_info"] = bundle["task_info"]
        srcs["dai_object_visual"] = bundle["object_visual"]
        srcs["dai_object_convex"] = bundle["object_convex"]
        report["dai_native_source"] = Path(str(item["override"])).name
        report["dai_native_trajectory"] = str(bundle["trajectory"])
        report["dai_native_hand_source"] = str(bundle["hand_source"])
        report["dai_native_hand_type"] = str(bundle["hand_type"])
        report["dai_native_task"] = str(bundle["task"])
        report["dai_native_keypoints"] = str(bundle["keypoints"])
        report["dai_native_quality"] = item["quality"]
        report["dai_native_retarget_quality"] = item.get("retarget_quality")
        break
    else:
        if candidates:
            report["dai_native_override_reason"] = "no_do_as_i_do_override_bundle"
    return report


def prepare_trajectory_assets(
    dst: Path,
    srcs: dict[str, Path | None],
    task: str,
    mode: str,
    *,
    trajectories: tuple[str, ...] = TRAJECTORIES,
    spider_fuse_ego_object_for_dai: bool = False,
) -> dict:
    report: dict[str, dict[str, str | None]] = {}
    for trajectory in trajectories:
        root = ensure_dir(dst / "assets" / "trajectory_6dof" / trajectory / task)
        if trajectory == "egoinfinity":
            provenance = {
                "object_track_source": "egoinfinity",
                "object_mesh_source": "egoinfinity",
                "retarget_object_source": "egoinfinity",
                "display_keypoints_hand_source": "estimated",
                "object_selection_source": "bbox_target_point_best",
                "recommended_route": True,
            }
            ego_visual = write_visual_mesh_obj(
                srcs.get("ego_spider_adapter_visual"),
                root / "object_meshes" / "visual.obj",
                float(srcs.get("ego_spider_object_scale") or 1.0),
            ) or write_visual_obj_from_ply(
                srcs["ego_spider_object_ply"],
                root / "object_meshes" / "visual.obj",
                float(srcs.get("ego_spider_object_scale") or 1.0),
            )
            entries = {
                "overlay": materialize(srcs["ego_overlay"], root / "overlay.mp4", mode),
                "depth": materialize(srcs["ego_depth"], root / "depth.mp4", mode),
                "pipeline_result": materialize(srcs["ego_pipeline"], root / "object_6dof_native.pkl.gz", mode),
                "object_selection_report": materialize(
                    srcs.get("ego_object_selection_report"),
                    root / "object_selection_report.json",
                    "copy",
                ),
                "native_object_meshes": materialize(srcs["ego_meshes"], root / "native_object_meshes", mode),
                "estimated_hands": materialize(srcs["ego_hands"], root / "estimated_hands", mode),
                "source_keypoints": materialize(srcs["ego_spider_keypoints"], root / "source_trajectory_keypoints.npz", mode),
                "object_visual": ego_visual
                or materialize(srcs["ego_spider_object_visual"], root / "object_meshes" / "visual.obj", mode),
                "object_convex": materialize(srcs["ego_spider_object_convex"], root / "object_meshes" / "convex", mode),
                "adapter_manifest": materialize(
                    srcs.get("ego_adapter_manifest"),
                    root / "egoinfinity_adapter_manifest.json",
                    "copy",
                ),
            }
        else:
            fused_keypoints = None
            fused_visual = None
            repaired_visual = None
            repaired_visual_manifest = None
            provenance = {
                "object_track_source": "dai_native",
                "object_mesh_source": "dai_native",
                "retarget_object_source": "dai_native",
                "display_keypoints_hand_source": "estimated",
                "recommended_route": False,
                "baseline_only": True,
            }
            native_adapter = load_json_optional(srcs.get("dai_native_adapter_manifest"))
            if native_adapter is not None:
                provenance["retarget_object_source"] = str(
                    native_adapter.get("retarget_object_source") or "dai_native"
                )
                provenance["hoi_contact_alignment"] = native_adapter.get("hoi_contact_alignment")
            if spider_fuse_ego_object_for_dai:
                provenance = {
                    "object_track_source": "fused_ego_for_spider",
                    "object_mesh_source": "egoinfinity",
                    "retarget_object_source": "fused_ego_for_spider",
                    "recommended_route": False,
                    "baseline_only": False,
                    "explicit_fusion": True,
                }
                fused_keypoints = write_dai_hand_ego_object_keypoints(
                    srcs["dai_keypoints"],
                    srcs["ego_spider_keypoints"],
                    root / "source_trajectory_keypoints.npz",
                )
                fused_visual = write_visual_obj_from_ply(
                    srcs["ego_spider_object_ply"],
                    root / "object_meshes" / "visual.obj",
                    float(srcs.get("ego_spider_object_scale") or 1.0),
                )
            entries = {
                "overlay": materialize(srcs["dai_overlay"], root / "overlay.mp4", mode),
                "depth": materialize(srcs["dai_depth"], root / "depth.mp4", mode),
                "source_keypoints": fused_keypoints
                or materialize(srcs["dai_keypoints"], root / "source_trajectory_keypoints.npz", mode),
                "task_info": materialize(srcs["dai_task_info"], root / "task_info.json", mode),
                "object_visual": fused_visual
                or repaired_visual
                or materialize(srcs["dai_object_visual"], root / "object_meshes" / "visual.obj", mode),
                "object_convex": materialize(
                    srcs["ego_spider_object_convex"] if spider_fuse_ego_object_for_dai else None,
                    root / "object_meshes" / "convex",
                    mode,
                )
                or materialize(srcs["dai_object_convex"], root / "object_meshes" / "convex", mode),
                "spider_fused_ego_object": fused_visual,
                "spider_repaired_object_visual": repaired_visual,
                "adapter_manifest": materialize(
                    srcs.get("dai_native_adapter_manifest"),
                    root / "dai_native_adapter_manifest.json",
                    "copy",
                ),
            }
            if repaired_visual_manifest is not None:
                write_json(root / "dai_spider_object_visual_repair.json", repaired_visual_manifest)
                entries["spider_repaired_object_visual_manifest"] = root / "dai_spider_object_visual_repair.json"
        entry_report = {k: rel(v, dst) for k, v in entries.items()}
        entry_report["provenance"] = provenance
        write_json(root / "reuse_source_manifest.json", entry_report)
        report[trajectory] = entry_report
    return report


def resolve_spider_input_timebase(
    input_root: Path,
    *,
    task: str,
    hand_type: str,
    data_id: int,
    adapter_manifest: Path | None,
) -> dict[str, object]:
    """Resolve the backend timestep from available input metadata."""
    warnings: list[str] = []
    evidence: list[dict[str, object]] = []
    task_info_path = input_root / "mano" / hand_type / task / "task_info.json"
    task_info = load_json_optional(task_info_path)
    if isinstance(task_info, dict):
        try:
            value = float(task_info["ref_dt"])
            if value > 0:
                evidence.append({"source": str(task_info_path), "ref_dt": value})
        except (KeyError, TypeError, ValueError):
            warnings.append("spider_task_info_ref_dt_invalid")
    adapter = load_json_optional(adapter_manifest)
    clock = (
        ((adapter or {}).get("temporal_alignment") or {}).get("target_clock") or {}
        if isinstance(adapter, dict)
        else {}
    )
    try:
        fps = float(clock["fps"])
        if fps > 0:
            evidence.append({"source": str(adapter_manifest), "ref_dt": 1.0 / fps})
    except (KeyError, TypeError, ValueError):
        warnings.append("spider_adapter_target_clock_invalid")
    values = [float(item["ref_dt"]) for item in evidence]
    if not values:
        ref_dt = 1.0 / 15.0
        warnings.append("spider_input_timebase_missing_using_15fps_default")
    else:
        ref_dt = values[0]
        if any(abs(value - ref_dt) > 1e-12 for value in values[1:]):
            warnings.append("spider_input_timebase_metadata_disagrees_using_task_info_priority")
    return {
        "status": "ok",
        "ref_dt": ref_dt,
        "task": task,
        "hand_type": hand_type,
        "data_id": data_id,
        "evidence": evidence,
        "warnings": warnings,
        "backend_tuning": False,
    }


def _native_spider_binding(route_root: Path) -> dict | None:
    if (route_root / "spider_output_manifest.json").exists():
        return None
    return load_json_optional(route_root / "native_spider_run_binding.json")


def write_native_spider_run_binding(
    *,
    route_root: Path,
    route: str,
    trajectory_6dof: str,
    hand_source: str,
    hand_type: str,
    task: str,
    robot_type: str,
    data_id: int,
    ref_dt_evidence: dict[str, object],
    input_root: Path,
    native_run_dir: Path,
    input_record_path: Path,
    log_path: Path,
    returncode: int,
) -> dict[str, object]:
    run_manifest_path = native_run_dir / "vanilla_spider_run_manifest.json"
    outcome_path = native_run_dir / "vanilla_spider_outcome.json"
    run_manifest = load_json_optional(run_manifest_path) or {}
    outcome = load_json_optional(outcome_path) or {}
    native_rc = int(outcome.get("returncode", returncode))
    videos = sorted(native_run_dir.rglob("visualization_mjwp.mp4"))
    robot = videos[0] if len(videos) == 1 else None
    native_log = log_path if log_path.is_file() else None
    metrics: dict[str, float] = {}
    if native_log is not None:
        match = re.search(
            r"Final object tracking error:\s*pos=([0-9.eE+-]+),\s*quat=([0-9.eE+-]+)",
            native_log.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            metrics = {
                "final_object_position_error_m": float(match.group(1)),
                "final_object_quaternion_error": float(match.group(2)),
            }
    payload: dict[str, object] = {
        "schema_version": 1,
        "route": route,
        "trajectory_6dof": trajectory_6dof,
        "hand_source": hand_source,
        "resolved_hand_type": hand_type,
        "resolved_task": task,
        "robot_type": robot_type,
        "data_id": data_id,
        "input_root": str(input_root.resolve()),
        "ref_dt_evidence": ref_dt_evidence,
        "native_run_dir": rel(native_run_dir, route_root),
        "native_run_manifest": rel(run_manifest_path, route_root),
        "native_outcome": rel(outcome_path, route_root),
        "native_returncode": native_rc,
        "native_returncode_authoritative": True,
        "native_robot_video": rel(robot, route_root) if robot is not None else None,
        "native_metrics": metrics,
        "external_qc": {"role": "diagnostic_only"},
        "input_record": rel(input_record_path, route_root),
        "log": rel(log_path, route_root),
        "backend_commit": ((run_manifest.get("backend") or {}).get("commit")),
        "backend_clean_after": outcome.get("backend_clean_after"),
    }
    write_json(route_root / "native_spider_run_binding.json", payload)
    return payload


def run_spider_cells(args: argparse.Namespace, dst: Path) -> list[dict]:
    selected_spider_cells = [
        (cell.trajectory_6dof, cell.hand_source)
        for cell in args.matrix_cells
        if cell.retargeting == "spider"
    ]
    if args.single_cell and not selected_spider_cells:
        return []
    if args.run_spider == "none":
        return collect_existing_spider_report(args, dst)
    cells = (
        selected_spider_cells
        if args.single_cell
        else spider_cell_order(args.run_spider)
    )

    results = []
    env = os.environ.copy()
    ensure_dir(dst / "logs")

    for trajectory, hand_source in cells:
        key = cell_key(trajectory, hand_source, "spider")
        robot = find_spider_robot(
            dst,
            key,
            args.prefer_spider_ik_video,
            allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
        )
        if args.skip_existing_spider and robot is not None and robot.exists():
            item = spider_report_item(args, dst, key, returncode=0)
            if item is not None:
                item["skipped_existing"] = True
                results.append(item)
            else:
                results.append({"cell": key, "status": "ok", "robot": rel(robot, dst), "skipped_existing": True})
            continue
        source_task = args.dai_task if trajectory == "egoinfinity" else args.dai_source_task
        source_hand_type = args.dai_hand_type if trajectory == "egoinfinity" else args.dai_source_hand_type
        input_bundle = find_exact_spider_input_bundle(
            REPO_ROOT / "experiments" / args.source_run,
            trajectory,
            hand_source,
            source_task,
            source_hand_type,
            args,
        )
        input_record_path = (
            dst
            / "intermediates"
            / "retargeting"
            / "spider"
            / key
            / "spider_input_record.json"
        )
        remove_path(input_record_path)
        cell_env = env.copy()
        if input_bundle is not None:
            cell_env["SPIDER_SOURCE_KEYPOINTS"] = str(input_bundle["keypoints"])
            cell_env["SPIDER_OBJECT_VISUAL"] = str(input_bundle["object_visual"])
            cell_env["SPIDER_SOURCE_ADAPTER_MANIFEST"] = str(input_bundle["adapter_manifest"])
        adapter_manifest = (
            load_json_optional(Path(str(input_bundle["adapter_manifest"])))
            if input_bundle is not None
            else None
        )
        keypoint_quality = summarize_keypoint_quality(
            Path(str(input_bundle["keypoints"])) if input_bundle is not None else None
        )
        hand_selection = spider_single_object_hand_selection(
            args.spider_hand_type,
            source_hand_type,
            keypoint_quality,
            adapter_manifest,
        )
        resolved_spider_hand_type = str(hand_selection["resolved_hand_type"])
        selection_path = (
            dst
            / "intermediates"
            / "retargeting"
            / "spider"
            / key
            / "spider_matrix_hand_selection.json"
        )
        write_json(selection_path, hand_selection)
        route_root = selection_path.parent
        if input_bundle is None:
            write_json(
                input_record_path,
                {
                    "route": key,
                    "status": "missing_required_input",
                    "trajectory_6dof": trajectory,
                    "hand_source": hand_source,
                },
            )
            results.append({"cell": key, "status": "invalid_input", "input_rc": 4})
            continue
        resolved_task = str(input_bundle["task"])
        staged_input = stage_exact_spider_input_bundle(
            input_bundle,
            output_root=route_root / "spider_exact_input",
            resolved_hand_type=resolved_spider_hand_type,
            data_id=args.spider_data_id,
        )
        input_root = Path(str(staged_input["input_root"]))
        timebase = resolve_spider_input_timebase(
            input_root,
            task=resolved_task,
            hand_type=resolved_spider_hand_type,
            data_id=args.spider_data_id,
            adapter_manifest=Path(str(input_bundle["adapter_manifest"])),
        )
        input_record = {
            "schema_version": 1,
            "route": key,
            "trajectory_6dof": trajectory,
            "hand_source": hand_source,
            "status": "ready",
            "input_rc": 0,
            "timebase": timebase,
            "resolved_hand_type": resolved_spider_hand_type,
            "resolved_task": resolved_task,
            "staging_manifest": rel(Path(str(staged_input["manifest_path"])), dst),
            "source_adapter_manifest": str(input_bundle["adapter_manifest"]),
            "adapter_route_binding": adapter_route_binding(
                Path(str(input_bundle["adapter_manifest"])),
                trajectory=trajectory,
                hand_source=hand_source,
            ),
        }
        write_json(input_record_path, input_record)
        attempt = 1
        while (route_root / "native_runs" / f"attempt{attempt:02d}").exists():
            attempt += 1
        native_run = route_root / "native_runs" / f"attempt{attempt:02d}"
        cmd = [
            str(REPO_ROOT / "scripts" / "run_spider_retarget.sh"),
            "--spider-root",
            args.spider_root,
            "--python",
            args.spider_python,
            "--input-root",
            str(input_root),
            "--output-dir",
            str(native_run),
            "--task",
            resolved_task,
            "--robot-type",
            args.spider_robot_type,
            "--embodiment-type",
            resolved_spider_hand_type,
            "--data-id",
            str(args.spider_data_id),
            "--ref-dt",
            str(timebase["ref_dt"]),
            "--cuda-visible-devices",
            args.spider_cuda_visible_devices,
            "--egl-device-id",
            args.spider_egl_device_id,
        ]
        log_path = dst / "logs" / f"reuse12_spider_{key}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=cell_env, stdout=log, stderr=subprocess.STDOUT)
        write_native_spider_run_binding(
            route_root=route_root,
            route=key,
            trajectory_6dof=trajectory,
            hand_source=hand_source,
            hand_type=resolved_spider_hand_type,
            task=resolved_task,
            robot_type=args.spider_robot_type,
            data_id=args.spider_data_id,
            ref_dt_evidence=timebase,
            input_root=input_root,
            native_run_dir=native_run,
            input_record_path=input_record_path,
            log_path=log_path,
            returncode=proc.returncode,
        )
        robot_after = find_spider_robot(
            dst,
            key,
            args.prefer_spider_ik_video,
            allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
        )
        item = spider_report_item(args, dst, key, returncode=proc.returncode, log_path=log_path)
        if item is None:
            item = {
                "cell": key,
                "status": "failed",
                "returncode": proc.returncode,
                "log": rel(log_path, dst),
                "robot": rel(robot_after, dst) if robot_after is not None and robot_after.exists() else None,
            }
        item["hand_selection"] = {
            **hand_selection,
            "path": rel(selection_path, dst),
        }
        results.append(item)
        if proc.returncode != 0 and not args.keep_going:
            raise SystemExit(f"SPIDER failed for {key}; see {log_path}")
    return results


def spider_cell_order(run_spider: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    if run_spider == "all":
        cells += [("egoinfinity", hand) for hand in HAND_SOURCES]
    if run_spider in {"do_as_i_do", "all"}:
        cells += [("do_as_i_do", hand) for hand in HAND_SOURCES]
    return cells


def collect_existing_spider_report(args: argparse.Namespace, dst: Path) -> list[dict]:
    results: list[dict] = []
    cells = (
        [cell for cell in args.matrix_cells if cell.retargeting == "spider"]
        if args.single_cell
        else [
            MatrixCell(trajectory, hand_source, "spider")
            for trajectory in TRAJECTORIES
            for hand_source in HAND_SOURCES
        ]
    )
    for cell in cells:
        item = spider_report_item(args, dst, cell.key)
        if item is not None:
            item["reindexed_existing"] = True
            results.append(item)
    return results


def spider_report_item(
    args: argparse.Namespace,
    dst: Path,
    key: str,
    *,
    returncode: int | None = None,
    log_path: Path | None = None,
) -> dict | None:
    root = dst / "intermediates" / "retargeting" / "spider" / key
    native = _native_spider_binding(root)
    authoritative_rc = returncode
    if native is not None:
        authoritative_rc = int(native.get("native_returncode", 1))
    robot = find_spider_robot(dst, key)
    if robot is None and authoritative_rc is None and not root.exists():
        return None
    probe = probe_video(robot) if robot is not None else None
    ok = (authoritative_rc in {None, 0}) and probe is not None
    item = {
        "cell": key,
        "status": "ok" if ok else "failed",
        "robot": rel(robot, dst) if ok else None,
        "returncode": authoritative_rc,
        "backend_success": authoritative_rc in {None, 0},
        "review_video": probe,
        "manual_review_required": ok,
        "success_standard": "backend_success_and_manual_video_review",
    }
    if log_path is not None:
        item["log"] = rel(log_path, dst)
    return item


def find_spider_robot(
    dst: Path,
    key: str,
    prefer_ik: bool = False,
    **_: object,
) -> Path | None:
    del prefer_ik
    root = dst / "intermediates" / "retargeting" / "spider" / key
    native = _native_spider_binding(root)
    candidates: list[Path | None] = []
    if native is not None and int(native.get("native_returncode", 1)) == 0:
        candidates.append(
            _recorded_manifest_path(native.get("native_robot_video"), relative_to=root)
        )
    output_manifest = load_json_optional(root / "spider_output_manifest.json") or {}
    alignment = output_manifest.get("alignment") or {}
    candidates.extend([
        _recorded_manifest_path(alignment.get("aligned_video"), relative_to=root)
        if isinstance(alignment, dict)
        else None,
        root / "visualization_mjwp_act_aligned.mp4",
        root / "visualization_mjwp_aligned.mp4",
        root / "visualization_mjwp.mp4",
        root / "robot" / "visualization_mjwp.mp4",
    ])
    for candidate in candidates:
        if candidate is not None and probe_video(candidate) is not None:
            return candidate
    return None


def compose_cell(dst: Path, key: str, duration: float) -> Path:
    out = dst / "videos" / f"{key}__triptych.mp4"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "review" / "compose_triptych.py"),
        "--overlay",
        str(dst / "cells" / key / "overlay.mp4"),
        "--depth",
        str(dst / "cells" / key / "depth.mp4"),
        "--robot",
        str(dst / "cells" / key / "robot.mp4"),
        "--output",
        str(out),
        "--layout",
        "vertical",
        "--labels",
        "reconstruction overlay|depth / input geometry|retargeting robot",
    ]
    if duration > 0:
        cmd += ["--duration", str(duration)]
    subprocess.run(cmd, check=True)
    return out


def dai_retarget_output_root(robot_src: Path | None) -> Path | None:
    if robot_src is None:
        return None
    for parent in robot_src.parents:
        if parent.name == "retargeting_outputs":
            return parent
    return None


def dai_adapter_manifest_for_robot(robot_src: Path | None) -> Path | None:
    output_root = dai_retarget_output_root(robot_src)
    if output_root is None:
        return None
    candidate = output_root.parent / "raw_dir" / "adapter_manifest.json"
    return candidate if candidate.exists() else None


def egoinfinity_native_cell_binding(
    *,
    cell: str,
    native_cell: str,
    source_robot: Path | None,
) -> dict[str, object]:
    errors: list[str] = []
    if cell != native_cell:
        errors.append(f"cell={cell!r}, native result belongs to {native_cell!r}")
    if source_robot is None or not source_robot.exists():
        errors.append("missing_native_egoinfinity_robot")
    return {
        "status": "ok" if not errors else "invalid",
        "backend": "egoinfinity",
        "requested_cell": cell,
        "native_cell": native_cell,
        "source_robot": str(source_robot) if source_robot is not None else None,
        "errors": errors,
    }


def spider_route_asset_binding(
    *,
    route_root: Path,
    route: str,
    trajectory: str,
    hand_source: str,
) -> dict[str, object]:
    binding_path = route_root / "native_spider_run_binding.json"
    input_record_path = route_root / "spider_input_record.json"
    native = load_json_optional(binding_path)
    input_record = load_json_optional(input_record_path)
    errors: list[str] = []
    if not isinstance(native, dict):
        errors.append("missing_native_spider_run_binding")
        native = {}
    if not isinstance(input_record, dict):
        errors.append("missing_spider_input_record")
        input_record = {}
    for label, payload in (("native", native), ("input", input_record)):
        for key, expected in (
            ("route", route),
            ("trajectory_6dof", trajectory),
            ("hand_source", hand_source),
        ):
            if payload.get(key) != expected:
                errors.append(
                    f"{label}.{key}={payload.get(key)!r}, expected {expected!r}"
                )
    if input_record.get("status") != "ready":
        errors.append(f"input.status={input_record.get('status')!r}, expected 'ready'")
    if native.get("native_returncode") != 0:
        errors.append(
            f"native.native_returncode={native.get('native_returncode')!r}, expected 0"
        )
    recorded_adapter_binding = input_record.get("adapter_route_binding")
    adapter_binding_source = "spider_input_record"
    if not isinstance(recorded_adapter_binding, dict):
        staging = load_json_optional(
            route_root / "spider_exact_input" / "spider_input_staging_manifest.json"
        )
        source_keypoints = (
            ((staging or {}).get("trajectory_keypoints") or {}).get("source")
            if isinstance(staging, dict)
            else None
        )
        source_adapter = None
        if isinstance(source_keypoints, str):
            keypoints_path = Path(source_keypoints)
            for parent in keypoints_path.parents:
                if parent.name == "retargeting_outputs":
                    candidate = parent.parent / "raw_dir" / "adapter_manifest.json"
                    if candidate.exists():
                        source_adapter = candidate
                    break
        if source_adapter is not None:
            recorded_adapter_binding = adapter_route_binding(
                source_adapter,
                trajectory=trajectory,
                hand_source=hand_source,
            )
            adapter_binding_source = "inferred_from_staged_source_keypoints"
    if not isinstance(recorded_adapter_binding, dict):
        errors.append("missing_input_adapter_route_binding")
    elif recorded_adapter_binding.get("status") != "ok":
        errors.append("input_adapter_route_binding_is_not_exact")
    return {
        "status": "ok" if not errors else "invalid",
        "backend": "spider",
        "requested_cell": route,
        "trajectory_6dof": trajectory,
        "hand_source": hand_source,
        "native_binding": str(binding_path),
        "input_record": str(input_record_path),
        "adapter_route_binding": recorded_adapter_binding,
        "adapter_route_binding_source": adapter_binding_source,
        "errors": errors,
    }


def _recorded_manifest_path(value: object, *, relative_to: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    for path in (relative_to / candidate, REPO_ROOT / candidate):
        if path.exists():
            return path
    return relative_to / candidate


def build_cells(dst: Path, source: Path, srcs: dict[str, Path | None], args: argparse.Namespace) -> list[dict]:
    reports = []
    fallback_spider = None
    if args.allow_spider_fallback:
        fallback_spider = first_existing(
            [
                find_spider_robot(
                    dst,
                    cell_key("do_as_i_do", "estimated", "spider"),
                    args.prefer_spider_ik_video,
                    allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
                )
                or Path("__missing__"),
                find_spider_robot(
                    dst,
                    cell_key("do_as_i_do", "aoe", "spider"),
                    args.prefer_spider_ik_video,
                    allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
                )
                or Path("__missing__"),
            ]
        )
    source_reuse_manifest = load_json_optional(source / "reuse" / "reuse_manifest.json") or {}
    native_ego_cell = (
        ((source_reuse_manifest.get("egoinfinity_full") or {}).get("cell"))
        if isinstance(source_reuse_manifest, dict)
        else None
    ) or cell_key("egoinfinity", "estimated", "egoinfinity")

    for selected_cell in args.matrix_cells:
        trajectory = selected_cell.trajectory_6dof
        hand_source = selected_cell.hand_source
        retargeting = selected_cell.retargeting
        overlay = dst / "assets" / "trajectory_6dof" / trajectory / args.task / "overlay.mp4"
        depth = dst / "assets" / "trajectory_6dof" / trajectory / args.task / "depth.mp4"
        key = selected_cell.key
        cell_dir = ensure_dir(dst / "cells" / key)
        assets_dir = ensure_dir(dst / "assets" / "cells" / key)
        if retargeting == "egoinfinity":
            asset_binding = egoinfinity_native_cell_binding(
                cell=key,
                native_cell=native_ego_cell,
                source_robot=srcs["ego_robot"],
            )
            if asset_binding["status"] == "ok":
                robot_src = srcs["ego_robot"]
                robot_note = "exact source EgoInfinity/G1 native cell"
            else:
                robot_src = None
                robot_note = (
                    f"unsupported exact EgoInfinity binding: native result belongs to {native_ego_cell}"
                )
        elif retargeting == "do_as_i_do":
            robot_task = args.dai_task if trajectory == "egoinfinity" else args.dai_source_task
            robot_hand_type = args.dai_hand_type if trajectory == "egoinfinity" else args.dai_source_hand_type
            robot_src, robot_note = find_dai_robot_in_overrides(
                args.dai_override_sources,
                trajectory,
                hand_source,
                robot_task,
                robot_hand_type,
                args,
            )
            if robot_src is None:
                robot_src, robot_note = find_quality_dai_robot_in_source(
                    source,
                    trajectory,
                    hand_source,
                    robot_task,
                    robot_hand_type,
                    args,
                )
            if robot_src is None and hand_source != "estimated" and args.allow_dai_hand_fallback:
                robot_src, robot_note = find_quality_dai_robot_in_source(
                    source,
                    trajectory,
                    "estimated",
                    robot_task,
                    robot_hand_type,
                    args,
                )
                if robot_src is None and (robot_task != args.task or robot_hand_type != args.hand_type):
                    robot_src, robot_note = find_quality_dai_robot_in_source(
                        source,
                        trajectory,
                        "estimated",
                        args.task,
                        args.hand_type,
                        args,
                    )
                robot_note = (
                    robot_note or "fallback Do-as-I-Do/Sharpa robot from same trajectory estimated-hand cell"
                    if robot_src is not None
                    else "missing source Do-as-I-Do/Sharpa cell"
                )
            elif robot_src is None:
                robot_note = "missing source Do-as-I-Do/Sharpa cell"
            asset_binding = adapter_route_binding(
                dai_adapter_manifest_for_robot(robot_src),
                trajectory=trajectory,
                hand_source=hand_source,
            )
            if robot_src is not None and asset_binding["status"] != "ok":
                robot_src = None
                robot_note = "rejected Do-as-I-Do robot with non-exact route provenance"
        else:
            exact = find_spider_robot(
                dst,
                key,
                args.prefer_spider_ik_video,
                allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
            )
            robot_src = exact or fallback_spider
            asset_binding = spider_route_asset_binding(
                route_root=dst / "intermediates" / "retargeting" / "spider" / key,
                route=key,
                trajectory=trajectory,
                hand_source=hand_source,
            )
            if exact and asset_binding["status"] == "ok":
                robot_note = "exact SPIDER aligned MJWP cell"
            elif fallback_spider is not None:
                robot_note = "fallback SPIDER robot reused from Do-as-I-Do trajectory cell"
            else:
                robot_note = "missing source SPIDER cell"
            if robot_src is not None and asset_binding["status"] != "ok":
                robot_src = None
                robot_note = "rejected SPIDER robot with non-exact route provenance"

        overlay_dst = materialize(overlay, cell_dir / "overlay.mp4", args.mode)
        depth_dst = materialize(depth, cell_dir / "depth.mp4", args.mode)
        robot_dst = materialize(robot_src, cell_dir / "robot.mp4", args.mode)
        robot_probe = probe_video(robot_dst) if robot_dst is not None else None
        missing = [
            name
            for name, value in {
                "overlay": overlay_dst,
                "depth": depth_dst,
                "robot": robot_dst if robot_probe is not None else None,
            }.items()
            if value is None
        ]
        video = None
        if args.compose and not missing:
            video = compose_cell(dst, key, args.duration)
        elif args.compose:
            remove_path(dst / "videos" / f"{key}__triptych.mp4")
        manifest = {
            "cell_key": key,
            "settings": {
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "retargeting": retargeting,
                "task": args.task,
                "hand_type": args.hand_type,
                "spider_hand_type": args.spider_hand_type,
                "dai_task": args.dai_task,
                "dai_hand_type": args.dai_hand_type,
                "dai_source_task": args.dai_source_task,
                "dai_source_hand_type": args.dai_source_hand_type,
            },
            "reuse_policy": {
                "source_run": args.source_run,
                "overlay_depth": f"from {trajectory} trajectory_6dof full run",
                "robot": robot_note,
            },
            "route_record": {
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "retargeting": retargeting,
                "source_robot": rel(robot_src, dst) if robot_src else None,
            },
            "asset_provenance_contract": asset_binding,
            "success_standard": "backend_success_and_manual_video_review",
            "manual_review_required": robot_probe is not None,
            "review_video": robot_probe,
            "assets": {
                "cell.overlay": rel(overlay_dst, dst),
                "cell.depth": rel(depth_dst, dst),
                "cell.robot": rel(robot_dst, dst),
                "source.overlay": rel(overlay, dst),
                "source.depth": rel(depth, dst),
                "source.robot": rel(robot_src, dst) if robot_src else None,
                "video.triptych": rel(video, dst) if video else None,
            },
            "missing": missing,
        }
        write_json(cell_dir / "manifest.json", manifest)
        write_json(assets_dir / "asset_manifest.json", manifest)
        reports.append(manifest)
    return reports


def collect_route_availability(
    source: Path,
    spider_report: list[dict],
) -> dict[str, dict]:
    source_manifest = load_json_optional(source / "reuse" / "reuse_manifest.json") or {}
    source_availability = source_manifest.get("route_availability") or {}
    result = {
        str(route): dict(entry)
        for route, entry in source_availability.items()
        if isinstance(entry, dict)
    }
    for item in spider_report:
        if not isinstance(item, dict) or not item.get("cell"):
            continue
        if item.get("status") != "invalid_input":
            continue
        result[str(item["cell"])] = {
            "input_rc": item.get("input_rc"),
            "post_retarget_rc": None,
            "status": "invalid_input",
            "available": False,
            "reason": "missing_required_backend_input",
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse one EgoInfinity full run and one Do-as-I-Do full run to write a provenance-checked 12-cell comparison report."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--cell",
        choices=[cell.key for cell in all_matrix_cells()],
        help="Materialize exactly one of the 12 matrix cells instead of the full matrix.",
    )
    parser.add_argument("--trajectory-6dof", choices=TRAJECTORIES)
    parser.add_argument("--hand-source", choices=HAND_SOURCES)
    parser.add_argument("--retargeting", choices=RETARGETERS)
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--hand-type", default="bimanual")
    parser.add_argument(
        "--spider-hand-type",
        default=None,
        help=(
            "Hand type for SPIDER retargeting. Auto selects one interaction hand per trajectory because "
            "SPIDER's bimanual model creates two independent object bodies for a single-object scene. "
            "Pass bimanual explicitly only when that two-object model is intended."
        ),
    )
    parser.add_argument(
        "--dai-task",
        default=None,
        help="Do-as-I-Do retargeting task to use for DAI robot cells. Defaults to --task.",
    )
    parser.add_argument(
        "--dai-hand-type",
        default=None,
        help="Do-as-I-Do hand type to use for DAI robot cells. Defaults to --hand-type.",
    )
    parser.add_argument(
        "--dai-source-task",
        default=None,
        help="Do-as-I-Do source trajectory asset task. Defaults to --task.",
    )
    parser.add_argument(
        "--dai-source-hand-type",
        default=None,
        help="Do-as-I-Do source trajectory asset hand type. Defaults to --hand-type.",
    )
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--run-spider", choices=["none", "do_as_i_do", "all"], default="do_as_i_do")
    parser.add_argument("--allow-spider-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-spider-mjwp-fallback-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow SPIDER cells whose MJWP output was replaced by the IK/object-reference "
            "fallback to enter composed videos. Default false so failed SPIDER dynamics "
            "remain visibly missing instead of being presented as successful retargeting."
        ),
    )
    parser.add_argument("--allow-dai-hand-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dai-auto-select-interaction-hand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When searching DAI override runs for EgoInfinity->DAI cells, also score bimanual/opposite-hand "
            "candidates and pick the hand with better hand-object interaction."
        ),
    )
    parser.add_argument(
        "--allow-cross-trajectory-dai-robot-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When an exact Do-as-I-Do robot cell has weak hand-object quality, allow a quality-valid "
            "robot cell from the other trajectory to be used as the visible DAI retarget result."
        ),
    )
    parser.add_argument(
        "--dai-override-run",
        action="append",
        default=[],
        help=(
            "Experiment run to search before --source-run for Do-as-I-Do robot cells. "
            "May be passed multiple times; useful for fresh repaired DAI retarget probes."
        ),
    )
    parser.add_argument(
        "--dai-override-ego-spider-keypoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also let DAI override runs replace EgoInfinity->SPIDER keypoints. "
            "Only quality-valid EgoInfinity override bundles are used; object meshes still come from the "
            "matrix trajectory assets."
        ),
    )
    parser.add_argument("--skip-existing-spider", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--spider-root",
        default=os.environ.get(
            "SPIDER_ROOT", str(REPO_ROOT / "third_party" / "SPIDER")
        ),
    )
    parser.add_argument("--spider-python", default=os.environ.get("SPIDER_PYTHON"))
    parser.add_argument("--spider-cuda-visible-devices", default="")
    parser.add_argument("--spider-egl-device-id", default="0")
    parser.add_argument("--spider-device", default="cuda:0")
    parser.add_argument("--spider-robot-type", default="xhand")
    parser.add_argument("--spider-dataset-name", default="do_as_i_do")
    parser.add_argument("--spider-data-id", type=int, default=0)
    parser.add_argument("--spider-max-sim-steps", default="-1")
    parser.add_argument("--spider-num-samples", default="1024")
    parser.add_argument("--spider-max-num-iterations", default="16")
    parser.add_argument("--spider-skip-mjwp", action="store_true")
    parser.add_argument("--prefer-spider-ik-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--spider-fuse-ego-object-for-dai",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly use DAI hand keypoints with EgoInfinity object pose/mesh for do_as_i_do->SPIDER cells. "
            "Default is false so pure DAI cells stay DAI-native or are marked invalid."
        ),
    )
    args = parser.parse_args()
    try:
        args.matrix_cells = select_matrix_cells(
            cell=args.cell,
            trajectory_6dof=args.trajectory_6dof,
            hand_source=args.hand_source,
            retargeting=args.retargeting,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.single_cell = len(args.matrix_cells) == 1
    if not args.run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = (
            f"cell_{args.matrix_cells[0].key}"
            if args.single_cell
            else "reuse12"
        )
        args.run_name = f"{args.source_run}__{suffix}_{stamp}"
    if args.dai_task is None:
        args.dai_task = args.task
    if args.dai_hand_type is None:
        args.dai_hand_type = args.hand_type
    if args.dai_source_task is None:
        args.dai_source_task = args.task
    if args.dai_source_hand_type is None:
        args.dai_source_hand_type = args.hand_type
    if args.spider_hand_type is None:
        args.spider_hand_type = "auto"
    args.dai_override_sources = [REPO_ROOT / "experiments" / run for run in args.dai_override_run]
    return args


def main() -> int:
    args = parse_args()
    if args.spider_fuse_ego_object_for_dai:
        raise SystemExit(
            "--spider-fuse-ego-object-for-dai is disabled: production routes may not replace only the "
            "object trajectory while keeping a different hand source"
        )
    forbidden_reuse = []
    if args.allow_spider_fallback:
        forbidden_reuse.append("--allow-spider-fallback")
    if args.allow_dai_hand_fallback:
        forbidden_reuse.append("--allow-dai-hand-fallback")
    if args.allow_cross_trajectory_dai_robot_fallback:
        forbidden_reuse.append("--allow-cross-trajectory-dai-robot-fallback")
    if args.dai_override_ego_spider_keypoints:
        forbidden_reuse.append("--dai-override-ego-spider-keypoints")
    if args.dai_override_run:
        forbidden_reuse.append("--dai-override-run")
    if forbidden_reuse:
        raise SystemExit(
            "strict matrix provenance forbids cross-cell reuse options: "
            + ", ".join(forbidden_reuse)
        )
    source = REPO_ROOT / "experiments" / args.source_run
    if not source.exists():
        raise SystemExit(f"source run does not exist: {source}")
    for override in args.dai_override_sources:
        if not override.exists():
            raise SystemExit(f"DAI override run does not exist: {override}")
    dst = REPO_ROOT / "experiments" / args.run_name
    ensure_dir(dst / "logs")
    ensure_dir(dst / "videos")
    srcs = source_paths(
        source,
        args.task,
        args.hand_type,
        args.dai_task,
        args.dai_hand_type,
        args.dai_source_task,
        args.dai_source_hand_type,
    )
    dai_override_asset_report = apply_dai_override_trajectory_assets(srcs, args.dai_override_sources, args)
    materialize(source, dst / "reuse" / "source_run", args.mode)
    traj_report = prepare_trajectory_assets(
        dst,
        srcs,
        args.task,
        args.mode,
        trajectories=tuple(dict.fromkeys(cell.trajectory_6dof for cell in args.matrix_cells)),
        spider_fuse_ego_object_for_dai=args.spider_fuse_ego_object_for_dai,
    )
    spider_report = run_spider_cells(args, dst)
    route_availability = collect_route_availability(source, spider_report)
    if args.single_cell:
        selected_keys = {cell.key for cell in args.matrix_cells}
        route_availability = {
            key: value
            for key, value in route_availability.items()
            if key in selected_keys
        }
    cell_reports = build_cells(dst, source, srcs, args)
    summary = {
        "run_name": args.run_name,
        "source_run": args.source_run,
        "experiment_root": str(dst),
        "git": git_metadata(),
        "command": {
            "argv": sys.argv,
            "cwd": str(REPO_ROOT),
        },
        "gpu_mapping": gpu_mapping_from_env(),
        "recommendation": {
            "object_reconstruction_and_6dof_tracking": "egoinfinity",
            "dai_native_reconstruction": "baseline_only",
            "note": "Pure DAI cells use DAI-native object assets; Ego trajectory cells require exact EgoInfinity object provenance.",
        },
        "success_policy": {
            "policy": "backend_success_and_manual_video_review",
            "criterion": "backend_returncode_zero_and_decodable_robot_video_then_manual_review",
            "matrix_completeness_required": False,
            "numerical_qc_is_diagnostic": True,
            "manual_review_required": True,
            "failed_routes_must_remain_explicit": True,
        },
        "mode": args.mode,
        "selection_mode": "single_cell" if args.single_cell else "full_12",
        "selected_cells": [cell.key for cell in args.matrix_cells],
        "task": args.task,
        "hand_type": args.hand_type,
        "spider_hand_type": args.spider_hand_type,
        "dai_task": args.dai_task,
        "dai_hand_type": args.dai_hand_type,
        "dai_source_task": args.dai_source_task,
        "dai_source_hand_type": args.dai_source_hand_type,
        "spider_fuse_ego_object_for_dai": args.spider_fuse_ego_object_for_dai,
        "dai_override_runs": args.dai_override_run,
        "dai_override_assets": dai_override_asset_report,
        "trajectory_assets": traj_report,
        "spider": spider_report,
        "route_availability": route_availability,
        "cells": len(cell_reports),
        "composed": sum(1 for item in cell_reports if item["assets"].get("video.triptych")),
        "missing_cells": {item["cell_key"]: item["missing"] for item in cell_reports if item["missing"]},
        "fallback_cells": [
            item["cell_key"]
            for item in cell_reports
            if item["reuse_policy"]["robot"].startswith("fallback")
        ],
        "dai_hand_fallback_cells": [
            item["cell_key"]
            for item in cell_reports
            if item["reuse_policy"]["robot"].startswith("fallback Do-as-I-Do")
        ],
    }
    write_json(dst / "reuse_12_demo_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

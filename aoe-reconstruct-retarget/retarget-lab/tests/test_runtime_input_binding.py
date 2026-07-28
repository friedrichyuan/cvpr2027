from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    pytest.importorskip("numpy")
    pytest.importorskip("trimesh")
    if name == "prepare_egoinfinity_do_as_i_do_raw_dir":
        pytest.importorskip("cv2")
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_reference_gravity_is_preserved(tmp_path):
    module = load_script("prepare_egoinfinity_do_as_i_do_raw_dir")
    source = tmp_path / "source_gravity.json"
    output = tmp_path / "gravity.json"
    source.write_text(
        json.dumps(
            {
                "vec3d": [0.071, -0.696, -0.714],
                "roll_deg": -5.9,
                "pitch_deg": -45.6,
                "gravity_quality": {
                    "status": "dynamic_camera_reference_frame",
                    "selected_frame": 155,
                    "reference_frame": 152,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = module.write_gravity_metadata(output, source, max_tilt_deg=25.0)

    assert payload["vec3d"] == [0.071, -0.696, -0.714]
    assert payload["gravity_clamped"] is False
    assert payload["gravity_preserved_reason"] == "dynamic_camera_reference_frame_selection"
    assert payload["vector_semantics"] == "camera_frame_world_up"


def test_unqualified_large_tilt_still_uses_upright_fallback(tmp_path):
    module = load_script("prepare_egoinfinity_do_as_i_do_raw_dir")
    source = tmp_path / "source_gravity.json"
    output = tmp_path / "gravity.json"
    source.write_text(
        json.dumps(
            {
                "vec3d": [0.0, -0.7, -0.7],
                "roll_deg": 0.0,
                "pitch_deg": -45.0,
                "gravity_quality": {"status": "static_camera_aggregate"},
            }
        ),
        encoding="utf-8",
    )

    payload = module.write_gravity_metadata(output, source, max_tilt_deg=25.0)

    assert payload["vec3d"] == [0.0, -1.0, 0.0]
    assert payload["gravity_clamped"] is True
    assert payload["original_gravity"]["vec3d"] == [0.0, -0.7, -0.7]


def test_invalid_dynamic_reference_gravity_fails_closed(tmp_path):
    module = load_script("prepare_egoinfinity_do_as_i_do_raw_dir")
    source = tmp_path / "source_gravity.json"
    output = tmp_path / "gravity.json"
    source.write_text(
        json.dumps(
            {
                "vec3d": [0.0, -0.7, -0.7],
                "roll_deg": 0.0,
                "pitch_deg": -45.0,
                "gravity_quality": {
                    "status": "dynamic_camera_reference_frame",
                    "selected_frame": None,
                    "reference_frame": 100,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing a valid reference-frame"):
        module.write_gravity_metadata(output, source, max_tilt_deg=25.0)
    assert not output.exists()


def test_spider_prefers_adapter_interaction_hand_over_heuristic():
    module = load_script("reuse_v4_for_12_demos")
    quality = {
        "sides": {
            "left": {
                "contact_active": 10,
                "contact_mean": 1.0,
                "finger_object_distance_median": 0.01,
                "finger_object_distance_min": 0.001,
            },
            "right": {
                "contact_active": 1,
                "contact_mean": 0.1,
                "finger_object_distance_median": 0.2,
                "finger_object_distance_min": 0.1,
            },
        }
    }
    adapter = {"surface_contact_geometry": {"selected_hand": "right"}}

    result = module.spider_single_object_hand_selection(
        "auto", "bimanual", quality, adapter
    )

    assert result["resolved_hand_type"] == "right"
    assert result["reason"] == "adapter_manifest_interaction_hand"


def test_spider_staging_rebinds_hand_without_changing_keypoints(tmp_path):
    module = load_script("reuse_v4_for_12_demos")
    source = tmp_path / "source"
    task = "box_task"
    source_keypoints = source / "mano" / "bimanual" / task / "0" / "trajectory_keypoints.npz"
    source_task_info = source_keypoints.parents[1] / "task_info.json"
    source_visual = source / "assets" / "objects" / task / "visual.obj"
    source_convex = source_visual.parent / "convex"
    source_keypoints.parent.mkdir(parents=True)
    source_keypoints.write_bytes(b"exact-keypoint-bytes")
    source_task_info.write_text(
        json.dumps(
            {
                "task": task,
                "dataset_name": "do_as_i_do",
                "robot_type": "mano",
                "embodiment_type": "bimanual",
                "data_id": 0,
                "right_object_mesh_dir": f"assets/objects/{task}",
                "right_object_convex_dir": f"assets/objects/{task}/convex",
                "left_object_mesh_dir": None,
            }
        ),
        encoding="utf-8",
    )
    source_visual.parent.mkdir(parents=True)
    source_visual.write_text("v 0 0 0\n", encoding="utf-8")
    source_convex.mkdir()
    (source_convex / "part.obj").write_text("v 0 0 0\n", encoding="utf-8")
    output = tmp_path / "staged"

    result = module.stage_exact_spider_input_bundle(
        {
            "task": task,
            "hand_type": "bimanual",
            "keypoints": source_keypoints,
            "task_info": source_task_info,
            "object_visual": source_visual,
            "object_convex": source_convex,
        },
        output_root=output,
        resolved_hand_type="right",
        data_id=0,
    )

    staged_payload = json.loads(Path(result["task_info"]).read_text(encoding="utf-8"))
    assert staged_payload["embodiment_type"] == "right"
    assert staged_payload["right_object_mesh_dir"] == f"assets/objects/{task}"
    assert staged_payload["left_object_mesh_dir"] is None
    assert Path(result["keypoints"]).read_bytes() == source_keypoints.read_bytes()
    assert result["manifest"]["trajectory_keypoints"]["byte_identical"] is True
    assert not any(path.is_symlink() for path in output.rglob("*"))


def write_adapter_manifest(
    path: Path,
    *,
    hand_source: str,
    object_source: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hand_source": hand_source,
        "hand_geometry_source": hand_source,
        "object_track_source": object_source,
        "object_mesh_source": object_source,
        "retarget_object_source": object_source,
    }
    if object_source == "egoinfinity":
        payload["object_geometry_source"] = "ego"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ego_to_dai_rejects_dai_native_reconstruction_assets(tmp_path):
    module = load_script("reuse_v4_for_12_demos")
    manifest = write_adapter_manifest(
        tmp_path / "adapter_manifest.json",
        hand_source="aoe",
        object_source="dai_native",
    )

    result = module.adapter_route_binding(
        manifest,
        trajectory="egoinfinity",
        hand_source="aoe",
    )

    assert result["status"] == "invalid"
    assert any("object_track_source" in error for error in result["errors"])
    assert any("object_geometry_source" in error for error in result["errors"])


def test_ego_to_dai_accepts_only_ego_object_assets(tmp_path):
    module = load_script("reuse_v4_for_12_demos")
    manifest = write_adapter_manifest(
        tmp_path / "adapter_manifest.json",
        hand_source="aoe",
        object_source="egoinfinity",
    )

    result = module.adapter_route_binding(
        manifest,
        trajectory="egoinfinity",
        hand_source="aoe",
    )

    assert result["status"] == "ok"
    assert result["actual"]["object_track_source"] == "egoinfinity"
    assert result["actual"]["object_mesh_source"] == "egoinfinity"
    assert result["actual"]["retarget_object_source"] == "egoinfinity"


@pytest.mark.parametrize("trajectory", ["egoinfinity", "do_as_i_do"])
@pytest.mark.parametrize("hand_source", ["aoe", "estimated"])
def test_all_dai_spider_input_axis_bindings_accept_only_exact_sources(
    tmp_path, trajectory, hand_source
):
    module = load_script("reuse_v4_for_12_demos")
    object_source = "egoinfinity" if trajectory == "egoinfinity" else "dai_native"
    manifest = write_adapter_manifest(
        tmp_path / f"{trajectory}_{hand_source}.json",
        hand_source=hand_source,
        object_source=object_source,
    )

    result = module.adapter_route_binding(
        manifest,
        trajectory=trajectory,
        hand_source=hand_source,
    )

    assert result["status"] == "ok"
    assert result["actual"]["hand_source"] == hand_source
    assert result["actual"]["object_track_source"] == object_source
    assert result["actual"]["object_mesh_source"] == object_source
    assert result["actual"]["retarget_object_source"] == object_source


@pytest.mark.parametrize(
    ("cell", "expected_status"),
    [
        (
            "traj_egoinfinity__hand_estimated__retarget_egoinfinity",
            "ok",
        ),
        (
            "traj_egoinfinity__hand_aoe__retarget_egoinfinity",
            "invalid",
        ),
        (
            "traj_do_as_i_do__hand_estimated__retarget_egoinfinity",
            "invalid",
        ),
        (
            "traj_do_as_i_do__hand_aoe__retarget_egoinfinity",
            "invalid",
        ),
    ],
)
def test_all_ego_retarget_cells_keep_native_support_boundary(
    tmp_path, cell, expected_status
):
    module = load_script("reuse_v4_for_12_demos")
    robot = tmp_path / "ego_robot.mp4"
    robot.write_bytes(b"video")
    native_cell = "traj_egoinfinity__hand_estimated__retarget_egoinfinity"

    result = module.egoinfinity_native_cell_binding(
        cell=cell,
        native_cell=native_cell,
        source_robot=robot,
    )

    assert result["status"] == expected_status


def test_native_ego_robot_cannot_be_relabelled_as_another_matrix_cell(tmp_path):
    module = load_script("reuse_v4_for_12_demos")
    robot = tmp_path / "ego_robot.mp4"
    robot.write_bytes(b"video")
    native_cell = "traj_egoinfinity__hand_estimated__retarget_egoinfinity"

    exact = module.egoinfinity_native_cell_binding(
        cell=native_cell,
        native_cell=native_cell,
        source_robot=robot,
    )
    wrong = module.egoinfinity_native_cell_binding(
        cell="traj_do_as_i_do__hand_aoe__retarget_egoinfinity",
        native_cell=native_cell,
        source_robot=robot,
    )

    assert exact["status"] == "ok"
    assert wrong["status"] == "invalid"


def test_spider_input_lookup_does_not_glob_another_task(tmp_path):
    module = load_script("reuse_v4_for_12_demos")
    source = tmp_path / "source"
    route = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    retargeting_outputs = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / route
        / "retargeting_outputs"
    )
    other_keypoints = (
        retargeting_outputs
        / "mano"
        / "right"
        / "other_task"
        / "0"
        / "trajectory_keypoints.npz"
    )
    other_keypoints.parent.mkdir(parents=True)
    other_keypoints.write_bytes(b"not-used")
    write_adapter_manifest(
        retargeting_outputs.parent / "raw_dir" / "adapter_manifest.json",
        hand_source="aoe",
        object_source="egoinfinity",
    )

    result = module.find_exact_spider_input_bundle(
        source,
        "egoinfinity",
        "aoe",
        "requested_task",
        "right",
        SimpleNamespace(dai_auto_select_interaction_hand=False),
    )

    assert result is None


def test_spider_output_binding_rejects_wrong_route(tmp_path):
    module = load_script("reuse_v4_for_12_demos")
    route_root = tmp_path / "route"
    route_root.mkdir()
    expected_route = "traj_egoinfinity__hand_aoe__retarget_spider"
    wrong_route = "traj_do_as_i_do__hand_aoe__retarget_spider"
    (route_root / "native_spider_run_binding.json").write_text(
        json.dumps(
            {
                "route": wrong_route,
                "trajectory_6dof": "do_as_i_do",
                "hand_source": "aoe",
                "native_returncode": 0,
            }
        ),
        encoding="utf-8",
    )
    (route_root / "spider_input_record.json").write_text(
        json.dumps(
            {
                "route": wrong_route,
                "trajectory_6dof": "do_as_i_do",
                "hand_source": "aoe",
                "status": "ready",
                "adapter_route_binding": {"status": "ok"},
            }
        ),
        encoding="utf-8",
    )

    result = module.spider_route_asset_binding(
        route_root=route_root,
        route=expected_route,
        trajectory="egoinfinity",
        hand_source="aoe",
    )

    assert result["status"] == "invalid"
    assert any("native.route" in error for error in result["errors"])


def test_asset_indexer_rejects_relabelled_native_ego_cell(tmp_path):
    module = load_script("index_cell_assets")

    result = module.exact_cell_asset_binding(
        tmp_path,
        trajectory_6dof="egoinfinity",
        hand_source="aoe",
        retargeting="egoinfinity",
    )

    assert result["status"] == "invalid"
    assert result["native_cell"] == (
        "traj_egoinfinity__hand_estimated__retarget_egoinfinity"
    )


def test_asset_indexer_rejects_ego_route_with_dai_native_object(tmp_path):
    module = load_script("index_cell_assets")
    key = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    manifest = (
        tmp_path
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / key
        / "raw_dir"
        / "adapter_manifest.json"
    )
    write_adapter_manifest(
        manifest,
        hand_source="aoe",
        object_source="dai_native",
    )

    result = module.exact_cell_asset_binding(
        tmp_path,
        trajectory_6dof="egoinfinity",
        hand_source="aoe",
        retargeting="do_as_i_do",
    )

    assert result["status"] == "invalid"
    assert result["actual"]["object_track_source"] == "dai_native"

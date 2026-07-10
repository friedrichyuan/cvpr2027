#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SPIDER_PYTHON:-python}"
run_name="foundation_jar_visual_demo_v1"
trajectory_6dof="do_as_i_do"
hand_source="estimated"
task="foundation_jar_bimanual_leftscale"
hand_type="bimanual"
robot_type="xhand"
dataset_name="do_as_i_do"
data_id=0
check_only=0
skip_mjwp=0
ik_end_idx="${SPIDER_IK_END_IDX:--1}"
max_sim_steps="${SPIDER_MAX_SIM_STEPS:--1}"
num_samples="${SPIDER_NUM_SAMPLES:-1024}"
max_num_iterations="${SPIDER_MAX_NUM_ITERATIONS:-16}"
device="${SPIDER_DEVICE:-cuda:0}"
simulator_viewer="${SPIDER_VIEWER:-mujoco}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) run_name="$2"; shift 2 ;;
    --trajectory-6dof) trajectory_6dof="$2"; shift 2 ;;
    --hand-source) hand_source="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --hand-type) hand_type="$2"; shift 2 ;;
    --robot-type) robot_type="$2"; shift 2 ;;
    --dataset-name) dataset_name="$2"; shift 2 ;;
    --data-id) data_id="$2"; shift 2 ;;
    --ik-end-idx) ik_end_idx="$2"; shift 2 ;;
    --max-sim-steps) max_sim_steps="$2"; shift 2 ;;
    --num-samples) num_samples="$2"; shift 2 ;;
    --max-num-iterations) max_num_iterations="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --viewer) simulator_viewer="$2"; shift 2 ;;
    --skip-mjwp) skip_mjwp=1; shift ;;
    --check-only) check_only=1; shift ;;
    -h|--help)
      cat <<USAGE
Usage: $0 --trajectory-6dof do_as_i_do|egoinfinity --hand-source aoe|estimated [options]

Runs the official SPIDER/MJWP path:
  prepare SPIDER dataset -> generate_xml.py -> ik_fast.py -> examples/run_mjwp.py

Required assets must already live under experiments/<run>/:
  assets/trajectory_6dof/<pipeline>/<task>/source_trajectory_keypoints.npz
  assets/trajectory_6dof/<pipeline>/<task>/object_meshes/visual.obj

Environment overrides:
  SPIDER_PYTHON, SPIDER_MAX_SIM_STEPS, SPIDER_NUM_SAMPLES,
  SPIDER_MAX_NUM_ITERATIONS, SPIDER_DEVICE, SPIDER_IK_END_IDX
USAGE
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

exp="$repo_root/experiments/$run_name"
cell_key="traj_${trajectory_6dof}__hand_${hand_source}__retarget_spider"
work="$exp/intermediates/retargeting/spider/$cell_key"
dataset_dir="$work/dataset"
log="$exp/logs/retarget_spider_${cell_key}.log"

mkdir -p "$work" "$exp/logs"

echo "cell_key=$cell_key"
echo "work=$work"
echo "dataset_dir=$dataset_dir"
echo "python=$python_bin"

if [[ "$check_only" -eq 1 ]]; then
  "$python_bin" -c "import pathlib, spider, mujoco, mujoco_warp, warp; print('spider import ok', pathlib.Path(spider.__file__).resolve())"
  exit 0
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1

cd "$repo_root/third_party/SPIDER"

{
  echo "[spider] preparing AoE assets for official SPIDER/MJWP"
  "$python_bin" - <<PY
import json
import os
import shutil
from pathlib import Path

repo_root = Path("$repo_root")
exp = Path("$exp")
work = Path("$work")
dataset_dir = Path("$dataset_dir")
dataset_name = "$dataset_name"
trajectory_6dof = "$trajectory_6dof"
hand_source = "$hand_source"
hand_type = "$hand_type"
task = "$task"
data_id = int("$data_id")

def first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

source_keypoints = os.environ.get("SPIDER_SOURCE_KEYPOINTS")
object_visual = os.environ.get("SPIDER_OBJECT_VISUAL")

traj_assets = exp / "assets" / "trajectory_6dof" / trajectory_6dof / task
keypoint_candidates = [
    Path(source_keypoints) if source_keypoints else None,
    traj_assets / "source_trajectory_keypoints.npz",
    traj_assets / "trajectory_keypoints.npz",
    exp / "intermediates" / "do_as_i_do" / "retargeting_outputs" / "mano" / hand_type / task / str(data_id) / "trajectory_keypoints.npz",
]
keypoint_candidates += sorted(exp.glob(f"intermediates/retargeting/*/*/retargeting_outputs/mano/{hand_type}/{task}/{data_id}/trajectory_keypoints.npz"))

mesh_candidates = [
    Path(object_visual) if object_visual else None,
    traj_assets / "object_meshes" / "visual.obj",
    exp / "intermediates" / "do_as_i_do" / "retargeting_outputs" / "assets" / "objects" / task / "visual.obj",
]
mesh_candidates += sorted(exp.glob(f"intermediates/retargeting/*/*/retargeting_outputs/assets/objects/{task}/visual.obj"))

keypoints = first_existing(keypoint_candidates)
visual = first_existing(mesh_candidates)

missing = []
if keypoints is None:
    missing.append("SPIDER trajectory_keypoints.npz")
if visual is None:
    missing.append("SPIDER object visual.obj")
if missing:
    print(json.dumps({
        "status": "missing_inputs",
        "missing": missing,
        "searched_keypoints": [str(p) for p in keypoint_candidates if p is not None],
        "searched_meshes": [str(p) for p in mesh_candidates if p is not None],
    }, indent=2))
    raise SystemExit(4)

if dataset_dir.exists():
    shutil.rmtree(dataset_dir)

mano_dir = dataset_dir / "processed" / dataset_name / "mano" / hand_type / task / str(data_id)
mesh_dir = dataset_dir / "processed" / dataset_name / "assets" / "objects" / task
mano_dir.mkdir(parents=True, exist_ok=True)
mesh_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(keypoints, mano_dir / "trajectory_keypoints.npz")
shutil.copy2(visual, mesh_dir / "visual.obj")

object_mesh_dir = f"processed/{dataset_name}/assets/objects/{task}"
right_object_mesh_dir = object_mesh_dir if hand_type in {"right", "bimanual"} else None
left_object_mesh_dir = object_mesh_dir if hand_type in {"left", "bimanual"} else None

task_info = {
    "task": task,
    "dataset_name": dataset_name,
    "robot_type": "mano",
    "embodiment_type": hand_type,
    "data_id": data_id,
    "right_object_mesh_dir": right_object_mesh_dir,
    "right_object_convex_dir": None,
    "left_object_mesh_dir": left_object_mesh_dir,
    "left_object_convex_dir": None,
    "ref_dt": 0.02,
    "source_keypoints": str(keypoints),
    "source_visual_obj": str(visual),
    "hand_source": hand_source,
    "trajectory_6dof": trajectory_6dof,
}
(mano_dir.parent / "task_info.json").write_text(json.dumps(task_info, indent=2), encoding="utf-8")
(work / "spider_input_manifest.json").write_text(json.dumps({
    "dataset_dir": str(dataset_dir),
    "dataset_name": dataset_name,
    "task": task,
    "hand_type": hand_type,
    "data_id": data_id,
    "source_keypoints": str(keypoints),
    "source_visual_obj": str(visual),
    "task_info": str(mano_dir.parent / "task_info.json"),
}, indent=2), encoding="utf-8")
print(json.dumps({"status": "prepared", "dataset_dir": str(dataset_dir), "source_keypoints": str(keypoints), "source_visual_obj": str(visual)}, indent=2))
PY

  echo "[spider] generate_xml"
  "$python_bin" spider/preprocess/generate_xml.py \
    --dataset-dir "$dataset_dir" \
    --dataset-name "$dataset_name" \
    --robot-type "$robot_type" \
    --embodiment-type "$hand_type" \
    --task "$task" \
    --data-id "$data_id" \
    --use-visual-mesh-as-collision \
    --object-floor-collision \
    --object-object-collision \
    --no-show-viewer

  echo "[spider] ik_fast"
  "$python_bin" spider/preprocess/ik_fast.py \
    --dataset-dir "$dataset_dir" \
    --dataset-name "$dataset_name" \
    --robot-type "$robot_type" \
    --embodiment-type "$hand_type" \
    --task "$task" \
    --data-id "$data_id" \
    --start-idx 0 \
    --end-idx "$ik_end_idx" \
    --ref-dt 0.02 \
    --save-video \
    --no-show-viewer

  robot_out="$dataset_dir/processed/$dataset_name/$robot_type/$hand_type/$task/$data_id"

  if [[ "$skip_mjwp" -eq 0 ]]; then
    echo "[spider] run_mjwp"
    "$python_bin" examples/run_mjwp.py \
      dataset_dir="$dataset_dir" \
      dataset_name="$dataset_name" \
      robot_type="$robot_type" \
      embodiment_type="$hand_type" \
      task="$task" \
      data_id="$data_id" \
      device="$device" \
      max_sim_steps="$max_sim_steps" \
      save_video=true \
      show_viewer=false \
      viewer="$simulator_viewer" \
      num_samples="$num_samples" \
      max_num_iterations="$max_num_iterations"
  else
    echo "[spider] skip MJWP physics optimization by request"
  fi

  echo "[spider] indexing stable outputs"
  "$python_bin" - <<PY
import json
import shutil
from pathlib import Path

work = Path("$work")
robot_out = Path("$dataset_dir") / "processed" / "$dataset_name" / "$robot_type" / "$hand_type" / "$task" / "$data_id"
robot_stable = work / "robot"
robot_stable.mkdir(parents=True, exist_ok=True)
outputs = {}
for name in [
    "scene.xml",
    "scene_eq.xml",
    "trajectory_kinematic.npz",
    "trajectory_ikrollout.npz",
    "trajectory_mjwp.npz",
    "visualization_ik.mp4",
    "visualization_mjwp.mp4",
    "config.yaml",
]:
    candidates = [robot_out / name, robot_out.parent / name]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        outputs[name] = None
        continue
    dst = robot_stable / name
    shutil.copy2(src, dst)
    root_dst = work / name
    shutil.copy2(src, root_dst)
    outputs[name] = str(dst)
(work / "spider_output_manifest.json").write_text(json.dumps({
    "robot_out": str(robot_out),
    "stable_robot_dir": str(robot_stable),
    "outputs": outputs,
}, indent=2), encoding="utf-8")
print(json.dumps(outputs, indent=2))
PY
} 2>&1 | tee "$log"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the official Do-as-I-Do retargeting entrypoint on an AoE reconstruction clip.

Required:
  --raw-dir PATH              Do-as-I-Do reconstruction clip directory.

Common options:
  --run-name NAME             Experiment name under experiments/. Default: foundation_jar_visual_demo_v1
  --task NAME                 Retargeting task/video name. Default: foundation_jar_bimanual_leftscale
  --trajectory-6dof NAME      Matrix trajectory source for asset paths. Default: do_as_i_do
  --hand-type TYPE            right, left, bimanual, or auto. Default: bimanual
  --hand-source SOURCE        aoe or estimated. Default: estimated
  --robot-type TYPE           Default: sharpa
  --max-sim-steps N           0 uses launch.py default behavior; -1 for full. Default: 0
  --force                     Force full official chain. Use for first full run only.
  --copy                      Copy raw-dir into experiments instead of symlinking it.
  --check-only                Validate inputs and print the command without running launch.py.

Example:
  scripts/run_do_as_i_do_official_retarget.sh \
    --raw-dir /path/to/do_as_i_do_reconstruction_clip \
    --run-name foundation_jar_visual_demo_v1 \
    --task foundation_jar_bimanual_leftscale \
    --robot-type sharpa \
    --force
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${RETARGETING_PYTHON:-python}"
run_name="foundation_jar_visual_demo_v1"
task="foundation_jar_bimanual_leftscale"
trajectory_6dof="do_as_i_do"
hand_type="bimanual"
hand_source="estimated"
robot_type="sharpa"
max_sim_steps="0"
force_flag="--no-force"
materialize_mode="symlink"
check_only=0
raw_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-dir)
      raw_dir="$2"; shift 2 ;;
    --run-name)
      run_name="$2"; shift 2 ;;
    --task)
      task="$2"; shift 2 ;;
    --trajectory-6dof)
      trajectory_6dof="$2"; shift 2 ;;
    --hand-type)
      hand_type="$2"; shift 2 ;;
    --hand-source)
      hand_source="$2"; shift 2 ;;
    --robot-type)
      robot_type="$2"; shift 2 ;;
    --max-sim-steps)
      max_sim_steps="$2"; shift 2 ;;
    --force)
      force_flag="--force"; shift ;;
    --copy)
      materialize_mode="copy"; shift ;;
    --check-only)
      check_only=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$raw_dir" ]]; then
  echo "Missing --raw-dir" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$raw_dir" ]]; then
  echo "raw-dir does not exist: $raw_dir" >&2
  exit 2
fi

experiment_root="$repo_root/experiments/$run_name"
cell_key="traj_${trajectory_6dof}__hand_${hand_source}__retarget_do_as_i_do"
raw_link="$experiment_root/intermediates/trajectory_6dof/do_as_i_do/reconstruction/raw_dir"
output_root="$experiment_root/intermediates/retargeting/do_as_i_do/$cell_key/retargeting_outputs"
asset_root="$experiment_root/assets/cells/$cell_key"
launch_dir="$repo_root/third_party/do-as-i-do/retargeting"
launch_py="$launch_dir/launch.py"

if [[ ! -f "$launch_py" ]]; then
  echo "Cannot find official launch.py: $launch_py" >&2
  exit 2
fi

mkdir -p "$(dirname "$raw_link")" "$output_root" "$asset_root" "$experiment_root/logs"
if [[ "$materialize_mode" == "copy" ]]; then
  rm -rf "$raw_link"
  mkdir -p "$raw_link"
  rsync -a --delete "$raw_dir"/ "$raw_link"/
else
  rm -rf "$raw_link"
  ln -s "$raw_dir" "$raw_link"
fi

missing=0
for required in config.json gravity.json; do
  if [[ ! -f "$raw_link/$required" ]]; then
    echo "Missing required official Do-as-I-Do raw-dir file: $raw_link/$required" >&2
    missing=1
  fi
done
if ! find -L "$raw_link" -mindepth 2 -maxdepth 2 -name all_hand_meshes.npz | grep -q .; then
  echo "Missing */all_hand_meshes.npz under raw-dir: $raw_link" >&2
  missing=1
fi
if [[ ! -d "$raw_link/obj_tracking_out" ]]; then
  echo "Missing object tracking directory: $raw_link/obj_tracking_out" >&2
  missing=1
fi
if [[ ! -d "$raw_link/video_segmentation/masks" ]]; then
  echo "Missing object mesh masks directory: $raw_link/video_segmentation/masks" >&2
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "The official launch.py raw-dir must be a complete Do-as-I-Do reconstruction output directory." >&2
  echo "Linked/copied input was left at: $raw_link" >&2
  exit 3
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"

cmd=(
  "$python_bin" "$launch_py"
  --raw-dir "$raw_link"
  --task "$task"
  --hand-type "$hand_type"
  --dataset-name do_as_i_do
  --robot-type "$robot_type"
  --output-root-dir "$output_root"
  --max-sim-steps "$max_sim_steps"
  --no-show-viewer
  --no-wait-on-finish
  "$force_flag"
)

printf '%q ' "${cmd[@]}"
printf '\n'
echo "experiment_root=$experiment_root"
echo "output_root=$output_root"
echo "asset_root=$asset_root"
echo "cell_key=$cell_key"

if [[ "$check_only" -eq 1 ]]; then
  exit 0
fi

cd "$launch_dir"
"${cmd[@]}" 2>&1 | tee "$experiment_root/logs/do_as_i_do_official_retarget.log"

robot_run_dir="$output_root/$robot_type/$hand_type/$task/0"
robot_scene="$robot_run_dir/scene.xml"
robot_traj="$robot_run_dir/trajectory_mjwp.npz"
robot_video="$robot_run_dir/visualization_mjwp.mp4"
if [[ -f "$robot_scene" && -f "$robot_traj" ]]; then
  "$python_bin" "$repo_root/scripts/render_mujoco_trajectory.py" \
    --scene "$robot_scene" \
    --trajectory "$robot_traj" \
    --output "$robot_video" \
    --width "${ROBOT_RENDER_WIDTH:-640}" \
    --height "${ROBOT_RENDER_HEIGHT:-480}" \
    --fps "${ROBOT_RENDER_FPS:-25}" \
    --stride "${ROBOT_RENDER_STRIDE:-8}" \
    2>&1 | tee "$experiment_root/logs/do_as_i_do_robot_render.log"
fi

"$python_bin" "$repo_root/scripts/index_cell_assets.py" \
  --run-name "$run_name" \
  --trajectory-6dof "$trajectory_6dof" \
  --hand-source "$hand_source" \
  --retargeting do_as_i_do \
  --task "$task" \
  --hand-type "$hand_type" \
  --robot "$robot_type"

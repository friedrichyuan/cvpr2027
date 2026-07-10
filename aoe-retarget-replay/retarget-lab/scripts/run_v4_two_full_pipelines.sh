#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="${V4_RUN_NAME:-foundation_jar_v4_two_full_$(date +%Y%m%d_%H%M%S)}"
exp="$repo_root/experiments/$run_name"
python_ego="${EGOINFINITY_PYTHON:-python}"
python_ret="${RETARGETING_PYTHON:-python}"

task="${V4_TASK:-foundation_jar_bimanual_leftscale}"
hand_type="${V4_HAND_TYPE:-bimanual}"
dai_raw_dir="${V4_DAI_RAW_DIR:?Set V4_DAI_RAW_DIR to a Do-as-I-Do reconstruction raw directory.}"
dai_clip_dir="${V4_DAI_CLIP_DIR:?Set V4_DAI_CLIP_DIR to the Do-as-I-Do clip directory used for visualization.}"
ego_video="${V4_EGO_VIDEO:?Set V4_EGO_VIDEO to an AoE raw RGB video path.}"
ego_clip_id="${V4_EGO_CLIP_ID:?Set V4_EGO_CLIP_ID to a stable clip id.}"
ego_objects="${V4_EGO_OBJECTS:?Set V4_EGO_OBJECTS to the EgoInfinity object prompt.}"
ego_start="${V4_EGO_START:-0.0}"
ego_end="${V4_EGO_END:-3.5}"
ego_fps="${V4_EGO_FPS:-15}"

main_cuda="${MAIN_CUDA:-0}"
sam3_cuda="${SAM3_WORKER_CUDA:-1}"
sam3d_cuda="${SAM3D_WORKER_CUDA:-7}"
sam3_python="${SAM3_PYTHON:?Set SAM3_PYTHON to the SAM3/SAM3.1 worker Python.}"
sam3_repo="${SAM3_REPO:?Set SAM3_REPO to the SAM3 source checkout.}"
sam3_bpe_path="${SAM3_BPE_PATH:-$sam3_repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
sam3d_python="${SAM3D_PYTHON:?Set SAM3D_PYTHON to the SAM3D worker Python.}"
sam3d_repo="${SAM3D_REPO:?Set SAM3D_REPO to the SAM 3D Objects source checkout.}"
sam3d_conda_prefix="${SAM3D_CONDA_PREFIX:-$(dirname "$(dirname "$sam3d_python")")}"
dinov2_local_repo="${DINOV2_LOCAL_REPO:-$repo_root/third_party/dinov2}"
sam3_socket="${SAM3_WORKER_SOCKET:-/tmp/egoinfinity_sam3_${USER:-user}_${run_name}.sock}"
sam3d_socket="${SAM3D_WORKER_SOCKET:-/tmp/egoinfinity_sam3d_${USER:-user}_${run_name}.sock}"
sam3_pid=""
sam3d_pid=""

mkdir -p "$exp/logs/workers" "$exp/videos" "$exp/reuse"

cleanup() {
  if [[ -n "$sam3_pid" ]]; then
    kill "$sam3_pid" 2>/dev/null || true
    wait "$sam3_pid" 2>/dev/null || true
    sam3_pid=""
  fi
  if [[ -n "$sam3d_pid" ]]; then
    kill "$sam3d_pid" 2>/dev/null || true
    wait "$sam3d_pid" 2>/dev/null || true
    sam3d_pid=""
  fi
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

wait_ready() {
  local pid="$1"
  local ready="$2"
  local name="$3"
  local log="$4"
  local timeout="${5:-900}"
  for _ in $(seq 1 "$timeout"); do
    if [[ -f "$ready" ]]; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited early; see $log" >&2
      return 1
    fi
    sleep 1
  done
  echo "$name did not become ready; see $log" >&2
  return 1
}

start_workers() {
  local sam3_log="$exp/logs/workers/sam3_worker.log"
  local sam3d_log="$exp/logs/workers/sam3d_worker.log"

  rm -f "$sam3_socket" "$sam3_socket.ready"
  (
    cd "$repo_root/third_party/EgoInfinity"
    CUDA_VISIBLE_DEVICES="$sam3_cuda" \
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
      exec "$sam3_python" scripts/sam3_worker.py \
        --socket "$sam3_socket" \
        --version "${EGOINFINITY_SAM3_VERSION:-sam3.1}" \
        --bpe_path "$sam3_bpe_path"
  ) > "$sam3_log" 2>&1 &
  sam3_pid=$!

  rm -f "$sam3d_socket" "$sam3d_socket.ready"
  (
    cd "$repo_root/third_party/EgoInfinity"
    CUDA_VISIBLE_DEVICES="$sam3d_cuda" \
    CONDA_PREFIX="$sam3d_conda_prefix" \
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
    TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" \
    SAM3D_REPO="$sam3d_repo" \
    DINOV2_LOCAL_REPO="$dinov2_local_repo" \
    DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-$dinov2_local_repo}" \
      exec "$sam3d_python" scripts/sam3d_worker.py \
        --socket "$sam3d_socket" \
        --repo "$sam3d_repo" \
        --quality_default "${SAM3D_QUALITY_DEFAULT:-tier1}"
  ) > "$sam3d_log" 2>&1 &
  sam3d_pid=$!

  wait_ready "$sam3_pid" "$sam3_socket.ready" "SAM3 worker" "$sam3_log" "${SAM3_WORKER_READY_TIMEOUT:-420}"
  wait_ready "$sam3d_pid" "$sam3d_socket.ready" "SAM3D worker" "$sam3d_log" "${SAM3D_WORKER_READY_TIMEOUT:-900}"
}

compose_cell() {
  local cell="$1"
  local output="$2"
  "$python_ego" "$repo_root/scripts/compose_triptych.py" \
    --overlay "$exp/cells/$cell/overlay.mp4" \
    --depth "$exp/cells/$cell/depth.mp4" \
    --robot "$exp/cells/$cell/robot.mp4" \
    --output "$output" \
    --duration 0
}

run_egoinfinity_full() {
  local log="$exp/logs/v4_egoinfinity_full.log"
  {
    echo "=== egoinfinity_full start $(date -Is) ==="
    cd "$repo_root"
    export CUDA_VISIBLE_DEVICES="$main_cuda"
    export SAM3_WORKER_SOCKET="$sam3_socket"
    export SAM3D_WORKER_SOCKET="$sam3d_socket"
    export EGOINFINITY_OBJECT_SELECTION_MODE="${EGOINFINITY_OBJECT_SELECTION_MODE:-best}"
    export EGOINFINITY_TARGET_PROMPT="${EGOINFINITY_TARGET_PROMPT:-$ego_objects}"
    scripts/run_egoinfinity_retarget.sh \
      --clip-dir "$ego_video" \
      --clip-id "$ego_clip_id" \
      --objects "$ego_objects" \
      --start "$ego_start" \
      --end "$ego_end" \
      --fps "$ego_fps" \
      --run-name "$run_name" \
      --robot g1 \
      --no-sam3d-worker
    "$python_ego" "$repo_root/scripts/index_cell_assets.py" \
      --run-name "$run_name" \
      --trajectory-6dof egoinfinity \
      --hand-source estimated \
      --retargeting egoinfinity \
      --task "$task" \
      --hand-type "$hand_type" \
      --robot g1
    compose_cell "traj_egoinfinity__hand_estimated__retarget_egoinfinity" "$exp/videos/v4_egoinfinity_full__triptych.mp4"
    echo "=== egoinfinity_full end $(date -Is) ==="
  } 2>&1 | tee "$log"
}

run_do_as_i_do_full() {
  local log="$exp/logs/v4_do_as_i_do_full.log"
  {
    echo "=== do_as_i_do_full start $(date -Is) ==="
    cd "$repo_root"
    export CUDA_VISIBLE_DEVICES="$main_cuda"
    export DO_AS_I_DO_FAST_DECOMP="${DO_AS_I_DO_FAST_DECOMP:-0}"
    scripts/prepare_do_as_i_do_trajectory_6dof.sh \
      --run-name "$run_name" \
      --task "$task" \
      --raw-dir "$dai_raw_dir"
    "$python_ret" "$repo_root/scripts/materialize_do_as_i_do_visuals.py" \
      --run-name "$run_name" \
      --clip-dir "$dai_clip_dir" \
      --task "$task" \
      --fps "$ego_fps"
    scripts/run_do_as_i_do_official_retarget.sh \
      --raw-dir "$dai_raw_dir" \
      --run-name "$run_name" \
      --task "$task" \
      --trajectory-6dof do_as_i_do \
      --hand-type "$hand_type" \
      --hand-source estimated \
      --robot-type sharpa \
      --max-sim-steps -1
    "$python_ret" "$repo_root/scripts/index_cell_assets.py" \
      --run-name "$run_name" \
      --trajectory-6dof do_as_i_do \
      --hand-source estimated \
      --retargeting do_as_i_do \
      --task "$task" \
      --hand-type "$hand_type" \
      --robot sharpa
    compose_cell "traj_do_as_i_do__hand_estimated__retarget_do_as_i_do" "$exp/videos/v4_do_as_i_do_full__triptych.mp4"
    echo "=== do_as_i_do_full end $(date -Is) ==="
  } 2>&1 | tee "$log"
}

{
  echo "run_name=$run_name"
  echo "task=$task"
  echo "hand_type=$hand_type"
  echo "dai_raw_dir=$dai_raw_dir"
  echo "dai_clip_dir=$dai_clip_dir"
  echo "ego_video=$ego_video"
  echo "ego_clip_id=$ego_clip_id"
  echo "ego_objects=$ego_objects"
  echo "ego_start=$ego_start"
  echo "ego_end=$ego_end"
  echo "ego_fps=$ego_fps"
  echo "MAIN_CUDA=$main_cuda"
  echo "SAM3_WORKER_CUDA=$sam3_cuda"
  echo "SAM3D_WORKER_CUDA=$sam3d_cuda"
  echo "SAM3_WORKER_SOCKET=$sam3_socket"
  echo "SAM3D_WORKER_SOCKET=$sam3d_socket"
  echo "DO_AS_I_DO_FAST_DECOMP=${DO_AS_I_DO_FAST_DECOMP:-0}"
  echo "CELL_TIMEOUT_SEC=disabled"
} > "$exp/run_env.txt"

start_workers
run_egoinfinity_full
run_do_as_i_do_full

cat > "$exp/reuse/reuse_manifest.json" <<EOF
{
  "run_name": "$run_name",
  "policy": "v4 computes EgoInfinity once and Do-as-I-Do once. Downstream demos should reuse these outputs instead of invoking the original pipelines per matrix cell.",
  "egoinfinity_full": {
    "cell": "traj_egoinfinity__hand_estimated__retarget_egoinfinity",
    "pipeline_result": "intermediates/egoinfinity/clip/pipeline_result.pkl.gz",
    "triptych": "videos/v4_egoinfinity_full__triptych.mp4"
  },
  "do_as_i_do_full": {
    "cell": "traj_do_as_i_do__hand_estimated__retarget_do_as_i_do",
    "retargeting_outputs": "intermediates/retargeting/do_as_i_do/traj_do_as_i_do__hand_estimated__retarget_do_as_i_do/retargeting_outputs",
    "triptych": "videos/v4_do_as_i_do_full__triptych.mp4"
  }
}
EOF

echo "v4_done=$exp"

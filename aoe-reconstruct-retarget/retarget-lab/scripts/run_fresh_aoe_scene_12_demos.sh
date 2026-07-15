#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_ego="${EGOINFINITY_PYTHON:-/mnt/nas/share/home/hjd/repro/conda_envs/egoinfinity/bin/python}"
python_ret="${RETARGETING_PYTHON:-/mnt/nas/share/home/hjd/repro/conda_envs/retargeting/bin/python}"
spider_python="${SPIDER_PYTHON:-/mnt/nas/share/home/hjd/repro/conda_envs/retargeting/bin/python}"

dataset_root="/mnt/nas/share/home/hjd/repro/datasets/openaoe-3hours/extracted/poc_deliver"
scene=""
segment=""
annotation_id=""
object_name=""
ego_objects=""
task=""
hand_type=""
anchor_hand=""
source_run=""
matrix_run=""
duration_sec="0"
extra_sec_after="0"
scale_width="1280"
main_cuda="${MAIN_CUDA:-5}"
sam3_cuda="${SAM3_WORKER_CUDA:-5}"
sam3d_cuda="${SAM3D_WORKER_CUDA:-6}"
sam3d_min_free_mb="${SAM3D_MIN_FREE_MB:-19000}"
sam3d_wait_for_gpu="${SAM3D_WAIT_FOR_GPU:-1}"
sam3d_wait_timeout_sec="${SAM3D_WAIT_TIMEOUT_SEC:-0}"
sam3d_wait_poll_sec="${SAM3D_WAIT_POLL_SEC:-60}"
spider_cuda_visible_devices="${SPIDER_CUDA_VISIBLE_DEVICES:-6}"
spider_device="${SPIDER_DEVICE:-cuda:0}"
spider_max_sim_steps="${SPIDER_MAX_SIM_STEPS:--1}"
spider_num_samples="${SPIDER_NUM_SAMPLES:-1024}"
spider_max_num_iterations="${SPIDER_MAX_NUM_ITERATIONS:-16}"
run_spider="${FULL12_RUN_SPIDER:-all}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_fresh_aoe_scene_12_demos.sh --scene NAME --segment SEG --annotation-id ID \
    --object-name TEXT --task TASK --hand-type left|right|bimanual --anchor-hand left|right|bimanual

This creates a clean source run from AoE raw RGB, runs fresh Do-as-I-Do
reconstruction, then runs the two full pipelines and expands 12 triptych demos.
No old source-run or old pipeline output is accepted as input.
EOF
}

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) need_value "$@"; scene="$2"; shift 2 ;;
    --segment) need_value "$@"; segment="$2"; shift 2 ;;
    --annotation-id) need_value "$@"; annotation_id="$2"; shift 2 ;;
    --object-name) need_value "$@"; object_name="$2"; shift 2 ;;
    --ego-objects) need_value "$@"; ego_objects="$2"; shift 2 ;;
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --hand-type) need_value "$@"; hand_type="$2"; shift 2 ;;
    --anchor-hand) need_value "$@"; anchor_hand="$2"; shift 2 ;;
    --source-run) need_value "$@"; source_run="$2"; shift 2 ;;
    --matrix-run) need_value "$@"; matrix_run="$2"; shift 2 ;;
    --duration-sec) need_value "$@"; duration_sec="$2"; shift 2 ;;
    --extra-sec-after) need_value "$@"; extra_sec_after="$2"; shift 2 ;;
    --scale-width) need_value "$@"; scale_width="$2"; shift 2 ;;
    --main-cuda) need_value "$@"; main_cuda="$2"; shift 2 ;;
    --sam3-cuda) need_value "$@"; sam3_cuda="$2"; shift 2 ;;
    --sam3d-cuda) need_value "$@"; sam3d_cuda="$2"; shift 2 ;;
    --sam3d-min-free-mb) need_value "$@"; sam3d_min_free_mb="$2"; shift 2 ;;
    --sam3d-wait-for-gpu) sam3d_wait_for_gpu="1"; shift ;;
    --no-sam3d-wait-for-gpu) sam3d_wait_for_gpu="0"; shift ;;
    --sam3d-wait-timeout-sec) need_value "$@"; sam3d_wait_timeout_sec="$2"; shift 2 ;;
    --sam3d-wait-poll-sec) need_value "$@"; sam3d_wait_poll_sec="$2"; shift 2 ;;
    --spider-cuda-visible-devices) need_value "$@"; spider_cuda_visible_devices="$2"; shift 2 ;;
    --spider-device) need_value "$@"; spider_device="$2"; shift 2 ;;
    --spider-max-sim-steps) need_value "$@"; spider_max_sim_steps="$2"; shift 2 ;;
    --spider-num-samples) need_value "$@"; spider_num_samples="$2"; shift 2 ;;
    --spider-max-num-iterations) need_value "$@"; spider_max_num_iterations="$2"; shift 2 ;;
    --run-spider) need_value "$@"; run_spider="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$scene" ]] || { echo "missing --scene" >&2; exit 2; }
[[ -n "$segment" ]] || { echo "missing --segment" >&2; exit 2; }
[[ -n "$annotation_id" ]] || { echo "missing --annotation-id" >&2; exit 2; }
[[ -n "$object_name" ]] || { echo "missing --object-name" >&2; exit 2; }
[[ -n "$task" ]] || { echo "missing --task" >&2; exit 2; }
[[ -n "$hand_type" ]] || { echo "missing --hand-type" >&2; exit 2; }
[[ -n "$anchor_hand" ]] || { echo "missing --anchor-hand" >&2; exit 2; }
if [[ -z "$ego_objects" ]]; then
  ego_objects="$object_name"
fi

date_tag="$(date +%Y%m%d)"
if [[ -z "$source_run" ]]; then
  source_run="${scene}_full_source_${date_tag}_fresh01"
fi
if [[ -z "$matrix_run" ]]; then
  matrix_run="${scene}_matrix12_${date_tag}_fresh01"
fi

exp="$repo_root/experiments/$source_run"
matrix_exp="$repo_root/experiments/$matrix_run"
dai_clip="$exp/intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip"
raw_video="$dataset_root/$segment/ego_process/ego_undistorted_video/raw_video_undistorted.mp4"
ego_clip_id="${segment}__a${annotation_id}__$(echo "$task" | tr ' ' '_')"
plan="$repo_root/experiments/${source_run}_fresh_plan.txt"

if [[ -e "$exp" ]]; then
  echo "Refusing to reuse existing source run: $exp" >&2
  exit 3
fi
if [[ -e "$matrix_exp" ]]; then
  echo "Refusing to reuse existing matrix run: $matrix_exp" >&2
  exit 3
fi

mkdir -p "$exp/logs" "$matrix_exp/logs"
{
  echo "scene=$scene"
  echo "source_run=$source_run"
  echo "matrix_run=$matrix_run"
  echo "dataset_root=$dataset_root"
  echo "segment=$segment"
  echo "annotation_id=$annotation_id"
  echo "raw_rgb=$raw_video"
  echo "dai_fresh_clip=$dai_clip"
  echo "egoinfinity_output=$exp/intermediates/egoinfinity/clip"
  echo "do_as_i_do_reconstruction_output=$exp/intermediates/trajectory_6dof/do_as_i_do/reconstruction"
  echo "do_as_i_do_retargeting_output=$exp/intermediates/retargeting/do_as_i_do"
  echo "spider_output=$matrix_exp/intermediates/retargeting/spider"
  echo "object_name=$object_name"
  echo "ego_objects=$ego_objects"
  echo "task=$task"
  echo "hand_type=$hand_type"
  echo "anchor_hand=$anchor_hand"
  echo "main_cuda=$main_cuda"
  echo "sam3_cuda=$sam3_cuda"
  echo "sam3d_cuda=$sam3d_cuda"
  echo "sam3d_min_free_mb=$sam3d_min_free_mb"
  echo "sam3d_wait_for_gpu=$sam3d_wait_for_gpu"
  echo "sam3d_wait_timeout_sec=$sam3d_wait_timeout_sec"
  echo "sam3d_wait_poll_sec=$sam3d_wait_poll_sec"
  echo "spider_cuda_visible_devices=$spider_cuda_visible_devices"
  echo "run_spider=$run_spider"
} | tee "$plan"

echo "=== Preflight old-output guard ===" | tee "$exp/logs/softlink_check_pre.log"
find "$exp" -type l -ls 2>/dev/null | tee -a "$exp/logs/softlink_check_pre.log"

"$python_ret" "$repo_root/scripts/prepare_fresh_aoe_clip.py" \
  --dataset-root "$dataset_root" \
  --segment "$segment" \
  --annotation-id "$annotation_id" \
  --run-name "$source_run" \
  --task "$task" \
  --object-name "$object_name" \
  --anchor-hand "$anchor_hand" \
  --duration-sec "$duration_sec" \
  --extra-sec-after "$extra_sec_after" \
  --scale-width "$scale_width" \
  2>&1 | tee "$exp/logs/prepare_fresh_aoe_clip.log"

ref_frame="$("$python_ret" - <<PY
import json
from pathlib import Path
print(json.loads(Path("$dai_clip/metadata.json").read_text())["ref_frame_in_clip"])
PY
)"
ego_start="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
print(meta["start_sec"])
PY
)"
ego_end="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
print(meta["start_sec"] + meta["duration"])
PY
)"
sam3_bbox="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
bbox=(meta.get("action") or {}).get("bbox") or []
print(",".join(str(int(x)) for x in bbox))
PY
)"
sam3_point="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
bbox=(meta.get("action") or {}).get("bbox") or []
if len(bbox) == 4:
    print(f"{(float(bbox[0]) + float(bbox[2])) * 0.5:.1f},{(float(bbox[1]) + float(bbox[3])) * 0.5:.1f}")
else:
    print("")
PY
)"
echo "sam3_target_bbox=${sam3_bbox:-<none>}" | tee -a "$plan"
echo "sam3_prompt_points=${sam3_point:-<none>}" | tee -a "$plan"

SAM3_PROMPT_POINTS="$sam3_point" \
SAM3_PROMPT_POINT_LABELS="1" \
SAM3_TARGET_BBOX="$sam3_bbox" \
SAM3_CUDA="$sam3_cuda" \
SAM3D_CUDA="$sam3d_cuda" \
SAM3D_MIN_FREE_MB="$sam3d_min_free_mb" \
SAM3D_WAIT_FOR_GPU="$sam3d_wait_for_gpu" \
SAM3D_WAIT_TIMEOUT_SEC="$sam3d_wait_timeout_sec" \
SAM3D_WAIT_POLL_SEC="$sam3d_wait_poll_sec" \
CUDA_VISIBLE_DEVICES="$main_cuda" \
  "$repo_root/scripts/run_do_as_i_do_reconstruction_fresh.sh" \
    "$dai_clip/raw.mp4" "$ref_frame" "$object_name" "$anchor_hand" \
  2>&1 | tee "$exp/logs/do_as_i_do_fresh_reconstruction.log"

FULL12_PYTHON="$python_ret" \
RETARGETING_PYTHON="$python_ret" \
EGOINFINITY_PYTHON="$python_ego" \
MAIN_CUDA="$main_cuda" \
SAM3_WORKER_CUDA="$sam3_cuda" \
SAM3D_WORKER_CUDA="$sam3d_cuda" \
SPIDER_PYTHON="$spider_python" \
SPIDER_CUDA_VISIBLE_DEVICES="$spider_cuda_visible_devices" \
SPIDER_DEVICE="$spider_device" \
SPIDER_MAX_SIM_STEPS="$spider_max_sim_steps" \
SPIDER_NUM_SAMPLES="$spider_num_samples" \
SPIDER_MAX_NUM_ITERATIONS="$spider_max_num_iterations" \
EGOINFINITY_OBJECT_SELECTION_MODE="${EGOINFINITY_OBJECT_SELECTION_MODE:-all}" \
  "$repo_root/scripts/run_full_12_demos.sh" \
    --run-name "$source_run" \
    --matrix-run-name "$matrix_run" \
    --task "$task" \
    --hand-type "$hand_type" \
    --ego-video "$raw_video" \
    --ego-clip-id "$ego_clip_id" \
    --ego-objects "$ego_objects" \
    --ego-start "$ego_start" \
    --ego-end "$ego_end" \
    --ego-fps "15" \
    --dai-raw-dir "$dai_clip" \
    --dai-clip-dir "$dai_clip" \
    --run-spider "$run_spider" \
    --no-spider-fallback \
    --force-spider \
    --duration 0 \
  2>&1 | tee "$exp/logs/run_full_12_demos_fresh.log"

{
  echo "=== readlink -f EgoInfinity clip ==="
  readlink -f "$exp/intermediates/egoinfinity/clip" || true
  echo "=== source run symlinks ==="
  find "$exp" -type l -ls || true
  echo "=== banned-link scan ==="
  if find "$exp" -type l -print0 | xargs -0 -r readlink -f | grep -E "experiments/.*20260629|openaoe-method-pilots/egoinfinity_native_aoe"; then
    echo "BANNED_SOURCE_LINK_FOUND"
    exit 9
  fi
  echo "OK: no source symlink resolves to old 20260629 outputs or openaoe-method-pilots EgoInfinity outputs"
} | tee "$exp/logs/softlink_check_post.log"

echo "source_run=$exp"
echo "matrix_run=$matrix_exp"
echo "videos=$matrix_exp/videos"
echo "manifest=$matrix_exp/reuse_12_demo_manifest.json"

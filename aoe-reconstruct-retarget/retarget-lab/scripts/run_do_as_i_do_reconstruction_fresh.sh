#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$repo_root/third_party/do-as-i-do/reconstruction/scripts"
recon_root="${DAI_RECON_ROOT:-$repo_root/third_party/do-as-i-do/reconstruction}"
module_root="${DAI_RECON_MODULE_ROOT:-$recon_root/modules}"
if [[ ! -d "$module_root/Fast-SAM3D" ]]; then
  module_root="${DAI_RECON_MODULE_ROOT_FALLBACK:-/mnt/nas/share/home/hjd/repro/openaoe-method-pilots/external/do-as-i-do/reconstruction/modules}"
fi

sam3d_env="${SAM3D_CONDA_PREFIX:-/mnt/nas/share/home/hjd/repro/conda_envs/sam3d}"
py="${SAM3D_PYTHON:-$sam3d_env/bin/python}"
sam3_env="${SAM3_ENV:-/mnt/nas/share/home/hjd/repro/conda_envs/sam3}"
sam3_py="${SAM3_PYTHON:-$sam3_env/bin/python}"
default_cuda="${CUDA_VISIBLE_DEVICES:-0}"
sam3_cuda="${SAM3_CUDA:-${SAM3_WORKER_CUDA:-$default_cuda}}"
sam3d_cuda="${SAM3D_CUDA:-${SAM3D_WORKER_CUDA:-$default_cuda}}"
sam3d_min_free_mb="${SAM3D_MIN_FREE_MB:-0}"
sam3d_wait_for_gpu="${SAM3D_WAIT_FOR_GPU:-0}"
sam3d_wait_timeout_sec="${SAM3D_WAIT_TIMEOUT_SEC:-0}"
sam3d_wait_poll_sec="${SAM3D_WAIT_POLL_SEC:-60}"

video_path="${1:?usage: run_do_as_i_do_reconstruction_fresh.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND}"
frame_n="${2:?usage: run_do_as_i_do_reconstruction_fresh.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND}"
object_name="${3:?usage: run_do_as_i_do_reconstruction_fresh.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND}"
anchor_hand="${4:?usage: run_do_as_i_do_reconstruction_fresh.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND}"
object_id="${object_name// /_}"

video_path="$(realpath "$video_path")"
video_dir="$(dirname "$video_path")"
video_name="$(basename "$video_path")"
video_name="${video_name%%.*}"
frame_path="$video_dir/$(printf "%04d.png" "$frame_n")"
pointmap_path="$video_dir/$(printf "%04d_pointmap.npy" "$frame_n")"
masks_dir="$video_dir/video_segmentation/masks/frame_$(printf "%06d" "$frame_n")_masks"
video_masks_dir="$video_dir/video_segmentation/masks"
hand_meshes_path="$video_dir/$video_name/all_hand_meshes.npz"

export RECON_ROOT="$recon_root"
export SAM3D_DIR="${SAM3D_DIR:-$module_root/sam-3d-objects}"
export FASTSAM3D_DIR="${FASTSAM3D_DIR:-$module_root/Fast-SAM3D}"
export HAWOR_DIR="${HAWOR_DIR:-$module_root/HaWoR}"
export TAPNET_DIR="${TAPNET_DIR:-$module_root/tapnet}"
export SAM3_PKG_DIR="${SAM3_PKG_DIR:-$module_root/sam3}"
export SAM3_VERSION="${SAM3_VERSION:-sam3}"
export SCRIPTS_DIR="$script_dir"
export SAM3D_REPO_ROOT="$FASTSAM3D_DIR"
export TAPNET_CKPT="${TAPNET_CKPT:-$recon_root/weights/tapnet/bootstapir_checkpoint_v2.pt}"
if [[ ! -f "$TAPNET_CKPT" ]]; then
  export TAPNET_CKPT="/mnt/nas/share/home/hjd/repro/openaoe-method-pilots/external/do-as-i-do/reconstruction/weights/tapnet/bootstapir_checkpoint_v2.pt"
fi
export CUDA_VISIBLE_DEVICES="$sam3d_cuda"
export CONDA_PREFIX="$sam3d_env"
export CUDA_HOME="$sam3d_env"
export HF_HOME="${HF_HOME:-/mnt/nas/share/home/hjd/repro/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCH_HOME="${TORCH_HOME:-/mnt/nas/share/home/hjd/repro/torch_cache}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$script_dir/shims:$FASTSAM3D_DIR:$SAM3D_DIR:$HAWOR_DIR:$TAPNET_DIR:$SAM3_PKG_DIR:$script_dir"

run_sam3_py() {
  CUDA_VISIBLE_DEVICES="$sam3_cuda" "$sam3_py" "$@"
}

run_sam3d_py() {
  CUDA_VISIBLE_DEVICES="$sam3d_cuda" "$py" "$@"
}

select_sam3d_gpu_if_needed() {
  [[ "$sam3d_min_free_mb" =~ ^[0-9]+$ ]] || {
    echo "error: SAM3D_MIN_FREE_MB must be an integer, got '$sam3d_min_free_mb'" >&2
    exit 2
  }
  if (( sam3d_min_free_mb <= 0 )); then
    return 0
  fi
  if [[ "${SAM3D_SKIP_FREE_MEM_CHECK:-0}" == "1" ]]; then
    echo "[gpu-check] SAM3D free-memory check skipped by SAM3D_SKIP_FREE_MEM_CHECK=1"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-check] nvidia-smi is unavailable; continuing without SAM3D free-memory check"
    return 0
  fi

  local candidates="$sam3d_cuda"
  local start_ts now_ts elapsed best_gpu best_free
  start_ts="$(date +%s)"

  while true; do
    best_gpu=""
    best_free="-1"
    IFS=',' read -r -a gpu_list <<< "$candidates"
    for gpu in "${gpu_list[@]}"; do
      gpu="$(echo "$gpu" | tr -d '[:space:]')"
      [[ -n "$gpu" ]] || continue
      local free_mb
      free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
      [[ "$free_mb" =~ ^[0-9]+$ ]] || {
        echo "[gpu-check] GPU $gpu free memory unavailable"
        continue
      }
      echo "[gpu-check] GPU $gpu free=${free_mb}MiB required=${sam3d_min_free_mb}MiB"
      if (( free_mb > best_free )); then
        best_free="$free_mb"
        best_gpu="$gpu"
      fi
      if (( free_mb >= sam3d_min_free_mb )); then
        sam3d_cuda="$gpu"
        export CUDA_VISIBLE_DEVICES="$sam3d_cuda"
        echo "[gpu-check] selected GPU $sam3d_cuda for SAM3D mesh decode"
        return 0
      fi
    done

    if [[ "$sam3d_wait_for_gpu" != "1" ]]; then
      echo "error: no SAM3D candidate GPU has ${sam3d_min_free_mb}MiB free; best GPU ${best_gpu:-<none>} has ${best_free}MiB. Set SAM3D_WAIT_FOR_GPU=1 to wait." >&2
      exit 86
    fi

    if (( sam3d_wait_timeout_sec > 0 )); then
      now_ts="$(date +%s)"
      elapsed=$((now_ts - start_ts))
      if (( elapsed >= sam3d_wait_timeout_sec )); then
        echo "error: timed out after ${sam3d_wait_timeout_sec}s waiting for SAM3D GPU memory; best GPU ${best_gpu:-<none>} has ${best_free}MiB." >&2
        exit 86
      fi
    fi
    echo "[gpu-check] waiting ${sam3d_wait_poll_sec}s for SAM3D GPU memory; best GPU ${best_gpu:-<none>} has ${best_free}MiB"
    sleep "$sam3d_wait_poll_sec"
  done
}

echo "=== Do-as-I-Do fresh reconstruction provenance ==="
echo "repo_root=$repo_root"
echo "video_path=$video_path"
echo "video_dir=$video_dir"
echo "recon_root=$recon_root"
echo "module_root=$module_root"
echo "scripts_dir=$script_dir"
echo "object_name=$object_name"
echo "object_id=$object_id"
echo "frame_n=$frame_n"
echo "anchor_hand=$anchor_hand"
echo "default_cuda=$default_cuda"
echo "sam3_cuda=$sam3_cuda"
echo "sam3d_cuda=$sam3d_cuda"
echo "SAM3D_MIN_FREE_MB=$sam3d_min_free_mb"
echo "SAM3D_WAIT_FOR_GPU=$sam3d_wait_for_gpu"
echo "SAM3D_WAIT_TIMEOUT_SEC=$sam3d_wait_timeout_sec"
echo "SAM3D_WAIT_POLL_SEC=$sam3d_wait_poll_sec"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

echo "=== [0] Extracting frames ==="
mkdir -p "$video_dir/all_frames"
ffmpeg -hide_banner -loglevel error -y -i "$video_path" -vsync 0 -start_number 0 "$video_dir/all_frames/%06d.png"
ffmpeg -hide_banner -loglevel error -y -i "$video_path" -vf "select=eq(n\\,$frame_n)" -vsync 0 -vframes 1 "$frame_path"

echo "=== [1] SAM3 video masks before SAM3D ==="
sam3_args=(
  --video "$video_dir/all_frames"
  --obj_id "$object_id"
  --frame_idx "$frame_n"
  --output_dir "$video_dir/video_segmentation"
  --version "$SAM3_VERSION"
  --min-valid-ratio "${SAM3_MIN_VALID_RATIO:-0.70}"
  --max-centroid-jump-px "${SAM3_MAX_CENTROID_JUMP_PX:-320}"
)
if [[ -n "${SAM3_PROMPT_POINTS:-}" ]]; then
  sam3_args+=(--points "$SAM3_PROMPT_POINTS" --point_labels "${SAM3_PROMPT_POINT_LABELS:-1}")
else
  sam3_args+=(--text "$object_name")
fi
if [[ -n "${SAM3_TARGET_POINT:-}" ]]; then
  sam3_args+=(--target-point "$SAM3_TARGET_POINT" --select-mode point)
fi
if [[ -n "${SAM3_TARGET_BBOX:-}" ]]; then
  sam3_args+=(--target-bbox "$SAM3_TARGET_BBOX" --select-mode bbox)
fi
run_sam3_py "$script_dir/run_sam3_video.py" "${sam3_args[@]}"

echo "=== [2a] SAM3D masks -> object mesh ==="
select_sam3d_gpu_if_needed
echo "SAM3D_SELECTED_CUDA=$sam3d_cuda"
cd "$SAM3D_DIR"
run_sam3d_py generate_mesh_sam3d.py \
  --image_path "$frame_path" \
  --masks_dir "$masks_dir" \
  --config "$SAM3D_DIR/checkpoints/hf/pipeline_mesh_only.yaml" \
  --with_texture_baking False \
  --use_vertex_color True

echo "=== [2b] MoGe pointmap for reference frame ==="
cd "$script_dir"
run_sam3d_py get_pointmap_dir.py --image "$frame_path" --output "$pointmap_path"

echo "=== [2c] Export AoE MANO hands as Do-as-I-Do hand meshes ==="
run_sam3d_py export_aoe_hands_to_dai_npz.py \
  --clip-dir "$video_dir" \
  --metadata "$video_dir/metadata.json" \
  --hawor-dir "$HAWOR_DIR" \
  --output "$hand_meshes_path" \
  --pose-space camera \
  --device cpu

echo "=== [2d] MoGe pointmaps for all frames ==="
run_sam3d_py get_pointmap_dir.py --image_dir "$video_dir/all_frames"

echo "=== [2d.1] Render AoE hand masks for scale optimization ==="
frame_count="$(find "$video_dir/all_frames" -maxdepth 1 -name '[0-9][0-9][0-9][0-9][0-9][0-9].png' | wc -l | tr -d '[:space:]')"
run_sam3d_py render_aoe_hand_masks_for_dai.py \
  --clip-dir "$video_dir" \
  --hand-meshes "$hand_meshes_path" \
  --frames "0:$frame_count" \
  --dilate "${DAI_HAND_MASK_DILATE:-3}"

echo "=== [2e] Gravity estimation ==="
if ! run_sam3d_py predict_video_gravity.py "$video_dir/all_frames" --output_path "$video_dir/gravity.json"; then
  cat > "$video_dir/gravity.json" <<'JSON'
{
  "vec3d": [0.0, 0.0, 1.0],
  "roll_deg": 0.0,
  "pitch_deg": 0.0,
  "note": "identity gravity shim because GeoCalib is unavailable"
}
JSON
fi

echo "=== [2.5] TAPIR velocity tracking ==="
run_sam3d_py tapir_velocity_tracking.py \
  --video "$video_path" \
  --mask-dir "$video_masks_dir" \
  --object "$object_id" \
  --checkpoint "$TAPNET_CKPT"

echo "=== [3] Fast-SAM3D guided pose tracking ==="
cd "$FASTSAM3D_DIR"
run_sam3d_py track_object.py \
  --config checkpoints/hf/pipeline.yaml \
  --vid_dir "$video_dir" \
  --masks_root "$video_masks_dir" \
  --object_name "$object_id" \
  --init_frame "$frame_n" \
  --output_dir "$video_dir/obj_tracking_out/$object_id" \
  --guidance_strength 1 \
  --save_layout \
  --fix_scale_to_init_frame \
  --pose_guidance_strength 0.5 \
  --num_pose_samples 8 \
  --scoring_metric render_iou \
  --pose_selection cluster \
  --cluster_dist_thresh 0.3 \
  --cluster_min_size 2 \
  --cluster_w_rot 1.5 \
  --chain_poses \
  --post_optimize \
  --no-enable_shape_icp \
  --chain_on_diffusion \
  --enable_ss_cache \
  --euler_steps 8 \
  --rotvel_json "$video_dir/perframe_tracking_$object_id/motion_stats.json"

cd "$script_dir"
echo "=== [3b] Project mesh ==="
run_sam3d_py run_project_mesh_combined.py \
  --video "$video_path" \
  --mesh "$masks_dir/$object_id/$object_id.obj" \
  --json "$video_dir/obj_tracking_out/$object_id/combined_visualization/layout.json" \
  --output-base "$video_dir/obj_tracking_out/$object_id/combined_visualization/projected"

echo "=== [3c] Convert layout to camera frame ==="
run_sam3d_py convert_layout_to_camera_frame.py \
  --input "$video_dir/obj_tracking_out/$object_id/combined_visualization/layout.json" \
  --output "$video_dir/obj_tracking_out/$object_id/combined_visualization/layout_camera_frame.json"

echo "=== [4] Optimize translation/scale ==="
run_sam3d_py optimize_translation_scale.py \
  --video-dir "$video_dir" \
  --layout-json "$video_dir/obj_tracking_out/$object_id/combined_visualization/layout_camera_frame.json" \
  --hand-meshes "$hand_meshes_path" \
  --anchor-hand "$anchor_hand" \
  --ref-frame "$frame_n"

cat <<EOF
=== Pipeline complete ===
layout: $video_dir/obj_tracking_out/$object_id/combined_visualization/layout_camera_frame_optimized.json
mesh:   $masks_dir/$object_id/$object_id.obj
hands:  $hand_meshes_path
EOF

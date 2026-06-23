#!/bin/bash
# smoke_e2e.sh — One-click AoE → GR00T N1.7 end-to-end pipeline
#
# Covers: data conversion → merge → validate → visualize → pretrain
# Mirrors the README Quick Start exactly.
#
# Usage:
#   bash scripts/smoke_e2e.sh
#   # Or override defaults:
#   bash scripts/smoke_e2e.sh --mode gripper
#
set -euo pipefail

# ============================================================
# Configuration (edit here or override via CLI flags)
# ============================================================
VENV="/media/hdd4tb/sankuai/code/07_具身/01_vla/Isaac-GR00T/.venv"
DATA_ROOT="/media/hdd4tb/sankuai/code/07_具身/dataset/07_Ego/poc_deliver"
OUTPUT="output"
BASE_MODEL="/media/hdd4tb/sankuai/code/07_具身/dataset/02_开源模型/GR00T-N1.7-3B"
VLM_MODEL_NAME="/media/hdd4tb/sankuai/code/07_具身/dataset/02_开源模型/Cosmos-Reason2-2B"

MODE="gripper" # sharpa | gripper
TRAIN_EPISODES=50
VIZ_EPISODES=3
NUM_GPUS=4
BATCH_SIZE=4 # 32
DEEPSPEED_STAGE=3
GRADIENT_CHECKPOINTING=true
MAX_STEPS=10000 # 1000 | 20000
LR=1e-4

# ============================================================
# CLI overrides
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)         MODE="$2"; shift 2 ;;
        --data-root)    DATA_ROOT="$2"; shift 2 ;;
        --output)       OUTPUT="$2"; shift 2 ;;
        --base-model)   BASE_MODEL="$2"; shift 2 ;;
        --vlm-model)    VLM_MODEL_NAME="$2"; shift 2 ;;
        --train-episodes) TRAIN_EPISODES="$2"; shift 2 ;;
        --viz-episodes) VIZ_EPISODES="$2"; shift 2 ;;
        --num-gpus)     NUM_GPUS="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --max-steps)    MAX_STEPS="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate venv if configured
if [ -n "${VENV}" ]; then
    if [ -f "${VENV}/bin/activate" ]; then
        source "${VENV}/bin/activate"
        echo "Activated venv: ${VENV}"
    else
        echo "ERROR: venv not found at ${VENV}/bin/activate"; exit 1
    fi
fi
PYTHON="${PYTHON:-python}"

# ============================================================
# Preflight checks
# ============================================================
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           AoE → GR00T N1.7  End-to-End Pipeline          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Mode:           ${MODE}"
echo "  Data root:      ${DATA_ROOT}"
echo "  Output:         ${OUTPUT}"
echo "  Base model:     ${BASE_MODEL}"
echo "  VLM backbone:   ${VLM_MODEL_NAME}"
echo "  Train episodes: ${TRAIN_EPISODES}"
echo "  Viz episodes:   ${VIZ_EPISODES}"
echo "  GPUs:           ${NUM_GPUS}"
echo "  Batch size:     ${BATCH_SIZE}"
echo "  Max steps:      ${MAX_STEPS}"
echo "  Learning rate:  ${LR}"
echo ""

for d in "${DATA_ROOT}" "${BASE_MODEL}" "${VLM_MODEL_NAME}"; do
    if [ ! -d "${d}" ]; then
        echo "ERROR: Directory not found: ${d}"; exit 1
    fi
done

cd "${PROJECT_ROOT}"

# ============================================================
# Step 1: Data Conversion
# ============================================================
echo ""
echo "━━━ Step 1a: MANO FK → 21 keypoints ━━━"
${PYTHON} scripts/convert_mano_to_keypoints.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT}" \
    --max-episodes "${TRAIN_EPISODES}"

echo ""
echo "━━━ Step 1b: Keypoints → ${MODE} retarget ━━━"
if [ "${MODE}" = "sharpa" ]; then
    ${PYTHON} scripts/retarget_to_sharpa.py \
        --data-root "${DATA_ROOT}" \
        --keypoints-dir "${OUTPUT}" \
        --output-dir "${OUTPUT}" \
        --max-episodes "${TRAIN_EPISODES}"
else
    ${PYTHON} scripts/retarget_to_gripper.py \
        --data-root "${DATA_ROOT}" \
        --keypoints-dir "${OUTPUT}" \
        --output-dir "${OUTPUT}" \
        --max-episodes "${TRAIN_EPISODES}"
fi

echo ""
echo "━━━ Step 1c: Assemble → LeRobot V2.1 ━━━"
${PYTHON} scripts/convert_ego_to_lerobot.py \
    --mode "${MODE}" \
    --data-root "${DATA_ROOT}" \
    --retarget-dir "${OUTPUT}" \
    --output-dir "${OUTPUT}" \
    --max-episodes "${TRAIN_EPISODES}"

# ============================================================
# Step 2: Merge
# ============================================================
echo ""
echo "━━━ Step 2: Merge per-episode datasets ━━━"
MERGED="${OUTPUT}/ego_${MODE}_merged"
if [ -d "${MERGED}" ]; then
    rm -rf "${MERGED}"
fi
${PYTHON} scripts/merge_lerobot_datasets.py \
    --input-dir "${OUTPUT}" \
    --mode "${MODE}" \
    --output-dir "${MERGED}"

# ============================================================
# Step 3: Validate
# ============================================================
echo ""
echo "━━━ Step 3: Validate merged dataset ━━━"
${PYTHON} scripts/validate_dataset.py "${MERGED}"

# ============================================================
# Step 4: Visualize (head 3 episodes)
# ============================================================
echo ""
echo "━━━ Step 4: Visualize retarget overlay (${VIZ_EPISODES} episodes) ━━━"
if [ "${MODE}" = "sharpa" ]; then
    ${PYTHON} scripts/visualize_sharpa_overlay_3d.py \
        --data-root "${DATA_ROOT}" \
        --keypoints-dir "${OUTPUT}" \
        --retarget-dir "${OUTPUT}" \
        --output-dir "${OUTPUT}" \
        --max-episodes "${VIZ_EPISODES}"
else
    ${PYTHON} scripts/visualize_gripper_overlay.py \
        --data-root "${DATA_ROOT}" \
        --keypoints-dir "${OUTPUT}" \
        --retarget-dir "${OUTPUT}" \
        --output-dir "${OUTPUT}" \
        --max-episodes "${VIZ_EPISODES}"
fi
echo "  Videos saved to: ${OUTPUT}/<episode>/vis_retargeted_*_overlay.mp4"

# ============================================================
# Step 5: Pretrain
# ============================================================
echo ""
echo "━━━ Step 5: Launch pretraining (${NUM_GPUS} GPUs, ${TRAIN_EPISODES} episodes) ━━━"
bash scripts/launch_pretrain.sh \
    --mode "${MODE}" \
    --base-model "${BASE_MODEL}" \
    --dataset "${MERGED}" \
    --vlm-model-name "${VLM_MODEL_NAME}" \
    --num-gpus "${NUM_GPUS}" \
    --batch-size "${BATCH_SIZE}" \
    --deepspeed-stage "${DEEPSPEED_STAGE}" \
    $([ "${GRADIENT_CHECKPOINTING}" = "true" ] && echo "--gradient-checkpointing") \
    --max-steps "${MAX_STEPS}" \
    --lr "${LR}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║               Pipeline complete! 🎉                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
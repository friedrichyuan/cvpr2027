#!/bin/bash
# Launch GR00T N1.7 pre-training on AoE egocentric human hand data.
#
# This corresponds to EgoScale Stage I: large-scale human pretraining with
# ALL model parameters unfrozen (VLM backbone + visual encoder + projector + DiT).
#
# Two modes:
#   sharpa:    62D state (EEF 9D + 22D Sharpa Wave joints per hand)
#   gripper:   20D state (EEF 9D + gripper per hand)
#
# Usage:
#   bash scripts/launch_pretrain.sh --mode sharpa \
#       --base-model /path/to/GR00T-N1.7-3B \
#       --dataset /path/to/merged_lerobot_dataset \
#       --vlm-model-name /path/to/Cosmos-Reason2-2B
#
set -euo pipefail

PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# === Defaults ===
MODE="sharpa"
BASE_MODEL=""
DATASET_DIR=""
OUTPUT_DIR=""
VLM_MODEL_NAME="nvidia/Cosmos-Reason2-2B"
MAX_STEPS=1000
GLOBAL_BATCH_SIZE=32
LR=1e-4
NUM_GPUS=1
DEEPSPEED_STAGE=3
GRADIENT_CHECKPOINTING=true
EXTRA_ARGS=""

# === Parse args ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --dataset) DATASET_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --vlm-model-name) VLM_MODEL_NAME="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --deepspeed-stage) DEEPSPEED_STAGE="$2"; shift 2 ;;
        --gradient-checkpointing) GRADIENT_CHECKPOINTING=true; shift ;;
        --no-gradient-checkpointing) GRADIENT_CHECKPOINTING=false; shift ;;
        --save-steps) EXTRA_ARGS="$EXTRA_ARGS --save-steps $2"; shift 2 ;;
        --skip-weight-loading) EXTRA_ARGS="$EXTRA_ARGS --skip-weight-loading"; shift ;;
        --) shift; EXTRA_ARGS="$EXTRA_ARGS $*"; break ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/output/${MODE}_pretrain_$(date +%Y%m%d_%H%M%S)"
fi

if [ -z "${BASE_MODEL}" ]; then
    echo "ERROR: --base-model is required"; exit 1
fi
if [ -z "${DATASET_DIR}" ]; then
    echo "ERROR: --dataset is required"; exit 1
fi

# === Mode-specific config (reads from canonical source) ===
EMBODIMENT_TAG=$(${PYTHON} -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}/configs')
from human_ego_config import MODE_CONFIGS
assert '${MODE}' in MODE_CONFIGS, 'mode must be sharpa or gripper'
print(MODE_CONFIGS['${MODE}']['embodiment_tag'])
")
STATE_DIM=$(${PYTHON} -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}/configs')
from human_ego_config import MODE_CONFIGS
print(MODE_CONFIGS['${MODE}']['state_dim'])
")

echo "=== GR00T N1.7 Pre-training (all params): ${MODE^^} ==="
echo "  Mode: ${MODE} (state=${STATE_DIM}D)"
echo "  Embodiment tag: ${EMBODIMENT_TAG}"
echo "  Base model: ${BASE_MODEL}"
echo "  VLM backbone: ${VLM_MODEL_NAME}"
echo "  Dataset: ${DATASET_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Max steps: ${MAX_STEPS}, batch: ${GLOBAL_BATCH_SIZE}, lr: ${LR}"
echo "  DeepSpeed: ZeRO-${DEEPSPEED_STAGE}, gradient_checkpointing: ${GRADIENT_CHECKPOINTING}"
echo ""

# === Verify ===
if [ ! -d "${DATASET_DIR}/meta" ]; then
    echo "ERROR: Dataset meta/ not found: ${DATASET_DIR}"
    echo "  Run: python merge_lerobot_datasets.py --input-dir <output> --mode ${MODE} --output-dir <dataset>"
    exit 1
fi
if [ ! -d "${BASE_MODEL}" ]; then
    echo "ERROR: Base model not found: ${BASE_MODEL}"; exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# === Launch ===
cd "${PROJECT_ROOT}"

if [ "${NUM_GPUS}" -gt 1 ]; then
    # Multi-GPU: use torchrun with DeepSpeed, each rank = one process
    GLOBAL_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE - (GLOBAL_BATCH_SIZE % NUM_GPUS) ))
    [ "${GLOBAL_BATCH_SIZE}" -eq 0 ] && GLOBAL_BATCH_SIZE=${NUM_GPUS}
    echo "  Launching torchrun with ${NUM_GPUS} GPUs, global_batch=${GLOBAL_BATCH_SIZE}"
    ${PYTHON} -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" --master_port=29501 \
        -m gr00t.experiment.launch_finetune \
        --base-model-path "${BASE_MODEL}" \
        --dataset-path "${DATASET_DIR}" \
        --embodiment-tag "${EMBODIMENT_TAG}" \
        --output-dir "${OUTPUT_DIR}" \
        --learning-rate "${LR}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --num-gpus "${NUM_GPUS}" \
        --max-steps "${MAX_STEPS}" \
        --state-dropout-prob 0.8 \
        --modality-config-path "${PROJECT_ROOT}/configs/register_aoe_modality.py" \
        --vlm-model-name "${VLM_MODEL_NAME}" \
        --deepspeed-stage "${DEEPSPEED_STAGE}" \
        $([ "${GRADIENT_CHECKPOINTING}" = "true" ] && echo "--gradient-checkpointing") \
        --tune-llm \
        --tune-visual \
        --tune-projector \
        --tune-diffusion-model \
        ${EXTRA_ARGS} \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
else
    ${PYTHON} -m gr00t.experiment.launch_finetune \
        --base-model-path "${BASE_MODEL}" \
        --dataset-path "${DATASET_DIR}" \
        --embodiment-tag "${EMBODIMENT_TAG}" \
        --output-dir "${OUTPUT_DIR}" \
        --learning-rate "${LR}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --num-gpus "${NUM_GPUS}" \
        --max-steps "${MAX_STEPS}" \
        --state-dropout-prob 0.8 \
        --modality-config-path "${PROJECT_ROOT}/configs/register_aoe_modality.py" \
        --vlm-model-name "${VLM_MODEL_NAME}" \
        --deepspeed-stage "${DEEPSPEED_STAGE}" \
        $([ "${GRADIENT_CHECKPOINTING}" = "true" ] && echo "--gradient-checkpointing") \
        --tune-llm \
        --tune-visual \
        --tune-projector \
        --tune-diffusion-model \
        ${EXTRA_ARGS} \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

echo ""
echo "=== Training complete (${MODE}) ==="
echo "  Model: ${OUTPUT_DIR}"

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AoE Pretrain Data Processing Pipeline
#
# Converts raw AoE data into training-ready format for H-RDT.
# This script runs from inside the recipe folder; the cloned H-RDT repository can
# live anywhere — point HRDT_PROJECT_ROOT at it. After this finishes, run training
# from the H-RDT repo with `bash pretrain_aoe.sh`.
#
# Steps:
#   1. Precompute 48D actions (segment by annotation, modular MANO repr) — uses HaWoR
#   2. Calculate normalization statistics
#   3. Encode language embeddings with T5-XXL — uses the H-RDT repo's T5 encoder
#

set -e

# ========== Configuration ==========
# Recipe folder (this script's directory). HaWoR is expected at $RECIPE_DIR/HaWoR.
export RECIPE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Path to the cloned + patched H-RDT repository (can be anywhere).
# Required for Step 3 (imports the T5 encoder and reads the AoE config from there).
export HRDT_PROJECT_ROOT="${HRDT_PROJECT_ROOT:?Set HRDT_PROJECT_ROOT to your cloned (patched) H-RDT repo path}"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

# Raw data location
export AOE_RAW_DATA="${AOE_RAW_DATA:?Set AOE_RAW_DATA to your raw AoE 数据集 directory}"

# Output directory for processed data
export AOE_PROCESSED_DATA="${AOE_PROCESSED_DATA:?Set AOE_PROCESSED_DATA to your output directory}"

# T5 model path (for language encoding)
export T5_MODEL_PATH="${T5_MODEL_PATH:?Set T5_MODEL_PATH to your t5-v1_1-xxl weights}"

# Processing parameters
export MIN_SEGMENT_FRAMES="${MIN_SEGMENT_FRAMES:-16}"
export NUM_GPUS="${NUM_GPUS:-1}"
export PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-1}"

echo "============================================"
echo "AoE Pretrain Data Processing Pipeline"
echo "============================================"
echo "Recipe dir:     $RECIPE_DIR"
echo "H-RDT repo:     $HRDT_PROJECT_ROOT"
echo "Raw data:       $AOE_RAW_DATA"
echo "Processed data: $AOE_PROCESSED_DATA"
echo "T5 model:       $T5_MODEL_PATH"
echo "Min seg frames: $MIN_SEGMENT_FRAMES"
echo "============================================"

# ========== Step 1: Precompute Actions ==========
echo ""
echo "[Step 1/3] Precomputing 48D actions (segmented by annotation)..."
echo "  Action format: translation(3) + rotation_6d(6) + fingertips(15) per hand"
echo ""

cd "${RECIPE_DIR}/HaWoR"
python "../precompute_actions.py" \
    --data_root "$AOE_RAW_DATA" \
    --output_root "$AOE_PROCESSED_DATA" \
    --min_segment_frames "$MIN_SEGMENT_FRAMES"

# ========== Step 2: Calculate Statistics ==========
echo ""
echo "[Step 2/3] Calculating normalization statistics..."
echo ""

cd "${RECIPE_DIR}"
python "calc_stat.py" \
    --data_root "$AOE_PROCESSED_DATA"

# ========== Step 3: Encode Language Embeddings ==========
echo ""
echo "[Step 3/3] Encoding language embeddings with T5-XXL..."
echo ""

cd "${RECIPE_DIR}"
python "encode_lang_batch.py" \
    --data_root "$AOE_PROCESSED_DATA" \
    --t5_model_path "$T5_MODEL_PATH" \
    --config_path "${HRDT_PROJECT_ROOT}/configs/hrdt_aoe_pretrain.yaml" \
    --num_gpus "$NUM_GPUS" \
    --processes_per_gpu "$PROCESSES_PER_GPU"

# ========== Done ==========
echo ""
echo "============================================"
echo "Pipeline completed!"
echo "============================================"
echo "Output structure:"
echo "  $AOE_PROCESSED_DATA/"
echo "    <episode_name>/"
echo "      0.hdf5  (48D actions per segment)"
echo "      0.mp4   (undistorted video)"
echo "      0.pt    (T5 language embeddings)"
echo "      0.meta.json (frame range)"
echo "    ..."
echo "    aoe_stat.json (min/max statistics)"
echo ""
echo "To start training (from the H-RDT repo):"
echo "  bash pretrain_aoe.sh   # set --data_root to \$AOE_PROCESSED_DATA"
echo "============================================"

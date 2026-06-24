#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download MANO hand model files.
#
# The MANO model is distributed under its own license and requires registration
# at https://mano.is.tue.mpg.de/.  This script does NOT download the files
# automatically; it guides you through the manual steps and then copies or
# symlinks the files to the expected locations for each sub-project.
#
# Usage:
#   bash download_mano.sh /path/to/downloaded/MANO_RIGHT.pkl /path/to/downloaded/MANO_LEFT.pkl
#
# After running this script, the MANO models will be placed under:
#   assets/mano/MANO_RIGHT.pkl
#   assets/mano/MANO_LEFT.pkl
#
# Sub-projects that need MANO models will look for them in their own
# directories or create symlinks to this shared location.  See the
# README of each sub-project for details.

set -euo pipefail

SHARED_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 2 ]; then
    echo "ERROR: Please provide paths to MANO_RIGHT.pkl and MANO_LEFT.pkl."
    echo ""
    echo "  1. Register at https://mano.is.tue.mpg.de/"
    echo "  2. Download MANO_RIGHT.pkl and MANO_LEFT.pkl"
    echo "  3. Run: bash download_mano.sh /path/to/MANO_RIGHT.pkl /path/to/MANO_LEFT.pkl"
    echo ""
    exit 1
fi

SRC_RIGHT="$1"
SRC_LEFT="$2"

if [ ! -f "$SRC_RIGHT" ]; then
    echo "ERROR: File not found: $SRC_RIGHT"
    exit 1
fi
if [ ! -f "$SRC_LEFT" ]; then
    echo "ERROR: File not found: $SRC_LEFT"
    exit 1
fi

# Copy to shared location
cp "$SRC_RIGHT" "$SHARED_DIR/MANO_RIGHT.pkl"
cp "$SRC_LEFT"  "$SHARED_DIR/MANO_LEFT.pkl"

echo "✓ MANO models copied to: $SHARED_DIR/"

# ── GR00T N1.7 recipe ──────────────────────────────────────────────────────────
GR00T_DIR="$SHARED_DIR/../../aoe-training-ready/gr00t_n1d7/scripts/mano_models"
if [ -d "$(dirname "$GR00T_DIR")" ]; then
    mkdir -p "$GR00T_DIR"
    ln -sf "$SHARED_DIR/MANO_RIGHT.pkl" "$GR00T_DIR/MANO_RIGHT.pkl"
    ln -sf "$SHARED_DIR/MANO_LEFT.pkl"  "$GR00T_DIR/MANO_LEFT.pkl"
    echo "✓ Symlinks created: $GR00T_DIR/"
fi

# ── Phantom retarget-replay ────────────────────────────────────────────────────
PHANTOM_DIR="$SHARED_DIR/../../aoe-retarget-replay/phantom/assets/mano_models"
if [ -d "$(dirname "$PHANTOM_DIR")" ]; then
    mkdir -p "$PHANTOM_DIR"
    ln -sf "$SHARED_DIR/MANO_RIGHT.pkl" "$PHANTOM_DIR/MANO_RIGHT.pkl"
    ln -sf "$SHARED_DIR/MANO_LEFT.pkl"  "$PHANTOM_DIR/MANO_LEFT.pkl"
    echo "✓ Symlinks created: $PHANTOM_DIR/"
fi

# ── Visualization (converts .pkl → .npz) ─────────────────────────────────────
VIS_DIR="$SHARED_DIR/../../aoe-visualization/assets/mano"
VIS_SCRIPT="$SHARED_DIR/../../aoe-visualization/scripts/convert_mano_pkl_to_npz.py"
if [ -d "$VIS_DIR" ] && [ -f "$VIS_SCRIPT" ]; then
    python3 "$VIS_SCRIPT" "$SHARED_DIR/MANO_RIGHT.pkl" "$VIS_DIR/MANO_RIGHT.npz"
    python3 "$VIS_SCRIPT" "$SHARED_DIR/MANO_LEFT.pkl"  "$VIS_DIR/MANO_LEFT.npz"
    echo "✓ MANO .npz files generated for visualization: $VIS_DIR/"
fi

echo ""
echo "Done. You may now delete the original downloaded files if you wish."
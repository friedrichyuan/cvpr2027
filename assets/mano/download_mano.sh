#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup MANO hand model files for Open-AoE sub-projects.
#
# The MANO model is distributed under its own license and requires registration
# at https://mano.is.tue.mpg.de/.  This script copies the files you downloaded
# into the shared location at assets/mano/ — all sub-projects that need MANO
# will automatically discover them from there.
#
# Usage:
#   bash assets/mano/download_mano.sh /path/to/MANO_RIGHT.pkl /path/to/MANO_LEFT.pkl

set -euo pipefail

SHARED_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 2 ]; then
    echo "ERROR: Please provide paths to MANO_RIGHT.pkl and MANO_LEFT.pkl."
    echo ""
    echo "  1. Register at https://mano.is.tue.mpg.de/"
    echo "  2. Download MANO_RIGHT.pkl and MANO_LEFT.pkl"
    echo "  3. Run: bash assets/mano/download_mano.sh /path/to/MANO_RIGHT.pkl /path/to/MANO_LEFT.pkl"
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

cp "$SRC_RIGHT" "$SHARED_DIR/MANO_RIGHT.pkl"
cp "$SRC_LEFT"  "$SHARED_DIR/MANO_LEFT.pkl"

echo "✓ MANO models copied to: $SHARED_DIR/"
echo ""
echo "All Open-AoE sub-projects will now discover MANO models from this location."
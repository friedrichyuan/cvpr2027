#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Downsample v2 npz images to a lower resolution (keeps the action vector unchanged).
Range-based for parallel workers: handles files[a:b]. Same filenames in DST (preserves order)."""
import sys, glob, os
import numpy as np
from PIL import Image

SRC, DST, RES, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
files = sorted(glob.glob(os.path.join(os.path.expanduser(SRC), "*.npz")))[a:b]
out_dir = os.path.expanduser(DST)
os.makedirs(out_dir, exist_ok=True)
for f in files:
    try:
        d = np.load(f)
        img = d["image"]
        act = d["action"]
    except Exception:
        continue
    if img.shape[1] == RES:                         # already target res
        out = img.astype(np.uint8)
    else:
        out = np.stack([np.asarray(Image.fromarray(img[t]).resize((RES, RES))) for t in range(img.shape[0])]).astype(np.uint8)
    np.savez_compressed(os.path.join(out_dir, os.path.basename(f)), image=out, action=act)

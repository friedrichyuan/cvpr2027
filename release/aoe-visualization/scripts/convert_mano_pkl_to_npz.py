# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python3
"""One-time converter: standard MANO model ``.pkl`` -> dependency-free NumPy ``.npz``.

The standard MANO hand model files (``MANO_RIGHT.pkl`` / ``MANO_LEFT.pkl``) store
some tensors as legacy lazy-array objects and the joint regressor as a sparse
matrix, which require extra libraries to load. This script converts the model
into dense plain-NumPy arrays so the runtime visualizer only needs ``numpy``.

Run it **once** in an environment that can load the original model files::

    python convert_mano_pkl_to_npz.py MANO_RIGHT.pkl ../assets/mano/MANO_RIGHT.npz
    python convert_mano_pkl_to_npz.py MANO_LEFT.pkl  ../assets/mano/MANO_LEFT.npz

The produced ``.npz`` is what the release ships under ``assets/mano/``.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np


def to_numpy(x) -> np.ndarray:
    """Convert legacy lazy-array / sparse / array objects to a dense NumPy array."""
    if hasattr(x, "r"):           # legacy lazy-array object exposes ``.r``
        return np.array(x.r)
    if hasattr(x, "toarray"):     # sparse matrix
        return np.asarray(x.toarray())
    return np.array(x)


def convert(pkl_path: str, out_path: str) -> None:
    with open(pkl_path, "rb") as f:
        m = pickle.load(f, encoding="latin1")
    out = {
        "f": to_numpy(m["f"]).astype(np.int64),
        "v_template": to_numpy(m["v_template"]).astype(np.float64),
        "shapedirs": to_numpy(m["shapedirs"]).astype(np.float64),
        "posedirs": to_numpy(m["posedirs"]).astype(np.float64),
        "J_regressor": to_numpy(m["J_regressor"]).astype(np.float64),
        "weights": to_numpy(m["weights"]).astype(np.float64),
        "kintree_table": to_numpy(m["kintree_table"]).astype(np.int64),
    }
    if "hands_mean" in m:
        out["hands_mean"] = to_numpy(m["hands_mean"]).astype(np.float64)
    np.savez(out_path, **out)
    print(f"saved {out_path}")
    for k, v in out.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])

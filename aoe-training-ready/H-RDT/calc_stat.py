#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Calculate min/max statistics for AoE precomputed action data.

Reads all HDF5 files produced by precompute_actions.py and computes
per-dimension min/max for normalization during training.

Output: aoe_stat.json with format:
  {"aoe": {"min": [...], "max": [...]}}
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def collect_action_stats(data_root, output_path, large_values_log=None):
    """
    Collect min/max statistics across all precomputed action HDF5 files.
    """
    import h5py

    global_min = None
    global_max = None
    file_count = 0
    error_files = []
    large_values_files = []
    threshold = 5.0

    root_path = Path(data_root)
    hdf5_files = []

    for episode_dir in root_path.iterdir():
        if episode_dir.is_dir():
            hdf5_files.extend(episode_dir.glob('*.hdf5'))

    print(f"Found {len(hdf5_files)} HDF5 files")

    for file_path in tqdm(hdf5_files, desc="Computing statistics"):
        try:
            with h5py.File(file_path, 'r') as f:
                if 'actions' not in f:
                    error_files.append((str(file_path), "Missing 'actions' dataset"))
                    continue

                action_data = f['actions'][:]

                if np.any(np.abs(action_data) > threshold):
                    large_values_files.append(str(file_path))

                if global_min is None:
                    global_min = np.min(action_data, axis=0)
                    global_max = np.max(action_data, axis=0)
                else:
                    global_min = np.minimum(global_min, np.min(action_data, axis=0))
                    global_max = np.maximum(global_max, np.max(action_data, axis=0))

                file_count += 1
        except Exception as e:
            error_files.append((str(file_path), str(e)))

    if global_min is None:
        print("ERROR: No valid files found!")
        return

    stat_dict = {
        "aoe": {
            "min": global_min.tolist(),
            "max": global_max.tolist(),
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stat_dict, f, indent=4)

    print(f"\nStatistics computed over {file_count} files")
    print(f"Action dim: {len(global_min)}")
    print(f"Range: [{global_min.min():.4f}, {global_max.max():.4f}]")
    print(f"Files with |action| > {threshold}: {len(large_values_files)}")
    print(f"Saved to: {output_path}")

    if large_values_log:
        with open(large_values_log, 'w') as f:
            f.write(f"Files with |action| > {threshold}:\n\n")
            for fp in large_values_files:
                f.write(f"{fp}\n")

    if error_files:
        print(f"\nErrors ({len(error_files)}):")
        for path, err in error_files[:10]:
            print(f"  {path}: {err}")


def main():
    parser = argparse.ArgumentParser(description='Calculate AoE action statistics')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root of processed data (contains episode subdirs)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output JSON path (default: <data_root>/aoe_stat.json)')
    parser.add_argument('--large_values_log', type=str, default=None,
                        help='Log file for large-value files')
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(args.data_root, 'aoe_stat.json')
    if args.large_values_log is None:
        args.large_values_log = os.path.join(args.data_root, 'aoe_large_values.txt')

    collect_action_stats(args.data_root, args.output_path, args.large_values_log)


if __name__ == "__main__":
    main()

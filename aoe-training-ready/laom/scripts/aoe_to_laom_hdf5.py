#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AoE -> laom (dunnolab/laom) HDF5 adapter.

Converts AoE episode npz (key `image` (T,64,64,3) uint8 + `action` (T,26) f32)
into laom's HDF5 trajectory format expected by DCSLAOMInMemoryDataset /
DCSLAOMTrueActionsDataset (src/utils.py):
  - one group per trajectory: {obs (T,H,W,3) uint8, actions (T,A) f32, states (T,1) f32}
  - root attr `img_hw` (scalar int)

DCSLAOMInMemoryDataset assumes a UNIFORM trajectory length, so we chunk each AoE
episode into fixed-length clips. Emits a full (unlabeled) file and a labeled subset.

Action convention: we ground the latent on the TASK action = hand (first 20 dims);
camera ego-motion (dims 20:26) is the in-video DISTRACTOR (the LAOM setup), so it is
NOT used as a label. act-dim default 20.
"""
import argparse, glob, os, random
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/PATH_TO/aoe_npz/aoe26")
    ap.add_argument("--out-dir", default="/PATH_TO/laom_data")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--act-dim", type=int, default=20,
                    help="hand=task action (20); camera 20:26 is the in-video distractor, not a label")
    ap.add_argument("--max-clips", type=int, default=3000)
    ap.add_argument("--labeled-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, "*.npz")))
    assert files, f"no npz under {args.src}"

    clips = []  # (obs (T,H,W,3) uint8, act (T,A) f32)
    for f in files:
        d = np.load(f)
        img = d["image"]                                   # (T,64,64,3) uint8
        act = d["action"][:, :args.act_dim].astype(np.float32)
        T = img.shape[0]
        for i in range(T // args.clip_len):
            s = i * args.clip_len
            clips.append((img[s:s + args.clip_len], act[s:s + args.clip_len]))
            if len(clips) >= args.max_clips:
                break
        if len(clips) >= args.max_clips:
            break
    assert clips, "no clips produced (clip-len too large?)"

    random.shuffle(clips)
    H = int(clips[0][0].shape[1])
    A = int(clips[0][1].shape[1])
    n_lab = max(1, int(round(len(clips) * args.labeled_frac)))

    def write(path, subset):
        with h5py.File(path, "w") as df:
            df.attrs["img_hw"] = H
            for i, (obs, act) in enumerate(subset):
                g = df.create_group(f"traj_{i:06d}")
                g.create_dataset("obs", data=obs.astype(np.uint8), compression="gzip")
                g.create_dataset("actions", data=act.astype(np.float32))
                g.create_dataset("states", data=np.zeros((obs.shape[0], 1), np.float32))
        print(f"wrote {path}: {len(subset)} clips x {args.clip_len}  (img_hw={H}, act_dim={A})")

    write(os.path.join(args.out_dir, "aoe_laom_full.hdf5"), clips)
    write(os.path.join(args.out_dir, "aoe_laom_labeled.hdf5"), clips[:n_lab])
    print(f"done: {len(clips)} clips total, {n_lab} labeled ({args.labeled_frac:.0%})")


if __name__ == "__main__":
    main()

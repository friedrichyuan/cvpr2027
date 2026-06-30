#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenieRedux LAM trainer on AoE (uses the official models.lam.LatentActionModel, no fork).

Trains GenieRedux's Latent Action Model core: video clips -> ST-ViViT encoder ->
VQ latent actions -> decode -> reconstruct future frames. Objective is the model's own
`loss = vq_loss_w*vq_loss + recon_loss_w*recon_loss` (forward() returns it directly).

GenieRedux's native data path is a complex retro "session library" (MultiEnvironmentDataset);
we bypass it and feed AoE frames directly (the model only needs (B,C,F,H,W) clips in [0,1]).
Run from the GenieRedux repo root, in its conda env. Params follow configs/lam/lam.yaml.
"""
import argparse, glob, os, csv
import numpy as np
import torch

from models.lam import LatentActionModel


def load_clips(npz_dir, clip_len, max_clips):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(npz_dir), "*.npz")))
    assert files, f"no npz in {npz_dir}"
    clips = []
    for f in files:
        img = np.load(f)["image"]                 # (T,64,64,3) uint8
        for i in range(img.shape[0] // clip_len):
            clips.append(img[i * clip_len:(i + 1) * clip_len])
            if len(clips) >= max_clips:
                break
        if len(clips) >= max_clips:
            break
    x = np.stack(clips).astype(np.float32) / 255.0           # (N,T,H,W,3)
    return torch.from_numpy(x).permute(0, 4, 1, 2, 3).contiguous()  # (N,C,T,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/PATH_TO/aoe_npz/aoe26")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--max-clips", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="aoe_genieredux_lam_loss.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_clips(args.npz_dir, args.clip_len, args.max_clips)
    print(f"[data] {x.shape[0]} clips, shape {tuple(x.shape)} (B,C,T,H,W)")

    # configs/lam/lam.yaml
    model = LatentActionModel(
        dim=512, codebook_size=7, image_size=64, patch_size=4, temporal_patch_size=1,
        num_blocks=8, codebook_dim=32, dim_head=64, heads=8, channels=3, ff_mult=4.0,
        recon_loss_w=1.0, vq_loss_w=1.0, wandb_mode="disabled",
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] GenieRedux LatentActionModel params={n_params/1e6:.2f}M, codebook_size=7")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    N, bs, rows = x.shape[0], args.batch, []
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for i in range(0, N, bs):
            vb = x[perm[i:i + bs]].to(dev)
            loss = model(vb)
            if isinstance(loss, (tuple, list)):
                loss = loss[0]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            rows.append({"step": len(rows), "loss": loss.item()})
        print(f"[ep {ep+1}/{args.epochs}] loss={tot/nb:.4f}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss"])
        for r in rows:
            w.writerow([r["step"], r["loss"]])
    print(f"[done] {len(rows)} steps; loss {rows[0]['loss']:.4f} -> {rows[-1]['loss']:.4f}; wrote {args.out}")


if __name__ == "__main__":
    main()

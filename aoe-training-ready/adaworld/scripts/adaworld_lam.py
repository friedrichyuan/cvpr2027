#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AdaWorld LAM trainer on AoE (uses the official lam.modules.lam.LatentActionModel, no fork).

Trains AdaWorld's latent-action autoencoder (VAE-style: spatiotemporal encoder -> z_mu/z_var ->
decode -> reconstruct next frame). Objective replicates their LAM LightningModule.shared_step:
  loss = mse(videos[:,1:] - recon) + beta * KL(z_mu, z_var)
Run from AdaWorld/lam/ (so `from lam.modules.lam import LatentActionModel` resolves), in AdaWorld's venv.
Params follow lam/config/lam.yaml (model_dim=1024, latent_dim=32, patch_size=16, 16+16 blocks).
"""
import os, glob, csv, argparse
import numpy as np
import torch

from lam.modules.lam import LatentActionModel


def load_clips(npz_dir, clip_len, max_clips):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(npz_dir), "*.npz")))
    assert files, f"no npz in {npz_dir}"
    clips = []
    for f in files:
        img = np.load(f)["image"]                  # (T,64,64,3) uint8
        for i in range(img.shape[0] // clip_len):
            clips.append(img[i * clip_len:(i + 1) * clip_len])
            if len(clips) >= max_clips:
                break
        if len(clips) >= max_clips:
            break
    x = np.stack(clips).astype(np.float32) / 255.0           # (N,T,H,W,C) channels-LAST
    return torch.from_numpy(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/PATH_TO/aoe_npz/aoe26")
    ap.add_argument("--clip-len", type=int, default=2)       # lam.yaml num_frames=2 (frame pairs)
    ap.add_argument("--max-clips", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--model-dim", type=int, default=1024)   # lam.yaml; drop to 512 (their default) if OOM
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--beta", type=float, default=0.0002)
    ap.add_argument("--out", default="aoe_adaworld_lam_loss.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_clips(args.npz_dir, args.clip_len, args.max_clips)
    print(f"[data] {x.shape[0]} clips, shape {tuple(x.shape)} (B,T,H,W,C)")

    model = LatentActionModel(
        in_dim=3, model_dim=args.model_dim, latent_dim=32, patch_size=16,
        enc_blocks=args.blocks, dec_blocks=args.blocks, num_heads=16, dropout=0.0,
    ).to(dev)
    print(f"[model] AdaWorld LatentActionModel dim={args.model_dim} blocks={args.blocks} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    N, bs, rows = x.shape[0], args.batch, []
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        agg = {"loss": 0.0, "mse": 0.0, "kl": 0.0}; nb = 0
        for i in range(0, N, bs):
            vb = x[perm[i:i + bs]].to(dev)
            out = model({"videos": vb})
            gt = vb[:, 1:]
            mse = ((gt - out["recon"]) ** 2).mean()
            kl = -0.5 * torch.sum(1 + out["z_var"] - out["z_mu"] ** 2 - out["z_var"].exp(), dim=1).mean()
            loss = mse + args.beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            agg["loss"] += loss.item(); agg["mse"] += mse.item(); agg["kl"] += kl.item(); nb += 1
            rows.append({"step": len(rows), "loss": loss.item(), "mse": mse.item(), "kl": kl.item()})
        print(f"[ep {ep+1}/{args.epochs}] loss={agg['loss']/nb:.5f} mse={agg['mse']/nb:.5f} kl={agg['kl']/nb:.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss", "mse", "kl"])
        for r in rows:
            w.writerow([r["step"], r["loss"], r["mse"], r["kl"]])
    print(f"[done] {len(rows)} steps; loss {rows[0]['loss']:.5f} -> {rows[-1]['loss']:.5f}; "
          f"mse {rows[0]['mse']:.5f} -> {rows[-1]['mse']:.5f}; wrote {args.out}")


if __name__ == "__main__":
    main()

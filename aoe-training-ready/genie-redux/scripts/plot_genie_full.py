#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot tokenizer + genie (dynamics+LAM) training loss curves for the full GenieRedux AoE run."""
import csv, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    xs, ys = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["step"])); ys.append(float(row["loss"]))
    return xs, ys


def ema(ys, a=0.05):
    out, m = [], ys[0]
    for y in ys:
        m = a * y + (1 - a) * m; out.append(m)
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "aoe_genie_full"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, stage, title in [
        (axes[0], "tokenizer", "Stage 1 · Tokenizer (video VQ-VAE)  vq+recon loss"),
        (axes[1], "genie", "Stage 2 · GenieRedux (LAM+dynamics)  token CE loss"),
    ]:
        p = os.path.join(out_dir, f"{stage}_loss.csv")
        if not os.path.exists(p):
            ax.set_title(title + "  [missing]"); continue
        xs, ys = read(p)
        ax.plot(xs, ys, lw=0.6, alpha=0.35, color="tab:blue")
        ax.plot(xs, ema(ys), lw=1.8, color="tab:blue")
        ax.set_title(f"{title}\n{ys[0]:.3f} → {ys[-1]:.4f}  ({len(xs)} steps)", fontsize=9)
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
    fig.suptitle("Full GenieRedux on AoE (egocentric) — multi-L40", fontsize=11)
    fig.tight_layout()
    png = os.path.join(out_dir, "aoe_genie_full_loss.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()

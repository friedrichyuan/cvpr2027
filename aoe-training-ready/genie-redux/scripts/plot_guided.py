#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot guided-26D GenieRedux dynamics token-CE loss."""
import csv, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "aoe_guided26d"
xs, ys = [], []
with open(os.path.join(out, "guided_loss.csv")) as f:
    for r in csv.DictReader(f):
        xs.append(int(r["step"])); ys.append(float(r["loss"]))


def ema(ys, a=0.05):
    o, m = [], ys[0]
    for y in ys:
        m = a * y + (1 - a) * m; o.append(m)
    return o


fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(xs, ys, lw=0.6, alpha=0.3, color="tab:green")
ax.plot(xs, ema(ys), lw=1.8, color="tab:green")
ax.set_title(f"Guided GenieRedux on AoE — conditioned on TRUE 26D action (hand20 + camera6)\n"
             f"token CE {ys[0]:.3f} → {ys[-1]:.4f}  ({len(xs)} steps, 3×L40)", fontsize=9)
ax.set_xlabel("step"); ax.set_ylabel("token CE loss"); ax.grid(alpha=0.3)
fig.tight_layout()
png = os.path.join(out, "aoe_guided26d_loss.png")
fig.savefig(png, dpi=130)
print("wrote", png)

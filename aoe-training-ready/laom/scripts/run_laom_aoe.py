#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin runner: run ONLY laom's LAOM pretraining (train_laom) on AoE HDF5.

We call dunnolab/laom's own `train_laom()` (no fork) but skip the downstream
BC / action-decoder phases (which need the dm_control env). wandb is disabled and
`wandb.log` is monkeypatched to capture the loss curve to CSV.

Usage (from the laom repo root, in its venv):
    python run_laom_aoe.py --data data/aoe_laom_full.hdf5 \
        --labeled data/aoe_laom_labeled.hdf5 --epochs 30 --out aoe_laom_loss.csv
"""
import argparse, csv, os

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import wandb  # noqa: E402

_rows = []
def _capture_log(d, *a, **k):
    if isinstance(d, dict):
        _rows.append({k2: (float(v) if hasattr(v, "__float__") else v) for k2, v in d.items()})
wandb.log = _capture_log
wandb.init = lambda *a, **k: None  # train_laom doesn't need a real run

from train_laom_labels import train_laom, LAOMConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--labeled-loss-coef", type=float, default=0.05)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--out", default="aoe_laom_loss.csv")
    args = ap.parse_args()

    cfg = LAOMConfig(
        data_path=args.data,
        labeled_data_path=args.labeled,
        eval_data_path=None,            # <- skip env / eval
        num_epochs=args.epochs,
        frame_stack=args.frame_stack,
        future_obs_offset=1,
        labeled_loss_coef=args.labeled_loss_coef,
    )
    print(f"[run] train_laom: data={args.data} labeled={args.labeled} "
          f"epochs={args.epochs} coef={args.labeled_loss_coef}")
    train_laom(cfg)

    keys = ["lapo/total_loss", "lapo/mse_loss", "lapo/true_action_mse_loss",
            "lapo/state_probe_mse_loss", "lapo/action_probe_mse_loss",
            "lapo/state_action_probe_mse_loss"]
    keys = [k for k in keys if any(k in r for r in _rows)] or sorted({k for r in _rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step"] + keys)
        for i, r in enumerate(_rows):
            w.writerow([i] + [r.get(k, "") for k in keys])
    if _rows:
        first, last = _rows[0], _rows[-1]
        print(f"[done] {len(_rows)} steps; "
              f"total_loss {first.get('lapo/total_loss')} -> {last.get('lapo/total_loss')}; "
              f"mse {first.get('lapo/mse_loss')} -> {last.get('lapo/mse_loss')}; "
              f"true_act_mse {first.get('lapo/true_action_mse_loss')} -> {last.get('lapo/true_action_mse_loss')}")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

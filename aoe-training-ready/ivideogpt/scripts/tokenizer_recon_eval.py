#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0a diagnostic: measure iVideoGPT (compressive VQGAN) tokenizer
reconstruction quality on AoE frames.

A world model's prediction ceiling is its tokenizer's reconstruction quality.
The pretrained tokenizer was trained on OXE robot data; AoE is egocentric hands,
so we quantify the domain gap (and later compare against a finetuned tokenizer).

recon = detokenize(tokenize(x)); report PSNR / SSIM / LPIPS over many segments.

Run inside the upstream iVideoGPT checkout (PYTHONPATH=., or set IVIDEOGPT_UPSTREAM_ROOT). Needs the ivideogpt package + inference/utils.
  python tokenizer_recon_eval.py \
    --tokenizer pretrained_models/ivideogpt-oxe-64-act-free \
    --npz_glob '/PATH_TO/aoe_npz/aoe/*.npz' --n 30 --reps 4
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

# make ivideogpt package + inference/utils importable regardless of cwd
REPO = os.environ.get("IVIDEOGPT_UPSTREAM_ROOT") or os.path.expanduser("~/iVideoGPT")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'inference'))

from ivideogpt.vq_model import CompressiveVQModel  # noqa: E402
from utils import NPZParser  # noqa: E402
import piqa  # noqa: E402
import lpips  # noqa: E402

device = 'cuda'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True, help='dir containing subfolder `tokenizer/`, or a tokenizer dir directly')
    ap.add_argument('--subfolder', default='tokenizer', help="set '' if --tokenizer points directly at the tokenizer")
    ap.add_argument('--npz_glob', required=True)
    ap.add_argument('--dataset_name', default='aoe')
    ap.add_argument('--segment_length', type=int, default=16)
    ap.add_argument('--context_length', type=int, default=2)
    ap.add_argument('--resolution', type=int, default=64)
    ap.add_argument('--n', type=int, default=30, help='max npz files')
    ap.add_argument('--reps', type=int, default=4, help='random segments sampled per file')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sub = args.subfolder if args.subfolder else None
    tok = CompressiveVQModel.from_pretrained(
        args.tokenizer, subfolder=sub, low_cpu_mem_usage=False).to(device).eval()
    print(f"tokenizer.context_length = {getattr(tok, 'context_length', 'NA')}")

    parser = NPZParser(args.segment_length, args.resolution)
    psnr = piqa.PSNR(value_range=1.0).to(device)
    ssim = piqa.SSIM(n_channels=3).to(device)
    lp = lpips.LPIPS(net='vgg', verbose=False).to(device)

    files = sorted(glob.glob(os.path.expanduser(args.npz_glob)))[:args.n]
    if not files:
        sys.exit(f"no npz matched {args.npz_glob}")

    P, S, L, nseg = [], [], [], 0
    # split context vs dynamic frames too (compressive tokenizer treats them differently)
    Pc, Pd = [], []
    for f in files:
        for _ in range(args.reps):
            imgs, _ = parser.parse(f, args.dataset_name, load_action=False)  # (T,C,H,W) in [0,1]
            pv = imgs.unsqueeze(0).to(device)
            with torch.no_grad():
                tokens, _ = tok.tokenize(pv, args.context_length)
                recon = tok.detokenize(tokens, args.context_length).clamp(0.0, 1.0)  # (1,T,C,H,W)
            gt, rc = pv[0], recon[0]  # (T,C,H,W)
            with torch.no_grad():
                P.append(psnr(rc, gt).item())
                S.append(ssim(rc, gt).item())
                L.append(lp(rc, gt, normalize=True).mean().item())
                c = args.context_length
                Pc.append(psnr(rc[:c], gt[:c]).item())
                Pd.append(psnr(rc[c:], gt[c:]).item())
            nseg += 1

    def stat(x):
        a = np.array(x)
        return f"{a.mean():.3f} ± {a.std():.3f}"

    print(f"\n=== Tokenizer reconstruction on AoE ({len(files)} files x {args.reps} reps = {nseg} segments,"
          f" {args.segment_length} frames @ {args.resolution}px) ===")
    print(f"PSNR  (dB)  : {stat(P)}   [higher better]")
    print(f"SSIM        : {stat(S)}   [higher better]")
    print(f"LPIPS (vgg) : {stat(L)}   [lower better]")
    print(f"PSNR context-frames : {stat(Pc)}")
    print(f"PSNR dynamic-frames : {stat(Pd)}")


if __name__ == '__main__':
    main()

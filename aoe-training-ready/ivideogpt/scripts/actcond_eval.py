#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 4: action-conditioned controllability eval for iVideoGPT × AoE.

Core scientific question: does conditioning on the TRUE hand/camera action predict
the GT future better than a SHUFFLED or ZERO action? If true >> shuffle, the model
genuinely uses the action signal (AoE's hand labels are useful supervision).

For each eval clip: take `context_length` context frames, roll out `segment_length`
frames under each action mode, detokenize, and score predicted FUTURE frames vs GT
(PSNR / SSIM / LPIPS). Aggregate per mode.

Run inside the upstream iVideoGPT checkout (PYTHONPATH=., or set IVIDEOGPT_UPSTREAM_ROOT).
  python actcond_eval.py \
    --tokenizer <ft_tokenizer_dir> \
    --transformer <trained_model_dir>           # dir containing model.safetensors (+ config.json)
    --npz_glob '/PATH_TO/aoe_npz_ac/aoe26/*.npz' --action_dim 26 \
    --modes true shuffle zero --n 40 --repeat 2
Use --action_dim 0 (or --act_free) for the action-free baseline model.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

REPO = os.environ.get("IVIDEOGPT_UPSTREAM_ROOT") or os.path.expanduser("~/iVideoGPT")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'inference'))

from transformers import AutoModelForCausalLM, AutoConfig  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from ivideogpt.vq_model import CompressiveVQModel  # noqa: E402
from ivideogpt.transformer import HeadModelWithAction  # noqa: E402
from utils import NPZParser  # noqa: E402
import piqa  # noqa: E402
import lpips  # noqa: E402

device = 'cuda'


def load_models(args):
    tok = CompressiveVQModel.from_pretrained(
        args.tokenizer, subfolder=args.tok_subfolder or None, low_cpu_mem_usage=False).to(device).eval()
    assert args.context_length == tok.context_length, \
        f"context_length {args.context_length} != tokenizer {tok.context_length}"
    sub = 'transformer'
    cfg_dir = args.transformer
    if args.act_free:
        model = AutoModelForCausalLM.from_pretrained(
            cfg_dir, subfolder=(sub if os.path.isdir(os.path.join(cfg_dir, sub)) else ''),
            low_cpu_mem_usage=False).to(device).eval()
    else:
        has_sub = os.path.isdir(os.path.join(cfg_dir, sub))
        sd_path = os.path.join(cfg_dir, sub, 'model.safetensors') if has_sub else os.path.join(cfg_dir, 'model.safetensors')
        config = AutoConfig.from_pretrained(cfg_dir, subfolder=(sub if has_sub else ''))
        # accelerate save_state does not persist the resized vocab in config.json
        # (act-cond/--special_token resizes embeddings); read it back from the weights.
        from safetensors import safe_open as _sopen
        with _sopen(sd_path, 'pt') as _f:
            config.vocab_size = _f.get_slice('llm.lm_head.weight').get_shape()[0]
        base = AutoModelForCausalLM.from_config(config)
        prelude = (256 + 1) * args.context_length - 1
        model = HeadModelWithAction(base, action_dim=args.action_dim, prelude_tokens_num=prelude,
                                    tokens_num_per_dyna=16, context=args.context_length,
                                    segment_length=args.segment_length).to(device).eval()
        model.load_state_dict(load_file(sd_path), strict=True)
    return tok, model


def vary_action(actions, mode, rng):
    """actions: (T, D) torch. Return varied (T, D)."""
    if mode == 'true':
        return actions
    if mode == 'zero':
        return torch.zeros_like(actions)
    if mode == 'shuffle':  # break temporal correspondence: permute over time
        perm = torch.randperm(actions.shape[0], generator=rng)
        return actions[perm]
    raise ValueError(mode)


@torch.no_grad()
def run(args):
    tok, model = load_models(args)
    parser = NPZParser(args.segment_length, args.resolution)
    psnr = piqa.PSNR(value_range=1.0).to(device)
    ssim = piqa.SSIM(n_channels=3).to(device)
    lp = lpips.LPIPS(net='vgg', verbose=False).to(device)
    rng = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    files = sorted(glob.glob(os.path.expanduser(args.npz_glob)))[:args.n]
    c = args.context_length
    max_new_tokens = (1 + 16) * (args.segment_length - c) - 1
    res = {m: {'psnr': [], 'ssim': [], 'lpips': [], 'loss': []} for m in args.modes}

    for f in files:
        imgs, acts = parser.parse(f, args.dataset_name, load_action=not args.act_free)  # (T,C,H,W),(T,D)
        pv = imgs.unsqueeze(0).to(device)
        tokens, labels = tok.tokenize(pv, c)
        gt_future = pv[0, c:]  # (T-c,C,H,W)
        for m in args.modes:
            a = None if args.act_free else vary_action(acts.to(device), m, rng).unsqueeze(0)
            # (1) teacher-forced loss: robust, no sampling — does the action lower next-token loss?
            mi = {'input_ids': tokens, 'labels': labels}
            if a is not None:
                mi['action'] = a
            res[m]['loss'].append(model(**mi).loss.item())
            # (2) sampled rollout metrics (optional; noisy)
            if args.gen:
                gk = dict(do_sample=True, temperature=1.0, top_k=100, max_new_tokens=max_new_tokens, pad_token_id=50256)
                gi = tokens[:, :c * (16 * 16 + 1)].repeat(args.repeat, 1)
                gen = model.generate(gi, **gk) if a is None else model.generate(gi, action=a.repeat(args.repeat, 1, 1), **gk)
                pf = tok.detokenize(gen, c).clamp(0.0, 1.0)[:, c:]
                for r in range(pf.shape[0]):
                    res[m]['psnr'].append(psnr(pf[r], gt_future).item())
                    res[m]['ssim'].append(ssim(pf[r], gt_future).item())
                    res[m]['lpips'].append(lp(pf[r], gt_future, normalize=True).mean().item())
            if args.act_free:
                break  # action modes identical for act-free

    tag = 'act-free' if args.act_free else f'act-cond D={args.action_dim}'
    print(f"\n=== Controllability ({len(files)} clips, future={args.segment_length-c} frames @ {args.resolution}px, model={tag}) ===")
    hdr = f"{'mode':<8} {'loss↓':>8}" + ("  " + f"{'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8}" if args.gen else "")
    print(hdr)
    for m in args.modes:
        if not res[m]['loss']:
            continue
        line = f"{m:<8} {np.mean(res[m]['loss']):8.4f}"
        if args.gen:
            line += f"  {np.mean(res[m]['psnr']):8.3f} {np.mean(res[m]['ssim']):8.3f} {np.mean(res[m]['lpips']):8.3f}"
        print(line)
        if args.act_free:
            break
    print("# true << shuffle/zero on loss  =>  model genuinely uses the action signal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--tok_subfolder', default='', help="'' if --tokenizer points directly at tokenizer dir")
    ap.add_argument('--transformer', required=True)
    ap.add_argument('--npz_glob', required=True)
    ap.add_argument('--dataset_name', default='aoe26')
    ap.add_argument('--action_dim', type=int, default=26)
    ap.add_argument('--act_free', action='store_true')
    ap.add_argument('--modes', nargs='+', default=['true', 'shuffle', 'zero'])
    ap.add_argument('--segment_length', type=int, default=16)
    ap.add_argument('--context_length', type=int, default=2)
    ap.add_argument('--resolution', type=int, default=64)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--gen', action='store_true', help='also do sampled rollout metrics (slow); default loss-only')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()

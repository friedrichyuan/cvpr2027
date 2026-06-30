#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full GenieRedux (tokenizer + LAM + dynamics) on AoE — a generative egocentric world model.

Uses the OFFICIAL models (Tokenizer, LatentActionModel, Dynamics/MaskGIT, GenieRedux) built via the
repo's own `construct_model`, so every dim stays exactly consistent with `configs/config/genie_redux.yaml`.
We bypass only the retro-specialized `MultiEnvironmentDataset`, feeding AoE npz clips straight to
`model.forward` / `model.sample` — the same clean precedent already used in `genie_redux_lam.py`.

Stages (run in order; tokenizer must finish before genie before rollout):
  tokenizer : train the video VQ-VAE                 loss = tokenizer(videos)
  genie     : train LAM + dynamics (tokenizer frozen) loss = model(videos)   [loads tokenizer ckpt]
  rollout   : 2 prime frames + LAM latent actions -> generate 14 frames; GIF(GT|pred) + PSNR
              + controllability: true vs. shuffled latent actions -> delta-PSNR

DDP:  torchrun --nproc_per_node=N genie_redux_full.py --stage tokenizer ...
"""
import os, glob, csv, argparse, math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.chdir(os.environ.get("GENIE_UPSTREAM_ROOT", os.path.dirname(os.path.abspath(__file__))))  # cwd = GenieRedux repo root (models/ + relative configs)
from omegaconf import OmegaConf
from models import construct_model


# ----- exact config mirror of configs/config/genie_redux.yaml (what construct_model reads) -----
def build_cfg(model_name, image_size=64, patch_size=4):
    common = dict(image_size=image_size, patch_size=patch_size, temporal_patch_size=1,
                  num_blocks=8, dim_head=64, heads=8, ff_mult=4,
                  vq_loss_weight=1.0, recons_loss_weight=1.0)
    return OmegaConf.create({
        "model": model_name,
        "train": {"wandb_mode": "disabled"},
        "tokenizer": dict(dim=512, codebook_size=1024, **common),
        "lam": dict(dim=512, codebook_size=7, **common),
        "dynamics": dict(dim=512, action_dim=5, image_size=image_size, patch_size=patch_size,
                         temporal_patch_size=1, num_blocks=12, dim_head=64, heads=8, ff_mult=4,
                         max_seq_len=8000, sample_temperature=1.0, sample_num_frames=15,
                         use_action_embeddings=True, use_token=False, is_guided=False),
    })


def load_clips(npz_dir, clip_len, max_clips, skip=0, max_files=None, rank=0, world=1):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(npz_dir), "*.npz")))
    assert files, f"no npz in {npz_dir}"
    files = files[skip:]                       # rollout uses held-out tail
    if max_files is not None:
        files = files[:max_files]              # training caps to first N eps (no holdout leakage)
    if world > 1:                              # DDP: each rank reads an equal contiguous file shard
        per = len(files) // world              #   (avoids every rank loading the FULL set -> 4x RAM/swap death at 128)
        files = files[rank * per:(rank + 1) * per]
    clips = []
    for f in files:
        img = np.load(f)["image"]              # (T,64,64,3) uint8
        for i in range(img.shape[0] // clip_len):
            clips.append(img[i * clip_len:(i + 1) * clip_len])
            if len(clips) >= max_clips:
                break
        if len(clips) >= max_clips:
            break
    x = np.stack(clips).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 4, 1, 2, 3).contiguous()  # (N,C,T,H,W)


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
        return True, rank, world, local, f"cuda:{local}"
    return False, 0, 1, 0, ("cuda" if torch.cuda.is_available() else "cpu")


def train_stage(args, ddp, rank, world, local, dev):
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    model_name = "tokenizer" if args.stage == "tokenizer" else "genie_redux"
    model = construct_model(build_cfg(model_name, args.image_size, args.patch_size)).to(dev)

    if args.stage == "genie":                  # load frozen tokenizer
        tok = torch.load(args.tokenizer_ckpt, map_location="cpu")
        model.tokenizer.load_state_dict(tok["model"])
        if is_main:
            print(f"[genie] loaded tokenizer from {args.tokenizer_ckpt}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    if is_main:
        print(f"[model] {model_name} trainable={n_params:.1f}M")

    if args.stream:                                        # streaming loader (file-amortized, constant RAM -> scales to 100h+)
        from aoe_stream_dataset import StreamingClipIterable, count_clips
        from torch.utils.data import DataLoader
        total = count_clips(args.npz_dir, args.clip_len, max_files=args.train_eps)
        spe = max(1, total // (max(world, 1) * args.batch))            # fixed steps/epoch -> DDP-safe
        ds = StreamingClipIterable(args.npz_dir, args.clip_len, max_files=args.train_eps,
                                   with_action=False, world=world, rank=rank)
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=(args.num_workers > 0))
        _it = iter(dl)
        if is_main:
            print(f"[data] STREAM clips={total} steps/epoch={spe}; eff_batch={args.batch*world}", flush=True)
        def epoch_batches(ep):
            return (next(_it) for _ in range(spe))
    else:
        x = load_clips(args.npz_dir, args.clip_len, args.max_clips, max_files=args.train_eps, rank=rank, world=world)
        if is_main:
            print(f"[data] per-rank {tuple(x.shape)}; world={world}; eff_batch={args.batch*world}", flush=True)
        def epoch_batches(ep):
            N = x.shape[0]; perm = torch.randperm(N)
            return (x[perm[i:i + args.batch]] for i in range(0, N, args.batch))

    if ddp:
        model = DDP(model, device_ids=[local], find_unused_parameters=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    rows, step = [], 0
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for vb in epoch_batches(ep):
            vb = vb.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(vb)
                if isinstance(loss, (tuple, list)):
                    loss = loss[0]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            tot += loss.item(); nb += 1; step += 1
            if is_main:
                rows.append({"step": step, "loss": loss.item()})
                if step % 20 == 0:
                    print(f"[{args.stage} ep{ep+1} step{step}] loss={loss.item():.4f}", flush=True)
                if step % args.save_every == 0:                       # periodic ckpt (resilience)
                    raw_ = model.module if ddp else model
                    torch.save({"model": raw_.state_dict()},
                               os.path.join(args.out_dir, f"{args.stage}.pt"))
        if is_main:
            print(f"[ep {ep+1}/{args.epochs}] mean loss={tot/nb:.4f}", flush=True)

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        raw = model.module if ddp else model
        ckpt = os.path.join(args.out_dir, f"{args.stage}.pt")
        torch.save({"model": raw.state_dict()}, ckpt)
        csv_path = os.path.join(args.out_dir, f"{args.stage}_loss.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step", "loss"])
            for r in rows:
                w.writerow([r["step"], r["loss"]])
        print(f"[done] {args.stage}: loss {rows[0]['loss']:.4f} -> {rows[-1]['loss']:.4f}; "
              f"ckpt={ckpt}; csv={csv_path}", flush=True)
    if ddp:
        dist.destroy_process_group()


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse <= 1e-10 else 10.0 * math.log10(1.0 / mse)


def random_different(idx, n, dev):
    """Random action indices in [0,n), each != idx (for the controllability demo)."""
    r = torch.randint(0, n, idx.shape, device=dev)
    while torch.any(r == idx):
        r = torch.where(r == idx, torch.randint(0, n, idx.shape, device=dev), r)
    return r


def save_gif(path, gt, pred, scale=4):
    """gt,pred: (C,T,H,W) in [0,1]. Writes a GIF with GT on top, pred below."""
    import imageio.v2 as imageio
    T = gt.shape[1]
    frames = []
    for t in range(T):
        g = (gt[:, t].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        p = (pred[:, t].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        col = np.concatenate([g, p], axis=0)            # stack vertical (GT over pred)
        col = np.kron(col, np.ones((scale, scale, 1), dtype=np.uint8))  # upscale for visibility
        frames.append(col)
    imageio.mimsave(path, frames, duration=0.15, loop=0)


@torch.no_grad()
def rollout_stage(args, dev):
    model = construct_model(build_cfg("genie_redux", args.image_size, args.patch_size)).to(dev)
    state = torch.load(args.model_ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    tok = torch.load(args.tokenizer_ckpt, map_location="cpu")
    model.tokenizer.load_state_dict(tok["model"])
    model.eval()
    print(f"[rollout] loaded model={args.model_ckpt} tokenizer={args.tokenizer_ckpt}")

    x = load_clips(args.npz_dir, args.clip_len, args.max_clips, skip=args.holdout_skip).to(dev)
    print(f"[rollout] held-out clips {tuple(x.shape)}")
    os.makedirs(args.out_dir, exist_ok=True)

    n_prime, n_gen = 2, args.clip_len - 2
    psnr_true, psnr_rand, dpsnr = [], [], []
    bs = args.batch
    for i in range(0, x.shape[0], bs):
        vb = x[i:i + bs]
        lam_idx = model.latent_action_model(vb, return_only_codebook_ids=True)  # (B, T-1)
        prime = vb[:, :, :n_prime]
        gt = vb[:, :, n_prime:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.sample(prime_frames=prime, actions=lam_idx[:, :n_prime + n_gen - 1],
                                num_frames=n_gen, inference_steps=args.inference_steps).float()
            # controllability: shuffle the latent actions -> different rollout
            rand_idx = random_different(lam_idx, 7, dev)   # 7 = lam codebook_size
            pred_rand = model.sample(prime_frames=prime, actions=rand_idx[:, :n_prime + n_gen - 1],
                                     num_frames=n_gen, inference_steps=args.inference_steps).float()
        for j in range(vb.shape[0]):
            psnr_true.append(psnr(pred[j], gt[j]))
            psnr_rand.append(psnr(pred_rand[j], gt[j]))
            dpsnr.append(psnr_true[-1] - psnr_rand[-1])
            gidx = i + j
            if gidx < args.n_gifs:
                save_gif(os.path.join(args.out_dir, f"rollout_{gidx}.gif"), gt[j], pred[j])
        print(f"[rollout] batch {i//bs}: PSNR_true={np.mean(psnr_true):.2f} "
              f"PSNR_rand={np.mean(psnr_rand):.2f} dPSNR={np.mean(dpsnr):.2f}", flush=True)

    summary = dict(n=len(psnr_true), psnr_true=float(np.mean(psnr_true)),
                   psnr_rand=float(np.mean(psnr_rand)), delta_psnr=float(np.mean(dpsnr)))
    import json
    with open(os.path.join(args.out_dir, "rollout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[rollout DONE] {summary}; gifs+summary in {args.out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tokenizer", "genie", "rollout"])
    ap.add_argument("--npz-dir", default="/PATH_TO/aoe_npz/aoe26")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--max-clips", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--stream", action="store_true")          # streaming dataloader (large data that won't fit RAM)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", default="aoe_genie_out")
    ap.add_argument("--tokenizer-ckpt", default="aoe_genie_out/tokenizer.pt")
    ap.add_argument("--model-ckpt", default="aoe_genie_out/genie.pt")
    ap.add_argument("--save-every", type=int, default=2000)    # periodic ckpt during training
    ap.add_argument("--inference-steps", type=int, default=25)
    ap.add_argument("--train-eps", type=int, default=1100)     # training uses first N eps
    ap.add_argument("--holdout-skip", type=int, default=1100)  # rollout uses eps 1100+ (held out)
    ap.add_argument("--n-gifs", type=int, default=8)
    args = ap.parse_args()

    if args.stage == "rollout":
        rollout_stage(args, "cuda" if torch.cuda.is_available() else "cpu")
    else:
        ddp, rank, world, local, dev = ddp_setup()
        train_stage(args, ddp, rank, world, local, dev)


if __name__ == "__main__":
    main()

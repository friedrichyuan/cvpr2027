#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guided GenieRedux on AoE conditioned on the TRUE 26D action (hand 20D + camera 6D).

Reuses the already-trained tokenizer (action-agnostic) and trains the GUIDED dynamics, which
conditions on the raw continuous 26D action (MaskGIT concatenates the action vector — no discretization,
so use_action_embeddings=False + action_dim=26 feeds 26D straight in). Official models via construct_model,
npz fed directly — no fork (same precedent as genie_redux_full.py).

Rollout = the camera-controllability money shot: generate with (a) true 26D, (b) camera-6D shuffled
(hand kept true), (c) hand-20D shuffled (camera kept true). Δ-PSNR_cam vs Δ-PSNR_hand quantifies which
sub-action drives the generated egocentric video — expect camera ≫ hand (our core finding, now on generated video).

Stages:  guided  (train dynamics, load frozen tokenizer)   |   rollout  (camera vs hand ablation + GIFs)
DDP:  torchrun --nproc_per_node=N genie_redux_guided26d.py --stage guided ...
"""
import os, glob, csv, argparse, math, json
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.chdir(os.environ.get("GENIE_UPSTREAM_ROOT", os.path.dirname(os.path.abspath(__file__))))  # cwd = GenieRedux repo root (models/ + relative configs)
from omegaconf import OmegaConf
from einops import rearrange
from models import construct_model


def build_cfg_guided(image_size=64, action_dim=26, patch_size=4):
    common = dict(image_size=image_size, patch_size=patch_size, temporal_patch_size=1,
                  num_blocks=8, dim_head=64, heads=8, ff_mult=4,
                  vq_loss_weight=1.0, recons_loss_weight=1.0)
    return OmegaConf.create({
        "model": "genie_redux_guided",
        "train": {"wandb_mode": "disabled"},
        "tokenizer": dict(dim=512, codebook_size=1024, **common),
        "lam": dict(dim=512, codebook_size=7, **common),   # unused by guided, kept for construct_model
        "dynamics": dict(dim=512, action_dim=action_dim, image_size=image_size, patch_size=patch_size,
                         temporal_patch_size=1, num_blocks=12, dim_head=64, heads=8, ff_mult=4,
                         max_seq_len=8000, sample_temperature=1.0, sample_num_frames=15,
                         use_action_embeddings=False, use_token=False, is_guided=True),
    })


def load_clips_actions(npz_dir, clip_len, max_clips, skip=0, max_files=None, rank=0, world=1):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(npz_dir), "*.npz")))
    assert files, f"no npz in {npz_dir}"
    files = files[skip:]
    if max_files is not None:
        files = files[:max_files]
    if world > 1:                              # DDP: each rank reads an equal contiguous file shard
        per = len(files) // world              #   (avoids every rank loading the FULL set -> 4x RAM/swap death at 128)
        files = files[rank * per:(rank + 1) * per]
    vids, acts = [], []
    for f in files:
        d = np.load(f)
        img, act = d["image"], d["action"].astype(np.float32)   # (T,64,64,3) uint8, (T,26)
        n = img.shape[0] // clip_len
        for i in range(n):
            vids.append(img[i * clip_len:(i + 1) * clip_len])
            acts.append(act[i * clip_len:(i + 1) * clip_len])
            if len(vids) >= max_clips:
                break
        if len(vids) >= max_clips:
            break
    x = np.stack(vids).astype(np.float32) / 255.0
    a = np.stack(acts)                                            # (N,clip_len,26)
    x = torch.from_numpy(x).permute(0, 4, 1, 2, 3).contiguous()  # (N,C,T,H,W)
    return x, torch.from_numpy(a).contiguous()


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
        return True, rank, world, local, f"cuda:{local}"
    return False, 0, 1, 0, ("cuda" if torch.cuda.is_available() else "cpu")


def align_actions(a, n_action_frames):
    """a: (B,clip_len,26) -> (B,n_action_frames,26). Use the per-transition action (drop frame 0)."""
    return a[:, 1:1 + n_action_frames].contiguous()


def train_guided(args, ddp, rank, world, local, dev):
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    model = construct_model(build_cfg_guided(args.image_size, args.action_dim, args.patch_size)).to(dev)
    tok = torch.load(args.tokenizer_ckpt, map_location="cpu")
    model.tokenizer.load_state_dict(tok["model"])
    if is_main:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[guided] tokenizer loaded; trainable={n:.1f}M (dynamics only)", flush=True)

    n_af = args.clip_len - 1                       # transitions; drop frame 0
    if args.stream:                                # streaming loader (file-amortized, constant RAM -> scales to 100h+)
        from aoe_stream_dataset import StreamingClipIterable, count_clips
        from torch.utils.data import DataLoader
        total = count_clips(args.npz_dir, args.clip_len, max_files=args.train_eps)
        spe = max(1, total // (max(world, 1) * args.batch))            # fixed steps/epoch -> DDP-safe
        ds = StreamingClipIterable(args.npz_dir, args.clip_len, max_files=args.train_eps,
                                   with_action=True, world=world, rank=rank)
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=(args.num_workers > 0))
        _it = iter(dl)
        if is_main:
            print(f"[data] STREAM clips={total} steps/epoch={spe}; eff_batch={args.batch*world}", flush=True)
        def epoch_batches(ep):
            return (next(_it) for _ in range(spe))
    else:
        x, a = load_clips_actions(args.npz_dir, args.clip_len, args.max_clips, max_files=args.train_eps, rank=rank, world=world)
        if is_main:
            print(f"[data] per-rank vids {tuple(x.shape)} acts {tuple(a.shape)}; eff_batch={args.batch*world}", flush=True)
        def epoch_batches(ep):
            N = x.shape[0]; perm = torch.randperm(N)
            return ((x[perm[i:i + args.batch]], a[perm[i:i + args.batch]]) for i in range(0, N, args.batch))

    if ddp:
        model = DDP(model, device_ids=[local], find_unused_parameters=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    rows, step = [], 0
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for vb, ab in epoch_batches(ep):
            vb = vb.to(dev, non_blocking=True)
            ab = align_actions(ab, n_af).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(vb, actions=ab)
                if isinstance(loss, (tuple, list)):
                    loss = loss[0]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            tot += loss.item(); nb += 1; step += 1
            if is_main:
                rows.append({"step": step, "loss": loss.item()})
                if step % 20 == 0:
                    print(f"[guided ep{ep+1} step{step}] loss={loss.item():.4f}", flush=True)
                if step % args.save_every == 0:
                    raw_ = model.module if ddp else model
                    torch.save({"model": raw_.state_dict()}, os.path.join(args.out_dir, "guided.pt"))
        if is_main:
            print(f"[ep {ep+1}/{args.epochs}] mean loss={tot/nb:.4f}", flush=True)

    if is_main:
        raw = model.module if ddp else model
        torch.save({"model": raw.state_dict()}, os.path.join(args.out_dir, "guided.pt"))
        with open(os.path.join(args.out_dir, "guided_loss.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step", "loss"])
            for r in rows:
                w.writerow([r["step"], r["loss"]])
        print(f"[done] guided: {rows[0]['loss']:.4f} -> {rows[-1]['loss']:.4f}", flush=True)
    if ddp:
        dist.destroy_process_group()


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse <= 1e-10 else 10.0 * math.log10(1.0 / mse)


def save_gif(path, *rows_chw_t, scale=4):
    """each arg: (C,T,H,W) in [0,1]; stacked vertically per frame."""
    import imageio.v2 as imageio
    T = rows_chw_t[0].shape[1]
    frames = []
    for t in range(T):
        cols = [(r[:, t].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for r in rows_chw_t]
        col = np.concatenate(cols, axis=0)
        col = np.kron(col, np.ones((scale, scale, 1), dtype=np.uint8))
        frames.append(col)
    imageio.mimsave(path, frames, duration=0.15, loop=0)


@torch.no_grad()
def guided_sample(model, prime_frames, actions_cont, num_frames, inference_steps):
    """Like GenieReduxGuided.sample but feeds CONTINUOUS actions straight to dynamics (skip one-hot)."""
    prime_token_ids = model.get_tokenizer_codebook_ids(prime_frames)
    prime_token_ids = rearrange(prime_token_ids, "b ... -> b (...)")
    pf = prime_frames.shape[2]
    num_tokens = model.num_tokens_per_frames(num_frames, num_first_frames=pf)
    patch_shape = model.get_video_patch_shape(num_frames + pf, num_first_frames=pf)
    vtok = model.dynamics.sample(prime_token_ids=prime_token_ids, actions=actions_cont,
                                 num_tokens=num_tokens, patch_shape=patch_shape,
                                 inference_steps=inference_steps, mask_schedule="cosine",
                                 sample_temperature=1.0, tokenizer=model.tokenizer)
    vtok = torch.cat((prime_token_ids, vtok), dim=-1)
    video = model.decode_from_codebook_indices(vtok)
    return video[:, :, pf:].float()


@torch.no_grad()
def rollout_guided(args, dev):
    model = construct_model(build_cfg_guided(args.image_size, args.action_dim, args.patch_size)).to(dev)
    st = torch.load(args.model_ckpt, map_location="cpu"); model.load_state_dict(st["model"])
    tok = torch.load(args.tokenizer_ckpt, map_location="cpu"); model.tokenizer.load_state_dict(tok["model"])
    model.eval()
    x, a = load_clips_actions(args.npz_dir, args.clip_len, args.max_clips, skip=args.holdout_skip)
    x, a = x.to(dev), a.to(dev)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[rollout] held-out {tuple(x.shape)} acts {tuple(a.shape)}", flush=True)

    pf, ng = 2, args.clip_len - 2
    n_af = pf + ng - 1
    pt, pcam, phand, phand6 = [], [], [], []
    bs = args.batch
    for i in range(0, x.shape[0], bs):
        vb = x[i:i + bs]; ab = align_actions(a[i:i + bs], n_af)
        prime, gt = vb[:, :, :pf], vb[:, :, pf:]
        roll = torch.roll(ab, shifts=1, dims=0)                     # another clip's actions (shuffle source)
        HAND6 = [0, 1, 2, 3, 4, 5]                                  # 6 hand dims = dim-fair to the 6 camera dims
        cam_shuf  = ab.clone(); cam_shuf[..., 20:26]   = roll[..., 20:26]   # true hand + shuffled camera (6D)
        hand_shuf = ab.clone(); hand_shuf[..., 0:20]   = roll[..., 0:20]    # shuffled hand (20D) + true camera
        hand6_shuf = ab.clone(); hand6_shuf[..., HAND6] = roll[..., HAND6]  # shuffle 6 hand dims (dim-fair to camera)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p_true  = guided_sample(model, prime, ab,         ng, args.inference_steps)
            p_cam   = guided_sample(model, prime, cam_shuf,   ng, args.inference_steps)
            p_hand  = guided_sample(model, prime, hand_shuf,  ng, args.inference_steps)
            p_hand6 = guided_sample(model, prime, hand6_shuf, ng, args.inference_steps)
        for j in range(vb.shape[0]):
            pt.append(psnr(p_true[j], gt[j])); pcam.append(psnr(p_cam[j], gt[j]))
            phand.append(psnr(p_hand[j], gt[j])); phand6.append(psnr(p_hand6[j], gt[j]))
            g = i + j
            if g < args.n_gifs:
                save_gif(os.path.join(args.out_dir, f"guided_rollout_{g}.gif"), gt[j], p_true[j], p_cam[j])
        print(f"[rollout] b{i//bs}: true={np.mean(pt):.2f} camShuf={np.mean(pcam):.2f} "
              f"hand6Shuf={np.mean(phand6):.2f} hand20Shuf={np.mean(phand):.2f} | "
              f"Δcam={np.mean(pt)-np.mean(pcam):.2f} vs Δhand6={np.mean(pt)-np.mean(phand6):.2f} (dim-fair) "
              f"| Δhand20={np.mean(pt)-np.mean(phand):.2f}", flush=True)

    s = dict(n=len(pt), psnr_true=float(np.mean(pt)), psnr_cam_shuf=float(np.mean(pcam)),
             psnr_hand6_shuf=float(np.mean(phand6)), psnr_hand20_shuf=float(np.mean(phand)),
             delta_psnr_camera=float(np.mean(pt) - np.mean(pcam)),
             delta_psnr_hand6_dimfair=float(np.mean(pt) - np.mean(phand6)),
             delta_psnr_hand20=float(np.mean(pt) - np.mean(phand)),
             camera_vs_hand6_ratio=float((np.mean(pt) - np.mean(pcam)) / max(1e-6, np.mean(pt) - np.mean(phand6))))
    with open(os.path.join(args.out_dir, "guided_rollout_summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    print(f"[rollout DONE] {s}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["guided", "rollout"])
    ap.add_argument("--npz-dir", default="/PATH_TO/aoe_npz/aoe26")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--max-clips", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--stream", action="store_true")          # streaming dataloader (large data that won't fit RAM)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--action-dim", type=int, default=26)
    ap.add_argument("--out-dir", default="aoe_guided26d")
    ap.add_argument("--tokenizer-ckpt", default="aoe_genie_full/tokenizer.pt")
    ap.add_argument("--model-ckpt", default="aoe_guided26d/guided.pt")
    ap.add_argument("--inference-steps", type=int, default=25)
    ap.add_argument("--train-eps", type=int, default=1100)
    ap.add_argument("--holdout-skip", type=int, default=1100)
    ap.add_argument("--n-gifs", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()
    if args.stage == "rollout":
        rollout_guided(args, "cuda" if torch.cuda.is_available() else "cpu")
    else:
        ddp, rank, world, local, dev = ddp_setup()
        train_guided(args, ddp, rank, world, local, dev)


if __name__ == "__main__":
    main()

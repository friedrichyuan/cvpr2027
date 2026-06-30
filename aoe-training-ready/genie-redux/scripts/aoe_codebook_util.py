#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Go/no-go gate: measure VQ codebook utilization of the (collapse-fixed) GenieRedux tokenizer.

The POC full-data tokenizer used 111 distinct codes; a collapsed one uses ~16-43. Before
training the genie/guided stages on top of a tokenizer, check it beats the POC level (>= 111
used codes) with a healthy perplexity. Run with the same VQ env as training
(`AOE_VQ_EMA=1 AOE_VQ_DEADCODE=2`) and `GENIE_UPSTREAM_ROOT` pointing at the GenieRedux checkout:

  IMAGE_SIZE=64 PATCH_SIZE=4 OPEN_AOE_NPZ_DIR=/PATH_TO/aoe_npz/aoe26 \
      python aoe_codebook_util.py /PATH_TO/aoe_runs/genie/tokenizer.pt
"""
import os, sys, glob, random, math, collections
import numpy as np, torch

os.chdir(os.environ.get("GENIE_UPSTREAM_ROOT", os.path.dirname(os.path.abspath(__file__))))  # cwd = GenieRedux repo root
from omegaconf import OmegaConf
from models import construct_model

IMG = int(os.environ.get("IMAGE_SIZE", "64"))
PS = int(os.environ.get("PATCH_SIZE", "4"))
NPZ_DIR = os.path.expanduser(os.environ.get("OPEN_AOE_NPZ_DIR", "/PATH_TO/aoe_npz/aoe26"))


def build_cfg(model_name, image_size=IMG, patch_size=PS):
    common = dict(image_size=image_size, patch_size=patch_size, temporal_patch_size=1,
                  num_blocks=8, dim_head=64, heads=8, ff_mult=4,
                  vq_loss_weight=1.0, recons_loss_weight=1.0)
    return OmegaConf.create({
        "model": model_name, "train": {"wandb_mode": "disabled"},
        "tokenizer": dict(dim=512, codebook_size=1024, **common),
        "lam": dict(dim=512, codebook_size=7, **common),
        "dynamics": dict(dim=512, action_dim=5, image_size=image_size, patch_size=patch_size,
                         temporal_patch_size=1, num_blocks=12, dim_head=64, heads=8, ff_mult=4,
                         max_seq_len=8000, sample_temperature=1.0, sample_num_frames=15,
                         use_action_embeddings=True, use_token=False, is_guided=False),
    })


K = 1024
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE", dev, "GPU", torch.cuda.get_device_name(0) if dev == "cuda" else "-")
model = construct_model(build_cfg("tokenizer", IMG, PS)).to(dev)
ckp = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "/PATH_TO/aoe_runs/genie/tokenizer.pt")
ck = torch.load(ckp, map_location="cpu")
model.load_state_dict(ck["model"]); model.eval()
print("LOADED", ckp)

fs = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
assert fs, f"no npz in {NPZ_DIR}"
random.seed(0); fs = random.sample(fs, min(60, len(fs)))
clip_len, want = 16, 96
clips = []
for f in fs:
    img = np.load(f)["image"]
    for i in range(img.shape[0] // clip_len):
        clips.append(img[i * clip_len:(i + 1) * clip_len])
        if len(clips) >= want:
            break
    if len(clips) >= want:
        break
x = torch.from_numpy(np.stack(clips).astype(np.float32) / 255.0).permute(0, 4, 1, 2, 3).contiguous()
print("CLIPS", tuple(x.shape))

hist = collections.Counter()
bs = 4
with torch.no_grad():
    for i in range(0, x.shape[0], bs):
        vb = x[i:i + bs].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model(vb, return_only_codebook_ids=True)
        if isinstance(ids, (tuple, list)):
            ids = ids[0]
        hist.update(ids.reshape(-1).detach().cpu().numpy().tolist())

tot = sum(hist.values()); used = len(hist)
p = np.array([c / tot for c in hist.values()])
ppl = math.exp(-(p * np.log(p + 1e-12)).sum())
top10 = sum(c for _, c in hist.most_common(10))
print("=================== CODEBOOK UTIL ===================")
print("TOKENS_TOTAL", tot)
print("UNIQUE_CODES_USED %d / %d  (%.1f%%)" % (used, K, 100 * used / K))
print("PERPLEXITY %.1f  (effective #codes; max=%d)" % (ppl, K))
print("TOP10_CODE_SHARE %.1f%%" % (100 * top10 / tot))
print("REF: POC=111 used / calib-collapse=43 used")
print("GO_NO_GO:", "PASS (>=111)" if used >= 111 else "FAIL (<111 -> still collapsed)")

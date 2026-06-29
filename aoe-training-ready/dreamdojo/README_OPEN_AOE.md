# DreamDojo on Open-AoE (zero-shot preview)

Open-AoE integration recipe for [NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo)
(Cosmos-Predict2.5) — runs the SOTA world model **zero-shot** on Open-AoE to verify the
data pipeline + inference on accessible hardware. This is an integration recipe; the
upstream is an external checkout, **not** vendored.

> **Status: ⚠️ zero-shot only.** The data pipeline and inference work; high-quality
> results need post-training (out of reach without 8×H100 — see *Post-train* below).

## Version

- Upstream: [NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo) (Cosmos-Predict2.5) — verified commit `02f119b`
- Patch: `patches/open_aoe_support.patch` (1 line — pins a checkpoint `revision` to `main` in `checkpoint_db.py`, a 404 fix). All other infra are environment setup (below), not code changes.

## Environment

```bash
git clone https://github.com/NVIDIA/DreamDojo
cd DreamDojo && git checkout 02f119b
git apply /PATH_TO/Open-AoE-dev/aoe-training-ready/dreamdojo/patches/open_aoe_support.patch
# follow the upstream install (uv sync). Known extra infra for inference:
#   - pure-torch pytorch3d shim on PYTHONPATH (groot_dreams only imports transforms)
#   - torchcodec needs FFmpeg libs (symlink clean sonames + LD_LIBRARY_PATH); ffmpeg binary on PATH
#   - export HF_TOKEN=<your_token>
export DREAMDOJO_UPSTREAM_ROOT=/PATH_TO/DreamDojo
```

## Data conversion

AoE undistorted video → 480×640 MP4 (the resolution DreamDojo hard-codes). The output
folder name must **not** contain `gr1`/`g1`/`yam`/`agibot`, so DreamDojo routes it to the
generic `VideoDataset` and the LAM infers latent actions from the video.

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/Open-AoE/poc_deliver
export DD_VIDEO_DIR=/PATH_TO/aoe_dd_videos
export FRAMES=150        # must exceed the inference --num-frames
./scripts/aoe_train convert
```

## Zero-shot inference

Needs the DreamDojo environment and a **≥45GB GPU** (2B teacher peaks ~31GB; a 24GB 4090 is
insufficient for the WAN-VAE decode). `cp_size=1`, so multi-GPU does not help.

```bash
export DD_CKPT_DIR=/PATH_TO/hf_ckpt/2B_GR1_post-train
export SAVE_DIR=/PATH_TO/aoe_runs/dreamdojo_aoe
export CUDA_VISIBLE_DEVICES=0
./scripts/aoe_train --dry-run infer   # prints the upstream inference command
./scripts/aoe_train infer
```

## Results (zero-shot, GR1 robot post-train weights)

- PSNR `10.01` / SSIM `0.214` / LPIPS `0.702`: scene + hands qualitatively preserved, pixel
  fidelity low — a **robot→human domain gap**, not a data-format problem.

## Post-train (deferred — needs 8×H100)

To close the domain gap, post-train on AoE MANO actions (mapped to DreamDojo's action vector,
dims 220–351). The data pipeline is ready; blocked on the MANO `.pkl` + 8×H100. Distilled
student weights are not released by upstream (only 2B/14B pretrain + per-robot post-train + LAM).

## Notes

- 480×640 is mandatory (position encodings are hard-coded; lower resolution NaNs/black-screens).
- Each clip must have **> `--num-frames`** frames (else the sampler raises `empty range for randrange`).

## License

Apache-2.0 (Open-AoE Contributors) for the Open-AoE converter/launcher. The upstream
DreamDojo / Cosmos-Predict2.5 is NVIDIA's and is referenced by checkout (not vendored); the
patch only annotates a checkpoint revision. See the root [`LEGAL.md`](../../LEGAL.md) and the
upstream license for terms.

## File reference

| File | Purpose |
|------|---------|
| `scripts/aoe_train` | Launcher: `convert` / `infer` |
| `scripts/aoe_to_dreamdojo.py` | AoE undistorted video → 480×640 MP4 for DreamDojo |
| `patches/open_aoe_support.patch` | 1-line upstream patch (checkpoint revision pin) |

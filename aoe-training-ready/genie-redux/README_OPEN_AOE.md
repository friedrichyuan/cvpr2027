# GenieRedux on Open-AoE

Open-AoE integration recipe for [insait-institute/GenieRedux](https://github.com/insait-institute/GenieRedux)
(CVPR 2025, an engineered Genie reproduction) — a Genie-style latent-action world model on
Open-AoE egocentric video. This is an integration recipe; the upstream is an external
checkout, **not** vendored. AoE clips are fed straight to the official models (bypassing only the
retro `MultiEnvironmentDataset`); the only upstream source change is a small **VQ codebook-collapse
fix** (`patches/open_aoe_support.patch`), required for the full-Genie tokenizer at scale.

Three legs:
1. **LAM** — the latent-action model core (ST-ViViT encoder + VQ latent actions + decoder).
2. **Full Genie** — tokenizer + LAM + dynamics → generates *controllable* rollout video.
3. **Guided-26D** — conditions dynamics on the **true 26D action** and, via a dim-fair ablation,
   shows the camera 6D dominates controllability *per dimension* (camera : hand ≈ 2.85×, hardened) —
   a clean confirmation that egocentric controllability is driven mainly by camera ego-motion.

## Version

- Upstream: [insait-institute/GenieRedux](https://github.com/insait-institute/GenieRedux) — verified commit `bc80fe4`
- Patch: `patches/open_aoe_support.patch` — VQ codebook-collapse fix (EMA + dead-code revival, env-gated); required for the full-Genie tokenizer at 100h scale. The LAM-only leg works without it.

## Environment

```bash
git clone https://github.com/insait-institute/GenieRedux
cd GenieRedux && git checkout bc80fe4
git apply /PATH_TO/Open-AoE-dev/aoe-training-ready/genie-redux/patches/open_aoe_support.patch  # VQ collapse fix
# build the upstream env (their genie_redux_env.yaml; python 3.13 / torch 2.8) + einops,
# omegaconf, hydra-core, imageio[ffmpeg], matplotlib
export GENIE_UPSTREAM_ROOT=/PATH_TO/GenieRedux
```

The launcher puts `$GENIE_UPSTREAM_ROOT` (the `models` package) and this recipe's `scripts/`
(the `aoe_stream_dataset` loader) on `PYTHONPATH`, and runs with cwd = the upstream root.

## Data format

AoE npz per clip: `image (T,64,64,3) uint8` + `action (T,26) float32` (hand 20D + camera 6D —
see [`../ACTION_SPEC.md`](../ACTION_SPEC.md)). The full/LAM legs use frames only; the guided leg
uses the 26D action. For large datasets, set `STREAM=1` to use the file-amortized streaming
loader (`aoe_stream_dataset.py`, constant RAM, DDP file-sharded — scales to 100h+).

## Train

```bash
export OPEN_AOE_NPZ_DIR=/PATH_TO/aoe_npz/aoe26
export OUT_DIR=/PATH_TO/aoe_runs/genie
export NP=3 CUDA_VISIBLE_DEVICES=0,1,2     # GenieRedux's ST-ViViT needs multi-GPU DDP for a real batch

# 1) latent-action model only (single GPU)
./scripts/aoe_train --dry-run lam
./scripts/aoe_train lam

# 2) full generative Genie: tokenizer -> genie(LAM+dynamics) -> rollout video -> plot
./scripts/aoe_train --dry-run full
./scripts/aoe_train full

# 3) guided-26D (reuses $OUT_DIR/tokenizer.pt): train -> camera-vs-hand ablation -> plot
./scripts/aoe_train guided-all
```

Individual stages are also exposed: `downsample`, `tokenizer`, `gate`, `genie`, `rollout`, `guided`,
`guided-rollout`. Per-stage knobs (`TOK_BATCH`, `GENIE_EPOCHS`, `STREAM`, …) are env vars; extra
trainer flags can be appended after the subcommand.

## Outputs

```text
$OUT_DIR/tokenizer.pt  $OUT_DIR/genie.pt  $OUT_DIR/*_loss.csv  $OUT_DIR/rollout_*.gif
$OUT_DIR/guided/guided.pt  $OUT_DIR/guided/guided_rollout_summary.json
```

## Results

POC (2.9h subset):
- **LAM** (168.7M, codebook 7): loss `0.634 → 0.0024` (6 ep, 4×4090 DDP) — first-try convergence.
- **Full Genie**: tokenizer loss `0.668 → 0.0038`; genie token CE `7.07 → 2.06`; rollout PSNR `16.6 dB`, controllability Δ-PSNR `+0.54 dB`.
- **Guided-26D**: token CE `7.12 → 1.86`.

100-hour run (128-res; hardened eval = seeded random-permutation shuffle, per-clip paired Δ, bootstrap CI95, LPIPS, n=128, two seeds):
- **Full Genie** rollout: PSNR `19.5 dB`, controllability Δ-PSNR `+0.66 dB` (true − shuffled latent).
- **Guided-26D** dim-fair ablation: **camera : hand ≈ 2.85×** (two seeds 2.70× / 2.98×, CI95 ~[1.9, 4.7]).
  - This is a **per-dimension** statement: by *total* signal Δhand20 (≈0.22) slightly exceeds Δcam (≈0.15) — the 20-D hand carries more aggregate signal, but each camera dim matters far more than each hand dim ("camera dominates" ≠ "hand doesn't matter").
  - The earlier POC headline "8.3×" was inflated by a deterministic roll-shuffle + single run + n=64, and is superseded by the hardened ~2.85×.

## Notes

- The 168M ST-ViViT OOMs above ~batch 2 per 24GB GPU; multi-GPU DDP gives a real effective batch. `find_unused_parameters=True` is required (already set).
- The tokenizer must finish before the genie/guided stages (they load the frozen `tokenizer.pt`).
- The guided rollout uses a hardened controllability protocol (seeded random-permutation shuffle, per-clip paired Δ-PSNR with bootstrap CI95, optional LPIPS, n=128); set the seed via `SEED=...`. It writes `guided_rollout_summary.json` (incl. `ratio_bootstrap_ci95`) and `guided_rollout_perclip.npz`.
- **VQ codebook collapse (key fix at 100-hour scale):** the upstream tokenizer VQ (`models/components/stvivit.py`) is a learnable cosine codebook with no EMA and no dead-code revival, so at 100h it collapses to ~16–20/1024 codes (perplexity ~9). The shipped `patches/open_aoe_support.patch` makes it env-gated; set `AOE_VQ_EMA=1 AOE_VQ_DEADCODE=2` (the launcher exports these by default) → 1024/1024 codes, perplexity ~792. Kmeans-init alone (→35) or revival-without-EMA (→4) do not work, and kmeans deadlocks under DDP. These env vars must be set for **every** stage. Before training genie, gate on codebook utilization ≥ 111 (POC level): `./scripts/aoe_train gate`.

## License

Apache-2.0 (Open-AoE Contributors). Upstream GenieRedux is referenced by checkout (not vendored);
see the root [`LEGAL.md`](../../LEGAL.md).

## File reference

| File | Purpose |
|------|---------|
| `scripts/aoe_train` | Launcher: `lam` / `full` / `guided-all` (+ individual stages) |
| `scripts/genie_redux_lam.py` | Thin LAM trainer (AoE clips → `LatentActionModel`) |
| `scripts/genie_redux_full.py` | Full Genie: `tokenizer` / `genie` / `rollout` stages (DDP) |
| `scripts/genie_redux_guided26d.py` | Guided-26D trainer + camera-vs-hand rollout ablation |
| `scripts/aoe_stream_dataset.py` | File-amortized streaming clip loader (constant RAM, DDP-sharded) |
| `scripts/aoe_downsample.py` | Resolution downsampler (keeps the action vector) |
| `scripts/aoe_codebook_util.py` | Codebook-utilization go/no-go gate (≥ 111 used codes before genie) |
| `scripts/plot_genie_full.py` / `plot_guided.py` | Loss-curve plots |
| `patches/open_aoe_support.patch` | Upstream VQ codebook-collapse fix (EMA + dead-code revival, env-gated) |

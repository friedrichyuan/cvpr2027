# GenieRedux on Open-AoE

Open-AoE integration recipe for [insait-institute/GenieRedux](https://github.com/insait-institute/GenieRedux)
(CVPR 2025, an engineered Genie reproduction) — a Genie-style latent-action world model on
Open-AoE egocentric video. This is an integration recipe; the upstream is an external
checkout, **not** vendored, and **no upstream source patch is required** (AoE clips are fed
straight to the official models, bypassing only the retro `MultiEnvironmentDataset`).

Three legs:
1. **LAM** — the latent-action model core (ST-ViViT encoder + VQ latent actions + decoder).
2. **Full Genie** — tokenizer + LAM + dynamics → generates *controllable* rollout video.
3. **Guided-26D** — conditions dynamics on the **true 26D action** and, via a dim-fair ablation,
   pins controllability to the camera 6D (camera : hand = 8.3×) — the cleanest confirmation that
   camera ego-motion dominates egocentric controllability.

## Version

- Upstream: [insait-institute/GenieRedux](https://github.com/insait-institute/GenieRedux) — verified commit `bc80fe4`
- Patch: none

## Environment

```bash
git clone https://github.com/insait-institute/GenieRedux
cd GenieRedux && git checkout bc80fe4
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

Individual stages are also exposed: `downsample`, `tokenizer`, `genie`, `rollout`, `guided`,
`guided-rollout`. Per-stage knobs (`TOK_BATCH`, `GENIE_EPOCHS`, `STREAM`, …) are env vars; extra
trainer flags can be appended after the subcommand.

## Outputs

```text
$OUT_DIR/tokenizer.pt  $OUT_DIR/genie.pt  $OUT_DIR/*_loss.csv  $OUT_DIR/rollout_*.gif
$OUT_DIR/guided/guided.pt  $OUT_DIR/guided/guided_rollout_summary.json
```

## Results (POC, 2.9h subset)

- **LAM** (168.7M, codebook 7): loss `0.634 → 0.0024` (full data, 6 ep, 4×4090 DDP) — first-try convergence.
- **Full Genie**: tokenizer (101.6M) loss `0.668 → 0.0038`; genie (LAM+dynamics, 249.7M) token CE `7.07 → 2.06`; rollout PSNR `16.60 dB`, controllability Δ-PSNR `+0.54 dB` (true − shuffled latent action).
- **Guided-26D**: token CE `7.12 → 1.86`; dim-fair ablation Δ-PSNR camera `0.81` vs hand-6 `0.10` → **camera : hand = 8.3×**.

## Notes

- The 168M ST-ViViT OOMs above ~batch 2 per 24GB GPU; multi-GPU DDP gives a real effective batch. `find_unused_parameters=True` is required (already set).
- The tokenizer must finish before the genie/guided stages (they load the frozen `tokenizer.pt`).
- VQ codebook collapse: if the tokenizer collapses to few active codes, enable EMA + dead-code revival in the upstream VQ config before training the genie stage.

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
| `scripts/plot_genie_full.py` / `plot_guided.py` | Loss-curve plots |

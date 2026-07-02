# iVideoGPT on Open-AoE

This directory is an Open-AoE integration recipe that adapts
[thuml/iVideoGPT](https://github.com/thuml/iVideoGPT) into an **action-conditioned**
egocentric video world model on Open-AoE data. It is an integration recipe, **not** a
vendored copy of iVideoGPT: the upstream ships as an external checkout, and only the
Open-AoE converters / launcher / eval and a 3-line patch live here.

Headline result: **camera ego-motion is the key to egocentric controllability** —
conditioning on 26D (hand 20D + camera 6D) is strongly controllable; hand-only 20D is not.

## Version

- Upstream: [thuml/iVideoGPT](https://github.com/thuml/iVideoGPT) — verified commit `d601d5c`
- Patch: `patches/open_aoe_support.patch` (3 lines — wires `aoe26`/`aoe20`/`aoe` into `ivideogpt/data/dataset_mixes.py`)

## Environment

```bash
git clone https://github.com/thuml/iVideoGPT
cd iVideoGPT && git checkout d601d5c
git apply /PATH_TO/Open-AoE-dev/aoe-training-ready/ivideogpt/patches/open_aoe_support.patch
# follow the upstream install guide; see Notes for required version pins
export IVIDEOGPT_UPSTREAM_ROOT=/PATH_TO/iVideoGPT
```

## Data format

The action-conditioned converter writes one npz per atomic-action clip:

```text
image   (T, res, res, 3) uint8   # decoded undistorted frames
action  (T, D) float32           # pre-normalized by per-dim stats (model does not normalize)
```

`D = 20` (hand) or `D = 26` (hand + camera 6D). Layout: [`../ACTION_SPEC.md`](../ACTION_SPEC.md).

## Data conversion

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/Open-AoE/poc_deliver
export OPEN_AOE_NPZ_ROOT=/PATH_TO/aoe_npz_ac     # npz go to $OPEN_AOE_NPZ_ROOT/<dataset_name>
export DATASET_NAME=aoe26
export DIM=26
./scripts/aoe_train stats         # 1) action stats (no video decode)
./scripts/aoe_train convert       # 2) export normalized act-cond npz (decodes video)
./scripts/aoe_train convert-free  # optional: action-free frames (tokenizer eval / baseline)
```

## Train

Place the npz where iVideoGPT's loader resolves the `aoe26` mix, then:

```bash
export STEPS=60000
export NP=4
export OUTPUT_DIR=/PATH_TO/aoe_runs/ivideogpt_aoe26
./scripts/aoe_train --dry-run train     # prints the accelerate command
./scripts/aoe_train train --per_device_train_batch_size 8 --learning_rate 1e-4 --context_length 2
```

Extra `train_gpt.py` flags (batch / lr / context_length / tokenizer ckpt) can be appended after `train`.

## Eval

```bash
export TOKENIZER_DIR=/PATH_TO/ft_tokenizer
export TRANSFORMER_DIR=/PATH_TO/aoe_runs/ivideogpt_aoe26/checkpoints/checkpoint_60000
export NPZ_GLOB='/PATH_TO/aoe_npz_ac/aoe26/*.npz'
./scripts/aoe_train eval       # controllability: true vs shuffle vs zero action
./scripts/aoe_train tok-eval   # tokenizer reconstruction quality
```

`true << shuffle/zero` on teacher-forced loss ⇒ the model genuinely uses the action signal.

## Outputs

```text
$OUTPUT_DIR/checkpoints/checkpoint_<step>/   # accelerate checkpoints
```

## Notes

- Upstream version pins (the AoE POC env): `numpy==1.26.4`, `huggingface_hub==0.20.3` (diffusers needs `cached_download`), `datasets==2.18.0`, `peft==0.10.0`; the i3d file is mandatory (the Evaluator is instantiated unconditionally).
- Act-conditioned `HeadModelWithAction` has no `save_pretrained`, so the final save crashes; the periodic accelerate checkpoint holds the weights, and the eval reads `vocab_size` back from `llm.lm_head.weight`.
- `--context_length` must equal the tokenizer's (`= 2` for the 64px act-free checkpoint).
- The eval scripts import the upstream `ivideogpt` package + `inference/utils.py`; run them with `IVIDEOGPT_UPSTREAM_ROOT` set (the launcher puts it on `PYTHONPATH`).

## Results

POC (2.9h subset, 30k steps, 4×4090):
- Training loss `2.78 → 1.63`.
- Controllability: zero-action loss exceeds true-action by `25% → 49% → 62%` (10k/20k/30k) — monotonic; hand-only 20D ≈ −3%.
- Tokenizer AoE-finetune (10k): `+1.6 dB PSNR`, `−36% LPIPS`.

At 100 hours (60k steps): controllability is **weaker** than the 2.9h proof-of-concept (true / shuffle / zero teacher-forced loss `1.85 / 2.09 / 2.25`; the zero÷true loss ratio falls `1.62 → 1.22`). This is likely a **data-quality** effect — the 100h set uses consumer-phone (vivo/OPPO/Xiaomi) monocular-SLAM camera trajectories, noisier than the POC's HoloLens — and is still under investigation (a validity/NaN audit of the 100h data is the open follow-up), not a known model regression.

## License

Apache-2.0 (Open-AoE Contributors). Upstream iVideoGPT is referenced by checkout (not vendored);
see its repository for license terms and the root [`LEGAL.md`](../../LEGAL.md).

## File reference

| File | Purpose |
|------|---------|
| `scripts/aoe_train` | Launcher: `stats` / `convert` / `convert-free` / `train` / `eval` / `tok-eval` |
| `scripts/aoe_to_ivideogpt_actcond.py` | AoE → action-conditioned npz (image + normalized 20D/26D action) + stats |
| `scripts/aoe_to_ivideogpt_npz.py` | AoE → action-free npz (frames only) |
| `scripts/aoe_to_lerobot.py` | Master converter reused for geometry / episode parsing |
| `scripts/actcond_eval.py` | Controllability eval (true vs shuffle vs zero action) |
| `scripts/tokenizer_recon_eval.py` | Tokenizer reconstruction quality (PSNR/SSIM/LPIPS) |
| `patches/open_aoe_support.patch` | 3-line upstream patch registering the AoE data mixes |

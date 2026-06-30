# SmolVLA on Open-AoE

This directory is an Open-AoE integration recipe for fine-tuning **SmolVLA**
([`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), a 450M
vision-language-action policy) on Open-AoE egocentric hand data — the VLA leg of
the downstream toolchain. It is an integration recipe, **not** a vendored copy of
LeRobot or SmolVLA: the upstream model ships as the `lerobot` pip package plus the
HuggingFace base checkpoint, and only the Open-AoE converter / launcher / eval live
here under `scripts/`.

## Version

- Upstream library: [LeRobot](https://github.com/huggingface/lerobot) — verified `0.4.4`
- Base model: [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (downloaded from HuggingFace)
- Patch: none (no upstream source change required)

## Environment

```bash
pip install -r requirements.txt   # installs lerobot==0.4.4 + glue deps
# HuggingFace must be reachable directly (do NOT set HF_ENDPOINT=hf-mirror)
```

## Data format

The converter writes a LeRobot v2.1 dataset:

```text
observation.images.ego   # egocentric RGB
observation.state        # 22D dual-hand EEF state (camera frame)
action                   # 20D (hand) or 26D (hand + camera 6D)
task                     # bilingual atomic-action instruction
```

The 26D action = hand 20D + camera 6D self-motion `inv(cam_c2w[t]) @ cam_c2w[t+1]`
(translation 3 + axis-angle 3). Full layout: [`../ACTION_SPEC.md`](../ACTION_SPEC.md).

## Data conversion

Two paths. The **fast path** reuses pre-decoded `aoe26` npz frames (PNG,
`use_videos=False`, no SVT-AV1 re-encode → seconds):

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/Open-AoE/poc_deliver   # raw segments (for episode ordering)
export OPEN_AOE_NPZ_DIR=/PATH_TO/aoe_npz/aoe26           # pre-decoded image+action26 npz
export OPEN_AOE_LEROBOT_ROOT=/PATH_TO/lerobot_aoe/aoe26  # output dataset
export OPEN_AOE_REPO_ID=aoe26
export DIM=26                                            # 20 = hand only, 26 = hand + camera
./scripts/aoe_train convert
```

The **slow path** decodes raw videos into a video-backed dataset
(`aoe_to_lerobot.py --full`, SVT-AV1, hours):

```bash
./scripts/aoe_train convert-full
```

Optional schema sanity check:

```bash
./scripts/aoe_train verify
```

## Train

Dry-run prints the exact `lerobot-train` command without running it:

```bash
./scripts/aoe_train --dry-run train
```

Fine-tune:

```bash
export AOE_RESULTS_ROOT=/PATH_TO/aoe_runs
export CUDA_VISIBLE_DEVICES=0
export STEPS=40000
./scripts/aoe_train train
```

Extra `lerobot-train` overrides can be appended after `train`.

## Offline eval

Held-out flow-matching loss + per-dimension R² (splits hand 20D vs camera 6D for
the 26D variant):

```bash
./scripts/aoe_train eval
```

## Outputs

```text
$AOE_RESULTS_ROOT/smolvla_open_aoe/                       # checkpoints
$AOE_RESULTS_ROOT/smolvla_open_aoe/logs/smolvla.log
$AOE_RESULTS_ROOT/smolvla_open_aoe/loss_curves/smolvla_loss_curves.png
```

## Notes

- LeRobot import paths are `lerobot.datasets` / `lerobot.policies` (not `lerobot.common.*`); the CLI is `lerobot-train`.
- SmolVLA base expects camera keys `camera1/2/3`; the launcher passes `--rename_map` to map `observation.images.ego → observation.images.camera1` (validation passes when the provided set ⊆ expected; the model pads the rest).
- Offline eval calls `make_pre_post_processors(...)` then `pre(raw_batch)` **before** `forward` / `predict_action_chunk` (rename + normalize + tokenize).
- Data-speed trap: `convert-full` re-encodes every raw video with SVT-AV1 (hours); prefer `convert` (reuse npz, `use_videos=False`) → seconds, at 64×64.

## Results (POC, 2.9h subset)

- Held-out flow-matching loss `1.03 → 0.43` (aoe20, hand) / `0.51` (aoe26, hand+camera); no overfitting.
- 256×256 gives a marginal gain at 2.7× cost → the bottleneck is data size/task, not resolution.

## License

Apache-2.0 (Open-AoE Contributors). Upstream LeRobot/SmolVLA are Apache-2.0; see the
root [`LEGAL.md`](../../LEGAL.md). Base weights are downloaded from HuggingFace under
their model-card license.

## File reference

| File | Purpose |
|------|---------|
| `scripts/aoe_train` | Launcher: `convert` / `convert-full` / `verify` / `train` / `eval` / `curves` |
| `scripts/aoe_npz_to_lerobot.py` | Fast converter: reuse decoded npz → LeRobot (no AV1) |
| `scripts/aoe_to_lerobot.py` | Full converter: AoE raw data → LeRobot v2.1 (master converter) |
| `scripts/smolvla_eval.py` | Offline held-out eval (flow-matching loss + R²) |
| `scripts/verify_lerobot.py` | Dataset schema sanity check |
| `scripts/plot_loss_curves.py` | Parse training log → loss-curve CSV/PNG |

## H-RDT Recipe — Pre-train H-RDT with AoE Ego-Centric Data

This recipe converts [AoE] ego-centric hand manipulation data into [H-RDT](https://github.com/HongzheBi/H_RDT) pre-training format and adds an `aoe` dataset to H-RDT's data-loading and training pipeline via a one-command patch.

### Overview

**What this recipe does:**
- Converts AoE data (MANO hand reconstructions + action annotations + ego videos) into
  per-segment training samples: 48D bimanual hand actions, undistorted video, and T5
  language embeddings.
- Provides a one-command patch (`hrdt_aoe_support.patch`) that drops the `aoe` dataset
  loader into H-RDT and wires up dataset routing, an AoE pre-train config, and a launch
  script.
- Ships the data-processing scripts (precompute → statistics → language encoding) that
  build a ready-to-train dataset.

**What this recipe does NOT do / ship:**
- It does **not** modify H-RDT's model or core training logic — the patch only adds
  `aoe` support.
- It does **not** vendor H-RDT or HaWoR source code — both are cloned from upstream.

The recipe folder lives inside the AoE project. The H-RDT repository is cloned **independently, anywhere** — the recipe is never copied into it; they are linked at runtime via the `HRDT_PROJECT_ROOT` environment variable.

### Prerequisites

1. **H-RDT repository**: Clone from https://github.com/HongzheBi/H_RDT (any location).
2. **HaWoR**: Clone from https://github.com/ThunderVVV/HaWoR into this recipe folder
   (used for MANO forward kinematics) — see Step 2.
3. **Model weights**: T5-v1_1-XXL (language encoding) and dino-siglip (vision encoder).
4. **Python environment**: the H-RDT conda env (`conda activate hrdt`) plus
   `chumpy==0.70` (added by the patch's `requirements.txt`).
5. **AoE data**: ego-centric clips in the standard AoE layout (each clip with
   `ego_process/ego_hands_reconstruction/hands.npz`).

The 48D action vector produced by this recipe is, per hand (24D each):
`translation(3) + rotation_6d(6) + fingertips(15)` → left(24) + right(24).

### Quick Start

#### Step 1: Patch the H-RDT repository

Clone H-RDT anywhere and apply the patch from its root:

```bash
git clone https://github.com/HongzheBi/H_RDT.git
cd H_RDT
git apply /path/to/recipe/hrdt_aoe_support.patch
```

The patch makes the following changes to H-RDT:

| File | Change |
|------|--------|
| `datasets/aoe_dataset.py` | **New** — the AoE dataset loader |
| `datasets/dataset.py` | Adds `aoe` dataset routing + single-camera image handling |
| `main.py` | Adds `--dataset_name` and `--data_root` CLI args |
| `train/train.py` | Forwards `dataset_name` / `data_root` to the dataset |
| `models/encoder/dinosiglip_vit.py` | Resolves dino-siglip weights via `DINO_SIGLIP_DIR` |
| `models/encoder/t5_encoder.py` | Removes the hard-coded T5 path; accepts any path / HF id |
| `datasets/pretrain/encode_lang_batch.py` | Full-traceback error logging |
| `requirements.txt` | Adds `chumpy==0.70` (MANO dependency) |
| `configs/hrdt_aoe_pretrain.yaml` | **New** — AoE pre-train config |
| `pretrain_aoe.sh` | **New** — AoE training launch script |


#### Step 2: Clone HaWoR into this recipe folder

```bash
# From this recipe folder
git clone https://github.com/ThunderVVV/HaWoR.git
# Then follow HaWoR's README to install its deps and place the MANO weights.
```

The data-processing step expects HaWoR at `<recipe>/HaWoR/`.

#### Step 3: Build the dataset

Run the pipeline from this recipe folder. Point `HRDT_PROJECT_ROOT` at the cloned
(patched) H-RDT repo — the language-encoding step imports H-RDT's T5 encoder and reads
the AoE config from there. All paths are passed as environment variables:

| Variable | Description |
|---|---|
| `HRDT_PROJECT_ROOT` | Path to the cloned + patched H-RDT repo |
| `AOE_RAW_DATA` | Raw AoE data root (the `AoE 数据集` directory) |
| `AOE_PROCESSED_DATA` | Output directory for processed data |
| `T5_MODEL_PATH` | T5-v1_1-XXL model weights |
| `MIN_SEGMENT_FRAMES` | Minimum frames per segment (shorter segments skipped) |
| `NUM_GPUS` / `PROCESSES_PER_GPU` | Parallelism for T5 encoding |

```bash
HRDT_PROJECT_ROOT=/path/to/H_RDT \
AOE_RAW_DATA=/path/to/AoE 数据集 \
AOE_PROCESSED_DATA=/path/to/processed \
T5_MODEL_PATH=/path/to/t5-v1_1-xxl \
bash run_pretrain_pipeline.sh
```

The three steps are:

1. **Precompute 48D actions** — reads MANO parameters from `hands.npz`, runs forward
   kinematics via HaWoR, converts to the 48D representation, and segments by annotation.
2. **Calculate statistics** — per-dimension min/max across all segments → `aoe_stat.json`.
3. **Encode language embeddings** — runs T5-v1_1-XXL on each segment's description → `.pt`.

#### Step 4: Train

From the cloned H-RDT repo, set `--data_root` to your `AOE_PROCESSED_DATA` and launch:

```bash
cd /path/to/H_RDT
bash pretrain_aoe.sh
```

`pretrain_aoe.sh` launches training via HuggingFace Accelerate + DeepSpeed ZeRO-1. The
dino-siglip weights are resolved via the `DINO_SIGLIP_DIR` env var. To resume, add
`--resume_from_checkpoint="latest"`.

### Technical Notes

#### 48D Action Representation

Each frame is encoded as a 48D bimanual vector. Per hand (24D): root translation (3D),
6D continuous rotation, and 5 fingertip positions (15D, OpenPose ordering: thumb, index,
middle, ring, pinky). MANO forward kinematics is run through HaWoR's `run_mano_twohands`
directly to guarantee numerical consistency with the reconstruction convention.

#### Segmentation

Clips are split into action segments using the boundaries in
`ego_annotation/ego_action_annotation.json`. Segments shorter than `MIN_SEGMENT_FRAMES`
are skipped; each surviving segment becomes one training sample with its own description.

### File Reference

| File | Description |
|------|-------------|
| `precompute_actions.py` | Raw AoE → 48D action HDF5 segments (MANO FK via HaWoR) |
| `calc_stat.py` | Computes min/max normalization statistics (`aoe_stat.json`) |
| `encode_lang_batch.py` | Encodes per-segment text with T5-XXL into `.pt` embeddings |
| `run_pretrain_pipeline.sh` | Runs the three data-processing steps end-to-end |
| `hrdt_aoe_support.patch` | Patch that adds `aoe` support (incl. the dataset loader) to upstream H-RDT |

> The AoE dataset loader (`aoe_dataset.py`), the `aoe` pre-train config, and the training
> launch script are delivered into the H-RDT repo by the patch, so they are not duplicated
> in this recipe folder.

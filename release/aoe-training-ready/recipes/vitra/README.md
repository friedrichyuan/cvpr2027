## VITRA Recipe — Train VITRA with AoE Ego-Centric Data

This recipe converts [AoE](https://github.com/AoE-Ego) ego-centric hand manipulation data into [VITRA](https://github.com/microsoft/VITRA) Human-VLA training format, enabling zero-setup pretraining of VITRA on AoE datasets.

### Overview

**What this recipe does:**
- Converts AoE data (MANO hand reconstructions + action annotations + ego videos) into VITRA's episodic annotation format
- Provides a one-command patch to add `aoe` dataset support to VITRA's data loading pipeline
- Includes a verification script to validate the conversion numerically and visually

**What this recipe does NOT do:**
- Modify VITRA's training code — for training instructions, refer to the [VITRA README](https://github.com/microsoft/VITRA)

### Prerequisites

1. **MANO weights**: Download from [MANO website](https://mano.is.tue.mpg.de/) and place under a `mano/` directory (you need `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`)
2. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **AoE data**: Obtain AoE ego-centric datasets (each clip should follow the standard AoE directory layout with `ego_process/ego_hands_reconstruction/hands.npz`)
4. **VITRA repository**: Clone from https://github.com/microsoft/VITRA

### Quick Start

#### Step 1: Convert AoE data to VITRA format

```bash
python convert_aoe_to_vitra.py \
    --root_dir /path/to/aoe_data \
    --out_root /path/to/vitra_data \
    --mano_dir /path/to/mano_weights \
    --dataset_name aoe
```

**Key arguments:**
- `--root_dir`: Root directory containing AoE clips (recursively scanned for `hands.npz`)
- `--out_root`: Output directory for VITRA-format data
- `--dataset_name`: Dataset identifier (default: `aoe`). This determines the episode_id prefix and must match the VITRA patch routing
- `--max_clips`: Limit number of clips to convert (default: all)
- `--device`: `cpu` or `cuda` for MANO forward kinematics (default: `cpu`)
- `--verify`: Run numerical cam-space self-consistency check per clip

#### Step 2: Verify conversion

```bash
python verify_conversion.py \
    --aoe_root /path/to/aoe_data \
    --out_root /path/to/vitra_data \
    --dataset_name aoe \
    --out_dir output/verify
```

Inspect the output images — green/blue dots should land precisely on the hand wrist/palm positions.

#### Step 3: Apply VITRA patch

```bash
cd /path/to/VITRA
git apply /path/to/vitra_aoe_support.patch
```

This patch adds three changes to VITRA:
- **`vitra/datasets/dataset.py`**: Adds `aoe` dataset path routing (annotation, video, statistics paths)
- **`vitra/datasets/human_dataset.py`**: Adds `aoe` video path resolution
- **`vitra/datasets/data_mixture.py`**: Adds `aoe` and `aoe_magic` data mixture definitions

#### Step 4: Train VITRA

Copy the smoke test config and update paths:

```bash
cp configs/aoe_smoke.json /path/to/VITRA/vitra/configs/
# Edit aoe_smoke.json: replace <YOUR_CONVERTED_DATA_ROOT> and <YOUR_OUTPUT_DIR>
```

Then follow the [VITRA training instructions](https://github.com/microsoft/VITRA#training) with config `aoe_smoke`.

### Technical Notes

#### MANO Convention Differences

The conversion handles two non-obvious conventions that VITRA imposes, both differing from HaWoR's raw reconstruction output:

1. **`transl_worldspace`** is the J0 (wrist joint) position in world space, NOT the MANO root translation. The difference is ~9.6cm (J0 canonical magnitude). We use `joints_worldspace[:, 0]` from MANO FK output.

2. **Left-hand `hand_pose`** is stored in MANO_RIGHT convention — i.e., the axis-angle that, when fed into MANO_RIGHT and then X-mirrored, yields the true left-hand mesh. The mirror transform is `(rx, ry, rz) → (rx, -ry, -rz)`.

#### Episode ID Format

Episode IDs follow the pattern `{dataset_name}_{clip_id}_ep_{idx:06d}`. The first token before `_` (extracted by `episode_id.split('_')[0]`) determines the dataset prefix for VITRA's data routing. With `--dataset_name aoe`, this becomes `aoe`.

#### Verified Results

Smoke test on AoE ego-centric data (100 steps, backbone frozen, action model only):
- **Dataset**: 314,253 frames across 111 clips from poc_deliver
- **Loss**: ~1.4 → ~1.0 (action model DiT converging with backbone frozen)
- **Environment**: NVIDIA GB10, single GPU, OpenCV video backend
- **Note**: The `aoe_smoke.json` config freezes the LLM backbone for the full 100 steps (`llm_freeze_step: 100`). For deeper convergence, increase `max_steps` or reduce `llm_freeze_step`.

### File Reference

| File | Description |
|------|-------------|
| `convert_aoe_to_vitra.py` | Main data conversion script |
| `verify_conversion.py` | Numerical + visual verification |
| `vitra_aoe_support.patch` | VITRA code patch for `aoe` dataset support |
| `configs/aoe_smoke.json` | Training config template |
| `requirements.txt` | Python dependencies |

# DreamZero on Open-AoE

This document describes the Open-AoE integration for a local DreamZero checkout.
Open-AoE-specific source changes are provided as
`patches/open_aoe_support.patch`; the project-local launcher is
`scripts/aoe_train`. This directory is an integration recipe, not a vendored
copy of DreamZero.

## Version

- Upstream project: DreamZero
- Verified commit: `ab790c198fbce33503358efbbd4187ce9a89adf3`
- Patch: `patches/open_aoe_support.patch`

## Environment

Clone DreamZero separately, check out the verified commit, and follow the
upstream installation guide in that checkout:

```bash
conda activate wam_dreamzero
export DREAMZERO_UPSTREAM_ROOT=/PATH_TO/dreamzero
cd "$DREAMZERO_UPSTREAM_ROOT"
git checkout ab790c198fbce33503358efbbd4187ce9a89adf3
pip install -e .
```

Required model assets:

```text
/PATH_TO/ckpts/HF_ckpts/Wan-AI/Wan2.2-TI2V-5B
/PATH_TO/ckpts/HF_ckpts/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
/PATH_TO/ckpts/HF_ckpts/google/umt5-xxl
```

## Apply Patch

From a clean DreamZero checkout:

```bash
cd "$DREAMZERO_UPSTREAM_ROOT"
git checkout ab790c198fbce33503358efbbd4187ce9a89adf3
git apply /PATH_TO/Open-AoE-dev/aoe-training-ready/dreamzero/patches/open_aoe_support.patch
```

The patch adds the `open_aoe` embodiment tag, 110D state/action support,
Open-AoE DreamZero data config, and safe loss/data-loader handling used in the
verified runs.

## Data Conversion

DreamZero uses WAM-v2/GEAR metadata. Convert raw Open-AoE data:

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/general_datasets/Open-AoE/poc_deliver
export OPEN_AOE_WAM_ROOT=/PATH_TO/dataset/Open_AoE/open_aoe_mano_nextstate_224_110d_wam_v2

./scripts/aoe_train convert
./scripts/aoe_train gear
```

Run these commands from
`/PATH_TO/Open-AoE-dev/aoe-training-ready/dreamzero`. The launcher uses
`DREAMZERO_UPSTREAM_ROOT` for DreamZero source files and this directory for
Open-AoE conversion/plotting scripts.

The converter writes 110D `observation.state` and `action`:

```text
left hand  = valid(1) + wrist_trans(3) + wrist_rot_axis_angle(3) + wrist_vel(3) + MANO pose(45)
right hand = valid(1) + wrist_trans(3) + wrist_rot_axis_angle(3) + wrist_vel(3) + MANO pose(45)
```

## Train

Dry-run:

```bash
./scripts/aoe_train --dry-run train
```

Train:

```bash
export CKPT_ROOT=/PATH_TO/ckpts/HF_ckpts
export AOE_RESULTS_DIR=/PATH_TO/aoe_runs/wam_dreamzero_open_aoe
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NGPU=8
export STEPS=100000
export SAVE_STEPS=100000

./scripts/aoe_train train
```

The launcher calls DreamZero with:

```text
data=dreamzero/open_aoe_relative_wan22
max_state_dim=110
max_action_dim=110
num_views=1
open_aoe_data_root=$OPEN_AOE_WAM_ROOT
```

Additional DreamZero/Hydra overrides can be appended after `train`.

## Outputs

```text
$AOE_RESULTS_DIR/logs/train.log
$AOE_RESULTS_DIR/dreamzero_train/
$AOE_RESULTS_DIR/curves/dreamzero/dreamzero_loss_summary.csv
$AOE_RESULTS_DIR/curves/dreamzero/dreamzero_loss_curves.png
```

Set `SAVE_STEPS=$STEPS` to align the checkpoint save step with the final
training step.

# LingBot-VA on Open-AoE

This document describes the Open-AoE integration for a local LingBot-VA checkout.
Open-AoE-specific source changes are provided as
`patches/open_aoe_support.patch`; the project-local launcher is
`scripts/aoe_train`. This directory is an integration recipe, not a vendored
copy of LingBot-VA.

## Version

- Upstream project: LingBot-VA
- Verified commit: `58c2ae5bac46bd8114065bea9d7d256eb67c16c3`
- Patch: `patches/open_aoe_support.patch`

## Environment

Clone LingBot-VA separately, check out the verified commit, and follow the
upstream installation guide in that checkout:

```bash
conda activate wam_lingbot_va
export LINGBOT_UPSTREAM_ROOT=/PATH_TO/lingbot-va
cd "$LINGBOT_UPSTREAM_ROOT"
git checkout 58c2ae5bac46bd8114065bea9d7d256eb67c16c3
pip install -e .
```

Required model assets:

```text
/PATH_TO/ckpts/HF_ckpts/LingBot-VA/lingbot-va-base
```

## Apply Patch

From a clean checkout:

```bash
cd "$LINGBOT_UPSTREAM_ROOT"
git checkout 58c2ae5bac46bd8114065bea9d7d256eb67c16c3
git apply /PATH_TO/Open-AoE-dev/aoe-training-ready/lingbot-va/patches/open_aoe_support.patch
```

The patch adds `open_aoe_train`, 110D action support, data-loader worker
controls, launcher compatibility, and text-mask/import fallbacks.

## Data Conversion

LingBot-VA uses the same WAM-v2 Open-AoE format as DreamZero and additionally
precomputes LingBot latents:

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/general_datasets/Open-AoE/poc_deliver
export OPEN_AOE_WAM_ROOT=/PATH_TO/dataset/Open_AoE/open_aoe_mano_nextstate_224_110d_wam_v2
export CKPT_ROOT=/PATH_TO/ckpts/HF_ckpts

./scripts/aoe_train convert
./scripts/aoe_train prepare-latents
```

Run these commands from
`/PATH_TO/Open-AoE-dev/aoe-training-ready/lingbot-va`. The launcher uses
`LINGBOT_UPSTREAM_ROOT` for LingBot-VA source files and this directory for
Open-AoE conversion/plotting scripts.

The WAM-v2 converter writes the 110D current hand state and next-frame hand
target using full MANO pose and wrist velocity.

## Train

Dry-run:

```bash
./scripts/aoe_train --dry-run train
```

Train:

```bash
export AOE_RESULTS_DIR=/PATH_TO/aoe_runs/wam_lingbot_open_aoe
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NGPU=8
export STEPS=5000
export SAVE_INTERVAL=5000

./scripts/aoe_train train
```

Useful memory/runtime overrides:

```bash
export LINGBOT_BATCH=1
export LINGBOT_GRAD_ACCUM=1
export LINGBOT_NUM_INIT_WORKER=1
export LINGBOT_LOAD_WORKER=0
```

## Outputs

```text
$AOE_RESULTS_DIR/logs/train.log
$AOE_RESULTS_DIR/lingbot_train/
$AOE_RESULTS_DIR/curves/lingbot/lingbot_loss_summary.csv
$AOE_RESULTS_DIR/curves/lingbot/lingbot_loss_curves.png
```

Set `SAVE_INTERVAL=$STEPS` to save only the final checkpoint.

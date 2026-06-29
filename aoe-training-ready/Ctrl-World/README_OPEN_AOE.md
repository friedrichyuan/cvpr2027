# Ctrl-World on Open-AoE

This document describes the Open-AoE integration for a local Ctrl-World checkout.
No upstream source patch is required; Open-AoE support is provided by
project-local scripts under `scripts/`. This directory is an integration
recipe, not a vendored copy of Ctrl-World.

## Version

- Upstream project: Ctrl-World
- Verified commit: `99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d`
- Patch: none

## Environment

Clone Ctrl-World separately, check out the verified commit, and use the upstream
Ctrl-World environment:

```bash
conda activate ctrl-world
export CTRLWORLD_UPSTREAM_ROOT=/PATH_TO/Ctrl-World
cd "$CTRLWORLD_UPSTREAM_ROOT"
git checkout 99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d
pip install -e .
```

Required pretrained assets:

```text
/PATH_TO/ckpts/HF_ckpts/Ctrl-World/stable-video-diffusion-img2vid
/PATH_TO/ckpts/HF_ckpts/Ctrl-World/clip-vit-base-patch32
/PATH_TO/ckpts/HF_ckpts/Ctrl-World/Ctrl-World/checkpoint-10000.pt
```

## Data Conversion

The converter builds a Ctrl-World latent dataset from raw Open-AoE samples:

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/general_datasets/Open-AoE/poc_deliver
export OPEN_AOE_CTRLWORLD_ROOT=/PATH_TO/dataset/Open_AoE/open_aoe_mano_nextstate_224_110d_ctrlworld_full
export CKPT_ROOT=/PATH_TO/ckpts/HF_ckpts/Ctrl-World

./scripts/aoe_train convert
```

Run these commands from
`/PATH_TO/Open-AoE-dev/aoe-training-ready/Ctrl-World`. The launcher uses
`CTRLWORLD_UPSTREAM_ROOT` for Ctrl-World source files and this directory for
Open-AoE conversion/plotting scripts.

The action/state vector is 110D:

```text
left hand  = valid(1) + wrist_trans(3) + wrist_rot_axis_angle(3) + wrist_vel(3) + MANO pose(45)
right hand = valid(1) + wrist_trans(3) + wrist_rot_axis_angle(3) + wrist_vel(3) + MANO pose(45)
```

Ctrl-World expects three view slots. The converter copies the Open-AoE
egocentric RGB stream into the three slots so the original model interface
remains unchanged.

## Train

Dry-run:

```bash
./scripts/aoe_train --dry-run train
```

Train:

```bash
export AOE_RESULTS_DIR=/PATH_TO/aoe_runs/wm_ctrl_world_open_aoe
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export STEPS=1000000
export CHECKPOINTING_STEPS=0

./scripts/aoe_train train
```

Resume from a final checkpoint:

```bash
export CTRLWORLD_CKPT_PATH=/PATH_TO/previous/final_model.pt
export AOE_RESULTS_DIR=/PATH_TO/aoe_runs/wm_ctrl_world_open_aoe_resume
./scripts/aoe_train train
```

The final rollout video layout is top row ground truth and bottom row
prediction.

## Outputs

```text
$AOE_RESULTS_DIR/logs/train.log
$AOE_RESULTS_DIR/loss_curves/ctrl_world_loss_summary.csv
$AOE_RESULTS_DIR/loss_curves/ctrl_world_loss_curves.png
$AOE_RESULTS_DIR/train/final_model.pt
$AOE_RESULTS_DIR/train/visualizations/training_sample_rgb_multicam.mp4
$AOE_RESULTS_DIR/train/visualizations/final_rollout_rgb_step_*.mp4
```

Intermediate checkpoints are disabled with `CHECKPOINTING_STEPS=0`; the script
still saves `final_model.pt`.

# LeRobot ACT/DP/pi0.5 on Open-AoE

This document describes Open-AoE training for ACT, Diffusion Policy, and pi0.5
through a local LeRobot checkout. No upstream source patch is required; the
Open-AoE converter and launcher live in this recipe under `scripts/`. This
directory is not a vendored copy of LeRobot.

## Version

- Upstream project: LeRobot
- Local verified checkout for this directory: `6a788fbdb02cabfae60f7408636945df0b1eafa0`
- Server training checkout used in earlier smoke runs: `3d63b871c09cdf37aeb0e137a2aae3e4e5c407a7`
- Patch: none for the current 110D Open-AoE dataset path

## Environment

Clone LeRobot separately, check out the verified commit, and follow the upstream
installation guide in that checkout:

```bash
conda activate lerobot2
export LEROBOT_UPSTREAM_ROOT=/PATH_TO/lerobot
cd "$LEROBOT_UPSTREAM_ROOT"
git checkout 6a788fbdb02cabfae60f7408636945df0b1eafa0
pip install -e .
```

Optional offline assets:

```text
/PATH_TO/ckpts/checkpoints/resnet
/PATH_TO/ckpts/checkpoints/pi05_base
```

## Data Conversion

The converter creates a LeRobot dataset with:

```text
observation.images.front
observation.state  # 110D current hand state
action             # 110D next hand state target
task
```

Run conversion:

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/general_datasets/Open-AoE/poc_deliver
export OPEN_AOE_LEROBOT_ROOT=/PATH_TO/dataset/Open_AoE/open_aoe_mano_nextstate_224_110d
export OPEN_AOE_REPO_ID=local/open-aoe-mano-nextstate-224-110d

./scripts/aoe_train convert
```

Run these commands from
`/PATH_TO/Open-AoE-dev/aoe-training-ready/lerobot`. The launcher uses
`LEROBOT_UPSTREAM_ROOT` for LeRobot source files and this directory for
Open-AoE conversion/plotting scripts.

The 110D state/action vector is:

```text
left hand  = valid(1) + wrist_trans(3) + wrist_rot_axis_angle(3) + wrist_vel(3) + MANO pose(45)
right hand = valid(1) + wrist_trans(3) + wrist_rot_axis_angle(3) + wrist_vel(3) + MANO pose(45)
```

## Train

Dry-run:

```bash
./scripts/aoe_train --dry-run --model act train
./scripts/aoe_train --dry-run --model dp train
./scripts/aoe_train --dry-run --model pi05 train
```

Train ACT:

```bash
export AOE_RESULTS_ROOT=/PATH_TO/aoe_runs
export CUDA_VISIBLE_DEVICES=0
export MODEL=act
export STEPS=5000
export SAVE_FREQ=9999999
./scripts/aoe_train train
```

Train Diffusion Policy:

```bash
export CUDA_VISIBLE_DEVICES=1
export MODEL=dp
export STEPS=5000
export SAVE_FREQ=9999999
./scripts/aoe_train train
```

Train pi0.5:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL=pi05
export STEPS=2200
export SAVE_FREQ=2200
export PI05_PRETRAINED_PATH=/PATH_TO/ckpts/checkpoints/pi05_base
./scripts/aoe_train train
```

Extra LeRobot config overrides can be appended after `train`.

## Outputs

```text
$AOE_RESULTS_ROOT/vla_<model>_open_aoe/logs/<model>.log
$AOE_RESULTS_ROOT/vla_<model>_open_aoe/loss_curves/<model>_loss_summary.csv
$AOE_RESULTS_ROOT/vla_<model>_open_aoe/loss_curves/<model>_loss_curves.png
$AOE_RESULTS_ROOT/vla_<model>_open_aoe/<model>_open_aoe/
```

Use a very large `SAVE_FREQ` for ACT/DP to avoid intermediate checkpoints. Use
`SAVE_FREQ=$STEPS` for pi0.5 to save only the final checkpoint.

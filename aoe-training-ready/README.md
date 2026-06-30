# AoE Training-Ready

This directory keeps each baseline as an independent Open-AoE integration
recipe. It intentionally does not vendor full upstream projects. Every recipe
owns its Open-AoE conversion script, launch script, loss-curve script, README,
and upstream patch when source changes are required.

## Data Representation

The verified Open-AoE action/state vector is 110D:

```text
left hand  = valid(1) + wrist_trans(3) + wrist_rot_axis_angle(3) + wrist_vel(3) + MANO pose(45)
right hand = valid(1) + wrist_trans(3) + wrist_rot_axis_angle(3) + wrist_vel(3) + MANO pose(45)
```

`state` uses the current frame hand vector. `action` uses the next-frame target
hand vector at the method-specific action stride.

> The world-model recipes (iVideoGPT, GenieRedux, laom, AdaWorld, DreamDojo) and SmolVLA
> use a different hand-as-EEF + camera action representation (22D state / 20–26D action),
> documented in [`ACTION_SPEC.md`](ACTION_SPEC.md).

## Recipes

| Model | Project | Open-AoE entry | Patch |
|---|---|---|---|
| DreamZero | [`dreamzero/`](dreamzero/) | [`dreamzero/README_OPEN_AOE.md`](dreamzero/README_OPEN_AOE.md) | [`dreamzero/patches/open_aoe_support.patch`](dreamzero/patches/open_aoe_support.patch) |
| LingBot-VA | [`lingbot-va/`](lingbot-va/) | [`lingbot-va/README_OPEN_AOE.md`](lingbot-va/README_OPEN_AOE.md) | [`lingbot-va/patches/open_aoe_support.patch`](lingbot-va/patches/open_aoe_support.patch) |
| Ctrl-World | [`Ctrl-World/`](Ctrl-World/) | [`Ctrl-World/README_OPEN_AOE.md`](Ctrl-World/README_OPEN_AOE.md) | none |
| ACT / DP / pi0.5 | [`lerobot/`](lerobot/) | [`lerobot/README_OPEN_AOE.md`](lerobot/README_OPEN_AOE.md) | none |
| VITRA | [`vitra/`](vitra/) | [`vitra/README.md`](vitra/README.md) | [`vitra/vitra_aoe_support.patch`](vitra/vitra_aoe_support.patch) |
| GR00T N1.7 | [`gr00t_n1d7/`](gr00t_n1d7/) | [`gr00t_n1d7/README.md`](gr00t_n1d7/README.md) | project-local changes |
| H-RDT | [`H-RDT/`](H-RDT/) | [`H-RDT/README.md`](H-RDT/README.md) | [`H-RDT/hrdt_aoe_support.patch`](H-RDT/hrdt_aoe_support.patch) |
| SmolVLA | [`smolvla/`](smolvla/) | [`smolvla/README_OPEN_AOE.md`](smolvla/README_OPEN_AOE.md) | none |
| iVideoGPT | [`ivideogpt/`](ivideogpt/) | [`ivideogpt/README_OPEN_AOE.md`](ivideogpt/README_OPEN_AOE.md) | [`ivideogpt/patches/open_aoe_support.patch`](ivideogpt/patches/open_aoe_support.patch) |
| GenieRedux | [`genie-redux/`](genie-redux/) | [`genie-redux/README_OPEN_AOE.md`](genie-redux/README_OPEN_AOE.md) | none |
| laom (LAOM) | [`laom/`](laom/) | [`laom/README_OPEN_AOE.md`](laom/README_OPEN_AOE.md) | none |
| AdaWorld | [`adaworld/`](adaworld/) | [`adaworld/README_OPEN_AOE.md`](adaworld/README_OPEN_AOE.md) | none |
| DreamDojo | [`dreamdojo/`](dreamdojo/) | [`dreamdojo/README_OPEN_AOE.md`](dreamdojo/README_OPEN_AOE.md) | [`dreamdojo/patches/open_aoe_support.patch`](dreamdojo/patches/open_aoe_support.patch) |

## Quick Dry Run

Each recipe exposes its own launcher. Point the launcher at a separate upstream
checkout with the method-specific `*_UPSTREAM_ROOT` variable listed in that
method's README.

```bash
cd dreamzero
./scripts/aoe_train --dry-run train

cd ../lingbot-va
./scripts/aoe_train --dry-run train

cd ../Ctrl-World
./scripts/aoe_train --dry-run train

cd ../lerobot
./scripts/aoe_train --dry-run --model act train
./scripts/aoe_train --dry-run --model dp train
./scripts/aoe_train --dry-run --model pi05 train
```

World-model recipes (and SmolVLA) expose the same dry-run interface:

```bash
cd ../smolvla      && ./scripts/aoe_train --dry-run train
cd ../ivideogpt    && ./scripts/aoe_train --dry-run train
cd ../genie-redux  && ./scripts/aoe_train --dry-run full
cd ../laom         && ./scripts/aoe_train --dry-run train
cd ../adaworld     && ./scripts/aoe_train --dry-run train
cd ../dreamdojo    && ./scripts/aoe_train --dry-run convert
```

Use environment variables to point scripts at local data, checkpoints, and
outputs. The common variables are:

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/general_datasets/Open-AoE/poc_deliver
export CKPT_ROOT=/PATH_TO/ckpts/HF_ckpts
export AOE_RESULTS_DIR=/PATH_TO/aoe_runs/<experiment>
```

Model-specific README files list the full environment setup, data conversion,
training commands, outputs, and verified upstream commit. For methods with a
patch, clone the upstream project separately, check out the verified commit, and
apply the patch from this repository before running the launcher.

## Contributing a New Recipe

For a new method, add the integration inside that method's own project folder:

1. `README_OPEN_AOE.md` with environment setup, data conversion, training, and outputs.
2. `scripts/aoe_train` or an equivalent project-local launch script.
3. Project-local converter and plotting scripts when needed.
4. `patches/open_aoe_support.patch` when upstream source changes are required.

Avoid hard-copying upstream projects into this repository unless the license has
been reviewed. Prefer documenting the upstream commit and providing a patch.

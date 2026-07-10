# AoE Retarget Lab

Language: **English** | [中文](docs/README.zh-CN.md)

AoE Retarget Lab connects Open-AoE egocentric RGB clips to two object 6DoF
reconstruction pipelines and three robot retargeting/rendering backends. The
main review artifact is a synchronized video stack:

```text
raw RGB + reconstructed mesh overlay
depth / input geometry
robot retargeting render
```

All newly generated assets should be written under `experiments/<run_name>/`.
Do not write generated meshes, trajectories, videos, logs, or manifests back
into an Open-AoE dataset folder or a third-party source tree.

## Documentation

| Document | English | Chinese |
| --- | --- | --- |
| README | this file | [docs/README.zh-CN.md](docs/README.zh-CN.md) |
| Installation | [docs/INSTALL.md](docs/INSTALL.md) | [docs/INSTALL.zh-CN.md](docs/INSTALL.zh-CN.md) |
| Script API | [docs/API.md](docs/API.md) | [docs/API.zh-CN.md](docs/API.zh-CN.md) |
| Principles | [docs/PRINCIPLES.md](docs/PRINCIPLES.md) | [docs/PRINCIPLES.zh-CN.md](docs/PRINCIPLES.zh-CN.md) |

Start with [Installation](docs/INSTALL.md), then use the command surface in
[Script API](docs/API.md). The design and debugging rules are summarized in
[Principles](docs/PRINCIPLES.md).

## Repository Layout

```text
scripts/                     runnable shell and Python entrypoints
scripts/setup_third_party.sh local fetch script for upstream packages
src/aoe_retarget_lab/        AoE-specific adapters and utilities
configs/                     sample configs
experiments/                 local experiment outputs
docs/                        bilingual documentation
third_party/                 local upstream clones, gitignored and created by setup
```

This public import does not vendor EgoInfinity, Do-as-I-Do, SPIDER, robot
assets, model weights, datasets, or generated videos. Run
`scripts/setup_third_party.sh` after cloning to fetch upstream packages into the
gitignored `third_party/` directory.

## Pipeline Matrix

The project is organized around three axes:

```text
trajectory_6dof = egoinfinity | do_as_i_do
hand_source     = aoe | estimated
retargeting     = egoinfinity | do_as_i_do | spider
```

The full comparison matrix contains:

```text
2 trajectory_6dof x 2 hand_source x 3 retargeting = 12 cells
```

Stable cell keys use:

```text
traj_<trajectory_6dof>__hand_<hand_source>__retarget_<retargeting>
```

## Quick Start

Install the environments, fetch third-party packages, and configure external
asset roots first:

```bash
bash scripts/setup_third_party.sh
source local_env.sh
```

Run the recommended full-12 entrypoint. It runs one full EgoInfinity pass and
one full Do-as-I-Do pass, then reuses those intermediate assets to compose the
12-cell comparison matrix.

```bash
scripts/run_full_12_demos.sh \
  --run-name <source_run_name> \
  --matrix-run-name <matrix_run_name> \
  --ego-video /path/to/raw_video_undistorted.mp4 \
  --ego-clip-id <clip_id> \
  --ego-objects "bottle of vinegar" \
  --ego-start 0.0 \
  --ego-end 3.0 \
  --dai-raw-dir /path/to/do_as_i_do_raw_dir \
  --dai-clip-dir /path/to/do_as_i_do_clip_dir \
  --task <task_name> \
  --hand-type bimanual \
  --main-cuda 0 \
  --sam3-cuda 1 \
  --sam3d-cuda 2 \
  --spider-cuda-visible-devices 3
```

Expected outputs:

```text
experiments/<source_run_name>/videos/v4_egoinfinity_full__triptych.mp4
experiments/<source_run_name>/videos/v4_do_as_i_do_full__triptych.mp4
experiments/<source_run_name>/reuse/reuse_manifest.json
experiments/<matrix_run_name>/videos/<cell>__triptych.mp4
experiments/<matrix_run_name>/reuse_12_demo_manifest.json
experiments/<matrix_run_name>/cells/
experiments/<matrix_run_name>/assets/
```

## Demo Acceptance Rules

- The overlay row must show raw RGB with reconstructed hand mesh and object
  mesh or point cloud, not just a segmentation mask, skeleton, or fingertips.
- The robot row must render the robot body and dexterous hand: EgoInfinity/G1,
  Do-as-I-Do/Sharpa, or SPIDER/XHand.
- Retargeted digital assets such as trajectories, meshes, `npz`, `scene.xml`,
  manifests, logs, and videos must be grouped under `experiments/<run_name>/`.
- Do-as-I-Do retargeting should use the official
  `third_party/do-as-i-do/retargeting/launch.py` entrypoint.
- Do not fix scale or drift by hard-coded visual shrinking. Diagnose mask,
  mesh reconstruction, 6DoF/layout scale, and retarget adapter transforms.

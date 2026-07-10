# Open-AoE

<p align="center">
  <strong>Open-Source Egocentric Data and Toolchain for Embodied Intelligence</strong>
</p>

<p align="center">
  English | <a href="README_zh.md">简体中文</a>
</p>

> [!IMPORTANT]
> Open-AoE-2000H and the accompanying technical report are being prepared for public release. This repository contains the open-source toolchain and data documentation; dataset download and citation links will be added when the release is finalized.

<p align="center">
  <img src="docs/fig1-open-aoe-overview.png" width="100%" alt="Open-AoE dataset, processing pipeline, and open-source toolchain overview">
</p>

## Overview

Open-AoE is a community-oriented release of large-scale, real-world egocentric manipulation data collected with consumer smartphones. The planned release contains roughly **2,000 hours** of first-person human manipulation video together with synchronized hand motion, camera motion, and atomic action annotations.

Open-AoE goes beyond publishing video and labels. It provides a reproducible data-to-model path that turns the same synchronized segment into reviewable visualizations, reusable hand-object assets, robot-facing motion, robotized video, and model-specific training interfaces.

| Scale | Contributors | Device types | Scenes | Tasks |
|---:|---:|---:|---:|---:|
| ~2,000 hours | 768 | 200 | 500 | 10,000+ |

### Why Open-AoE?

- **Smartphone-first collection.** Consumer devices make real-world egocentric capture easier to scale than specialized headsets or robot-only collection.
- **Manipulation-aligned signals.** RGB video is synchronized with camera calibration and trajectory, MANO hand reconstruction, validity masks, and atomic action annotations.
- **A complete toolchain.** The repository connects data inspection, 4D reconstruction, human-to-robot retargeting, robot overlay, action conversion, and downstream training recipes.
- **Modular by design.** Each component can be used independently; downstream methods consume the representation that matches their task instead of one rigid universal format.
- **Community extensibility.** New robot embodiments, reconstruction backends, and training recipes can be contributed as self-contained subprojects.

## Open-AoE-2000H Dataset

Each released segment is designed as a synchronized multimodal record of an egocentric manipulation episode.

| Signal | Main artifact | What it provides |
|---|---|---|
| Raw and undistorted RGB | `raw_video.mp4`, `raw_video_undistorted.mp4` | First-person visual observation and calibrated video |
| Camera metadata | `video_info.json`, `undistorted_video_info.json` | Device information, intrinsics, distortion, resolution, and frame rate |
| Camera motion | `camera_traj.npz` and transforms in `hands.npz` | Metric-scale 6-DoF camera trajectory and world/camera transforms |
| Hand reconstruction | `hands.npz` | Per-frame MANO pose, shape, root transform, and validity for both hands |
| Atomic actions | `ego_action_annotation.json` | Temporally aligned action segments, verbs, objects, hands, and descriptions |

The complete sample layout, field definitions, coordinate conventions, and validation notes are documented in the [Open-AoE-2000H data specification](open-aoe-2000h/README.md).

### Data processing and quality control

<p align="center">
  <img src="docs/fig2-data-pipline.png" width="100%" alt="Open-AoE online capture, offline processing, reconstruction, annotation, and quality-control pipeline">
</p>

The release pipeline has four stages:

1. **On-device capture control** checks hand visibility, wearing conditions, lighting, motion quality, and device health before upload.
2. **Offline quality control and scene labeling** filters invalid or sensitive content, standardizes frame rate, slices videos, and assigns scene/task metadata.
3. **Reconstruction and annotation** estimates camera trajectories, reconstructs MANO hands, and produces atomic action segments.
4. **Quality inspection and delivery** applies completeness, correctness, and temporal-consistency gates followed by human review.

## Open-AoE Toolchain

The toolchain projects a common Open-AoE segment into the representation spaces required by different robot-learning workflows.

| Component | Purpose | Entry point |
|---|---|---|
| **AoE-Visualization** | Inspect RGB, MANO meshes/keypoints, camera motion, action annotations, and timelines in one review video | [`aoe-visualization/`](aoe-visualization/) |
| **AoE-Retarget-Replay** | Reconstruct hand-object assets, retarget human motion to robot embodiments, validate/render trajectories, and synthesize robotized video | [`aoe-retarget-replay/`](aoe-retarget-replay/) |
| **AoE-Training-Ready** | Convert synchronized AoE signals into model-specific state/action semantics and reproducible training recipes | [`aoe-training-ready/`](aoe-training-ready/) |

### AoE-Visualization

AoE-Visualization produces one end-to-end review video per sample. It overlays MANO hand meshes, 21-keypoint skeletons, future wrist trajectories, atomic-action information, a world-frame 3D view, and a timeline on the undistorted video.

```bash
cd aoe-visualization
pip install -r requirements.txt

# One sample
python visualize.py --sample /path/to/open_aoe_sample

# A directory of samples
python visualize.py --data_dir /path/to/open_aoe_data --output_dir ./output
```

See the [AoE-Visualization guide](aoe-visualization/README.md) for rendering requirements and output details.

### Reconstruction and Retargeting

<p align="center">
  <img src="docs/fig3-reconstruct-retarget.png" width="100%" alt="Open-AoE 4D reconstruction, motion retargeting, and robot-overlay routes">
</p>

The reconstruction and retargeting stack exposes three complementary outputs: a **4D hand-object representation**, **robot-usable motion**, and a **robotized video** in which rendered robot motion is composited into the original scene.

| Subproject | Robot/method coverage | Main capabilities |
|---|---|---|
| [**Phantom**](aoe-retarget-replay/phantom/) | Unitree G1 + Dex3 / Inspire | Arm IK, dexterous-hand retargeting, MuJoCo visualization, and SAM2 + E2FGVI robot overlay |
| [**Retarget Galbot**](aoe-retarget-replay/retarget_galbot/) | Galbot / Galaxea bimanual platform | Palm-to-TCP Pinocchio IK, parallel-jaw gripper mapping, MuJoCo egoview synthesis, and LeRobot/Rerun export |
| [**AoE Retarget Lab**](aoe-retarget-replay/retarget-lab/) | EgoInfinity/G1, Do-as-I-Do/Sharpa, SPIDER/XHand | External-method integration, 6-DoF reconstruction adapters, and a 12-cell comparison matrix |

Retarget Lab does not vendor third-party repositories, model weights, robot assets, or generated videos. Follow its installation guide to fetch and configure the required upstream projects locally.

### AoE-Training-Ready

<p align="center">
  <img src="docs/fig4-training-ready.png" width="100%" alt="Open-AoE training-ready annotation spectrum for VLA, world-action, and world models">
</p>

Training-Ready treats conversion as **action-semantics adaptation**, not simple file-format translation. The same segment can be represented as dense MANO state/action, robot-facing hand or gripper actions, hand-plus-camera dynamics, or latent/weak actions.

| Model family | Included integration recipes |
|---|---|
| VLA policies | VITRA, GR00T N1.7, H-RDT, ACT, Diffusion Policy, π0.5, SmolVLA |
| World/video action models | DreamZero, LingBot-VA, Ctrl-World, iVideoGPT |
| Latent-action and world models | GenieRedux, LAOM, AdaWorld, DreamDojo |

Every recipe owns its conversion or launch scripts, documentation, and upstream patch when source changes are required. Start from the [Training-Ready recipe index](aoe-training-ready/README.md) and the shared [action specification](aoe-training-ready/ACTION_SPEC.md).

## Getting Started

### 1. Clone the toolchain

```bash
git clone https://github.com/woxue/Open-AoE-dev.git
cd Open-AoE-dev
```

### 2. Prepare MANO models when required

MANO model files are not distributed in this repository. Register and download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from the [MANO website](https://mano.is.tue.mpg.de/), then install them into the shared location:

```bash
bash assets/mano/download_mano.sh \
  ~/Downloads/MANO_RIGHT.pkl \
  ~/Downloads/MANO_LEFT.pkl
```

### 3. Choose a workflow

| Goal | Start here |
|---|---|
| Understand the released sample format | [`open-aoe-2000h/README.md`](open-aoe-2000h/README.md) |
| Render and inspect samples | [`aoe-visualization/README.md`](aoe-visualization/README.md) |
| Retarget motion or render robot overlays | [`aoe-retarget-replay/README.md`](aoe-retarget-replay/README.md) |
| Convert data for model training | [`aoe-training-ready/README.md`](aoe-training-ready/README.md) |

## Repository Layout

```text
Open-AoE-dev/
├── docs/                    # Technical-report overview figures
├── open-aoe-2000h/         # Dataset format and usage documentation
├── aoe-visualization/      # Synchronized data review and rendering
├── aoe-retarget-replay/    # Reconstruction, retargeting, replay, and overlays
│   ├── phantom/
│   ├── retarget_galbot/
│   └── retarget-lab/
├── aoe-training-ready/     # Model adapters, converters, launchers, and patches
├── assets/mano/            # Shared MANO setup helper; model files are not tracked
├── CONTRIBUTING.md
├── LEGAL.md
├── LICENSE
└── README_old.md           # Archived collaboration/progress-oriented README
```

## Release Resources

- **Open-AoE-2000H dataset:** coming soon
- **Technical report:** coming soon
- **AoE capture application:** release information coming soon
- **Detailed data specification:** [open-aoe-2000h/README.md](open-aoe-2000h/README.md)

## Contributing

We welcome contributions that add robot embodiments, reconstruction or retargeting backends, visualization features, data converters, and training recipes. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Keep each integration self-contained, document external dependencies, and do not commit datasets, model weights, MANO files, or license-incompatible upstream source.

## License and Third-Party Components

Original source code in this repository is released under the [Apache License 2.0](LICENSE). Dataset distribution terms, model weights, robot assets, and third-party components may use different licenses. See [LEGAL.md](LEGAL.md) and each subproject README before redistribution or commercial use.

## Citation

The official BibTeX entry will be added when the Open-AoE technical report is released.

## Acknowledgements

Open-AoE is built by the AoE community and integrates ideas and interfaces from many open-source projects. We thank all data contributors, toolchain contributors, maintainers, and upstream research teams. Component provenance and license notices are maintained in [LEGAL.md](LEGAL.md).

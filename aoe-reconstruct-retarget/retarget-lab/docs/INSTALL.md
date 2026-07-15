# Installation Guide

Language: **English** | [中文](INSTALL.zh-CN.md)

Navigation: [README](../README.md) | [API](API.md) | [Principles](PRINCIPLES.md)

This guide is for public users who want to clone AoE Retarget Lab, connect
their own Open-AoE data, install third-party packages, and run the demo scripts.

This repository intentionally does not vendor `third_party/EgoInfinity`,
`third_party/do-as-i-do`, `third_party/SPIDER`, robot meshes/assets, model
weights, datasets, Hugging Face caches, MANO assets, conda environments, or
generated experiment outputs. The third-party source trees are fetched locally
with `scripts/setup_third_party.sh` and are ignored by Git.

## 1. System Requirements

Recommended:

```text
Linux x86_64
NVIDIA GPU with CUDA 12.x
conda or mamba
git
ffmpeg
gcc/g++, cmake, ninja, pkg-config
EGL-capable NVIDIA driver for headless MuJoCo rendering
```

Use separate environments because CUDA, MuJoCo, Warp, JAX, PyTorch, and Python
pins differ across upstream packages.

| Environment | Python | Used for |
| --- | --- | --- |
| `aoe-base` | 3.10+ | lightweight repo tools |
| `egoinfinity` | 3.10 | EgoInfinity and G1 retargeting |
| `sam3` | upstream-defined | SAM3/SAM3.1 segmentation worker |
| `sam3d` | upstream-defined | SAM3D Objects, Fast-SAM3D, MoGe |
| `hawor` | upstream-defined | HaWoR and projection diagnostics |
| `dai-retarget` | 3.12 | Do-as-I-Do Sharpa / MuJoCo-Warp |
| `spider` | 3.12 | SPIDER / MJWP / XHand |

Install only the environments needed for the pipelines you plan to run.

## 2. Clone This Repo

```bash
git clone <AOE_RETARGET_LAB_GIT_URL> aoe-retarget-lab
cd aoe-retarget-lab

conda create -y -n aoe-base python=3.10 pip
conda activate aoe-base
pip install -e .

export AOE_RETARGET_LAB_ROOT="$PWD"
```

## 3. Fetch Third-Party Source Trees

Run the setup script once after cloning:

```bash
bash scripts/setup_third_party.sh
```

The script creates:

```text
third_party/EgoInfinity/
third_party/do-as-i-do/
third_party/SPIDER/
third_party/THIRD_PARTY_LOCK.tsv
```

These paths are machine-local and should not be committed. By default the
script uses the official upstream repositories:

```text
EgoInfinity  -> https://github.com/Rice-RobotPI-Lab/EgoInfinity.git
Do-as-I-Do  -> https://github.com/malik-group/do-as-i-do.git
SPIDER      -> https://github.com/facebookresearch/spider.git
```

For reproducible releases, pin the refs explicitly:

```bash
EGOINFINITY_REF=<commit-or-tag> \
DO_AS_I_DO_REF=<commit-or-tag> \
SPIDER_REF=<commit-or-tag> \
bash scripts/setup_third_party.sh
```

`THIRD_PARTY_LOCK.tsv` records the resolved commits in your local checkout.

## 4. Configure Local Paths

Create a machine-local `local_env.sh`; do not commit it.

```bash
export AOE_RETARGET_LAB_ROOT=/path/to/aoe-retarget-lab
export AOE_DATA_ROOT=/path/to/open-aoe
export AOE_MODEL_ROOT=/path/to/aoe-retarget-models

export HF_HOME=/path/to/huggingface_cache
export TORCH_HOME=/path/to/torch_cache
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
```

After caches are populated, offline runs can use:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 5. External Assets

Do-as-I-Do and EgoInfinity use different SAM versions:

```text
Do-as-I-Do reconstruction Stage 1 -> facebook/sam3, sam3.pt
EgoInfinity object segmentation     -> facebook/sam3.1, sam3.1_multiplex.pt
```

Do not substitute SAM3.1 for Do-as-I-Do Stage 1. Check local caches with:

```bash
find -L "$HF_HOME/hub/models--facebook--sam3" -name sam3.pt -print
find -L "$HF_HOME/hub/models--facebook--sam3.1" -name sam3.1_multiplex.pt -print
```

Other required assets may include:

```text
EgoInfinity retarget checkpoints, e.g. retarget/ckpts/g1.pt
MANO_LEFT.pkl / MANO_RIGHT.pkl
MANO_LEFT.npz / MANO_RIGHT.npz
MoGe weights
Fast-SAM3D / SAM3D Objects configs and checkpoints
TAPIR / BootsTAPIR checkpoint
HaWoR and Metric3D weights
```

Keep all of these outside the Git repo.

## 6. EgoInfinity Environment

```bash
conda env create -f third_party/EgoInfinity/retarget/environment.yml
conda activate egoinfinity
pip install -e "$AOE_RETARGET_LAB_ROOT"

export EGOINFINITY_PYTHON=$(which python)
```

Verify:

```bash
python - <<'PY'
import mujoco, torch, trimesh
print("mujoco", mujoco.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("trimesh", trimesh.__version__)
PY
```

## 7. SAM3 and SAM3D Workers

Install SAM3 and SAM3D Objects according to their upstream instructions, then
export the Python binaries and source/model roots:

```bash
export SAM3_PYTHON=/path/to/sam3_env/bin/python
export SAM3_REPO=/path/to/sam3_source

export SAM3D_PYTHON=/path/to/sam3d_env/bin/python
export SAM3D_REPO=/path/to/sam-3d-objects
export SAM3D_REPO_ROOT=/path/to/fastsam3d_or_sam3d_config_root

export DINOV2_LOCAL_REPO=/path/to/facebookresearch_dinov2_main
export DINOV2_REPO_DIR=$DINOV2_LOCAL_REPO
```

Verify:

```bash
$SAM3_PYTHON -c "import torch; print('sam3 cuda', torch.cuda.is_available())"
$SAM3D_PYTHON -c "import torch; print('sam3d cuda', torch.cuda.is_available())"
```

If a run is interrupted, a stale SAM3D worker may keep GPU memory. Check before
starting a long run:

```bash
pgrep -af "egoinfinity_sam3d|sam3d_worker.py|sam3_worker.py"
nvidia-smi
```

Only kill a process after confirming it is an orphan from an old run.

## 8. HaWoR / Projection Diagnostics

Follow the upstream HaWoR and Do-as-I-Do reconstruction setup, then set:

```bash
export HAWOR_PYTHON=/path/to/hawor_env/bin/python
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Verify:

```bash
$HAWOR_PYTHON -c "import torch; print('hawor cuda', torch.cuda.is_available())"
```

Use an environment with `pytorch3d` for Do-as-I-Do mesh projection diagnostics;
the Do-as-I-Do retargeting environment does not necessarily provide it.

## 9. Do-as-I-Do Retargeting / Sharpa

```bash
conda create -y -n dai-retarget python=3.12 pip
conda activate dai-retarget

cd "$AOE_RETARGET_LAB_ROOT/third_party/do-as-i-do/retargeting"
pip install -e .
cd "$AOE_RETARGET_LAB_ROOT"

export RETARGETING_PYTHON=$(which python)
```

This package pins `mujoco==3.4.0`, `warp-lang==1.10.1`, and a pinned
`mujoco-warp` revision. Verify:

```bash
python - <<'PY'
import mujoco, torch, warp
print("mujoco", mujoco.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp", warp.__version__)
PY
```

The wrapper uses the official entrypoint:

```text
third_party/do-as-i-do/retargeting/launch.py
```

## 10. SPIDER / MJWP

```bash
conda create -y -n spider python=3.12 pip
conda activate spider

cd "$AOE_RETARGET_LAB_ROOT/third_party/SPIDER"
pip install -e .
cd "$AOE_RETARGET_LAB_ROOT"

export SPIDER_PYTHON=$(which python)
```

SPIDER uses a newer MJWP stack than Do-as-I-Do:

```text
mujoco == 3.7.0
mujoco-warp == 3.7.0.1
warp-lang == 1.12.1
```

If heavy GPU dependencies are already installed, re-apply only the editable
source link:

```bash
cd "$AOE_RETARGET_LAB_ROOT/third_party/SPIDER"
pip install --no-deps -e .
```

Verify:

```bash
python - <<'PY'
import pathlib, spider, mujoco, mujoco_warp, warp
print("spider", pathlib.Path(spider.__file__).resolve())
print("mujoco", mujoco.__version__)
print("mujoco_warp", getattr(mujoco_warp, "__version__", "unknown"))
print("warp", warp.__version__)
PY
```

The `spider` path should resolve inside `third_party/SPIDER/spider/`.

## 11. Prepare Open-AoE Inputs

For EgoInfinity:

```text
raw_video_undistorted.mp4
```

For Do-as-I-Do, provide a complete reconstruction directory containing:

```text
config.json
gravity.json
obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
video_segmentation/masks/frame_*_masks/<object>/<object>.obj
*/all_hand_meshes.npz
```

For `hand_source=aoe`, provide Open-AoE hand reconstruction such as
`ego_hands_reconstruction/hands.npz` and make sure the selected adapter can
convert it into the target retargeter format.

## 12. Preflight

```bash
source local_env.sh

$EGOINFINITY_PYTHON -c "import torch, mujoco; print('egoinfinity ok', torch.cuda.is_available())"
$SAM3_PYTHON -c "import torch; print('sam3 ok', torch.cuda.is_available())"
$SAM3D_PYTHON -c "import torch; print('sam3d ok', torch.cuda.is_available())"
$RETARGETING_PYTHON -c "import mujoco, warp; print('dai-retarget ok')"
$SPIDER_PYTHON -c "import spider, mujoco, mujoco_warp, warp; print('spider ok')"
```

## 13. Smoke Demo

```bash
source local_env.sh

V4_RUN_NAME=smoke_v4 \
V4_EGO_VIDEO=/path/to/raw_video_undistorted.mp4 \
V4_EGO_CLIP_ID=my_clip \
V4_EGO_OBJECTS="bottle of vinegar" \
V4_EGO_START=0.0 \
V4_EGO_END=3.0 \
V4_DAI_RAW_DIR=/path/to/do_as_i_do_raw_dir \
V4_DAI_CLIP_DIR=/path/to/do_as_i_do_clip_dir \
V4_TASK=vinegar_demo \
V4_HAND_TYPE=bimanual \
MAIN_CUDA=0 \
SAM3_WORKER_CUDA=1 \
SAM3D_WORKER_CUDA=2 \
scripts/run_v4_two_full_pipelines.sh
```

Expected:

```text
experiments/smoke_v4/videos/v4_egoinfinity_full__triptych.mp4
experiments/smoke_v4/videos/v4_do_as_i_do_full__triptych.mp4
experiments/smoke_v4/reuse/reuse_manifest.json
```

## 14. Git Hygiene

Do not commit `third_party/`, Open-AoE raw data, generated `experiments/`,
conda or venv directories, Hugging Face caches, model weights, generated `npz`
files, videos, robot meshes, or large simulator assets.

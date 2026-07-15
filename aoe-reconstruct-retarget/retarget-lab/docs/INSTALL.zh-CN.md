# 安装文档

语言： [English](INSTALL.md) | **中文**

导航： [README](README.zh-CN.md) | [API](API.zh-CN.md) | [原理](PRINCIPLES.zh-CN.md)

本文档面向 public 用户，说明如何 clone AoE Retarget Lab、接入自己的
Open-AoE 数据、安装第三方包并运行 demo。

本 repo 不会 vendor `third_party/EgoInfinity`、`third_party/do-as-i-do`、
`third_party/SPIDER`、机器人 mesh/assets、模型权重、数据集、Hugging Face
cache、MANO 文件、conda 环境或生成结果。第三方源码由
`scripts/setup_third_party.sh` 在本机拉取到已 gitignore 的 `third_party/`。

## 1. 系统要求

推荐环境：

```text
Linux x86_64
NVIDIA GPU + CUDA 12.x
conda 或 mamba
git
ffmpeg
gcc/g++, cmake, ninja, pkg-config
支持 EGL headless MuJoCo 渲染的 NVIDIA driver
```

不同上游包的 CUDA、MuJoCo、Warp、JAX、PyTorch 和 Python 版本约束会冲突，
建议拆成多个环境：

| 环境 | Python | 用途 |
| --- | --- | --- |
| `aoe-base` | 3.10+ | repo 轻量工具 |
| `egoinfinity` | 3.10 | EgoInfinity 和 G1 retargeting |
| `sam3` | 以上游为准 | SAM3/SAM3.1 segmentation worker |
| `sam3d` | 以上游为准 | SAM3D Objects、Fast-SAM3D、MoGe |
| `hawor` | 以上游为准 | HaWoR 和 projection 诊断 |
| `dai-retarget` | 3.12 | Do-as-I-Do Sharpa / MuJoCo-Warp |
| `spider` | 3.12 | SPIDER / MJWP / XHand |

只安装你需要运行的管线对应环境即可。

## 2. Clone 本 repo

```bash
git clone <AOE_RETARGET_LAB_GIT_URL> aoe-retarget-lab
cd aoe-retarget-lab

conda create -y -n aoe-base python=3.10 pip
conda activate aoe-base
pip install -e .

export AOE_RETARGET_LAB_ROOT="$PWD"
```

## 3. 拉取第三方源码

clone 后先运行：

```bash
bash scripts/setup_third_party.sh
```

脚本会在本机生成：

```text
third_party/EgoInfinity/
third_party/do-as-i-do/
third_party/SPIDER/
third_party/THIRD_PARTY_LOCK.tsv
```

这些路径只属于本机环境，不提交到 Git。脚本默认使用官方上游：

```text
EgoInfinity  -> https://github.com/Rice-RobotPI-Lab/EgoInfinity.git
Do-as-I-Do  -> https://github.com/malik-group/do-as-i-do.git
SPIDER      -> https://github.com/facebookresearch/spider.git
```

正式复现或 release 时建议显式 pin 到 commit 或 tag：

```bash
EGOINFINITY_REF=<commit-or-tag> \
DO_AS_I_DO_REF=<commit-or-tag> \
SPIDER_REF=<commit-or-tag> \
bash scripts/setup_third_party.sh
```

`THIRD_PARTY_LOCK.tsv` 会记录本机实际 checkout 到的 commit。

## 4. 配置本机路径

创建一个本机专用的 `local_env.sh`，不要提交它。

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

cache 准备好之后，可以切到离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 5. 外部模型资产

Do-as-I-Do 和 EgoInfinity 使用不同 SAM 版本：

```text
Do-as-I-Do reconstruction Stage 1 -> facebook/sam3, sam3.pt
EgoInfinity object segmentation     -> facebook/sam3.1, sam3.1_multiplex.pt
```

不要把 Do-as-I-Do Stage 1 静默替换成 SAM3.1。检查 cache：

```bash
find -L "$HF_HOME/hub/models--facebook--sam3" -name sam3.pt -print
find -L "$HF_HOME/hub/models--facebook--sam3.1" -name sam3.1_multiplex.pt -print
```

其他常见必需资产：

```text
EgoInfinity retarget checkpoints，例如 retarget/ckpts/g1.pt
MANO_LEFT.pkl / MANO_RIGHT.pkl
MANO_LEFT.npz / MANO_RIGHT.npz
MoGe weights
Fast-SAM3D / SAM3D Objects configs and checkpoints
TAPIR / BootsTAPIR checkpoint
HaWoR and Metric3D weights
```

这些文件都应放在 Git repo 外部。

## 6. EgoInfinity 环境

```bash
conda env create -f third_party/EgoInfinity/retarget/environment.yml
conda activate egoinfinity
pip install -e "$AOE_RETARGET_LAB_ROOT"

export EGOINFINITY_PYTHON=$(which python)
```

验证：

```bash
python - <<'PY'
import mujoco, torch, trimesh
print("mujoco", mujoco.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("trimesh", trimesh.__version__)
PY
```

## 7. SAM3 和 SAM3D worker

按 SAM3 和 SAM3D Objects 上游说明安装，然后导出 Python 和源码/模型路径：

```bash
export SAM3_PYTHON=/path/to/sam3_env/bin/python
export SAM3_REPO=/path/to/sam3_source

export SAM3D_PYTHON=/path/to/sam3d_env/bin/python
export SAM3D_REPO=/path/to/sam-3d-objects
export SAM3D_REPO_ROOT=/path/to/fastsam3d_or_sam3d_config_root

export DINOV2_LOCAL_REPO=/path/to/facebookresearch_dinov2_main
export DINOV2_REPO_DIR=$DINOV2_LOCAL_REPO
```

验证：

```bash
$SAM3_PYTHON -c "import torch; print('sam3 cuda', torch.cuda.is_available())"
$SAM3D_PYTHON -c "import torch; print('sam3d cuda', torch.cuda.is_available())"
```

如果任务中断，旧 SAM3D worker 可能继续占 GPU。长任务前检查：

```bash
pgrep -af "egoinfinity_sam3d|sam3d_worker.py|sam3_worker.py"
nvidia-smi
```

只清理确认是旧 run 遗留的 orphan 进程。

## 8. HaWoR / projection 诊断环境

按 HaWoR 和 Do-as-I-Do reconstruction 上游说明安装，然后设置：

```bash
export HAWOR_PYTHON=/path/to/hawor_env/bin/python
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

验证：

```bash
$HAWOR_PYTHON -c "import torch; print('hawor cuda', torch.cuda.is_available())"
```

Do-as-I-Do mesh projection 诊断需要带 `pytorch3d` 的环境，不要默认认为
Do-as-I-Do retargeting 环境里有它。

## 9. Do-as-I-Do Retargeting / Sharpa

```bash
conda create -y -n dai-retarget python=3.12 pip
conda activate dai-retarget

cd "$AOE_RETARGET_LAB_ROOT/third_party/do-as-i-do/retargeting"
pip install -e .
cd "$AOE_RETARGET_LAB_ROOT"

export RETARGETING_PYTHON=$(which python)
```

该包固定使用 `mujoco==3.4.0`、`warp-lang==1.10.1` 和指定 revision 的
`mujoco-warp`。验证：

```bash
python - <<'PY'
import mujoco, torch, warp
print("mujoco", mujoco.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp", warp.__version__)
PY
```

本 repo 的 wrapper 使用官方入口：

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

SPIDER 使用比 Do-as-I-Do 更新的 MJWP stack：

```text
mujoco == 3.7.0
mujoco-warp == 3.7.0.1
warp-lang == 1.12.1
```

如果重 GPU 依赖已经装好，只需要重新指向 editable 源码：

```bash
cd "$AOE_RETARGET_LAB_ROOT/third_party/SPIDER"
pip install --no-deps -e .
```

验证：

```bash
python - <<'PY'
import pathlib, spider, mujoco, mujoco_warp, warp
print("spider", pathlib.Path(spider.__file__).resolve())
print("mujoco", mujoco.__version__)
print("mujoco_warp", getattr(mujoco_warp, "__version__", "unknown"))
print("warp", warp.__version__)
PY
```

`spider` 路径应指向 `third_party/SPIDER/spider/`。

## 11. 准备 Open-AoE 输入

EgoInfinity 需要：

```text
raw_video_undistorted.mp4
```

Do-as-I-Do 需要完整 reconstruction 目录，至少包含：

```text
config.json
gravity.json
obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
video_segmentation/masks/frame_*_masks/<object>/<object>.obj
*/all_hand_meshes.npz
```

如果使用 `hand_source=aoe`，需要提供 Open-AoE hand reconstruction，例如
`ego_hands_reconstruction/hands.npz`，并确认所选 adapter 能转成目标
retargeter 所需格式。

## 12. Preflight

```bash
source local_env.sh

$EGOINFINITY_PYTHON -c "import torch, mujoco; print('egoinfinity ok', torch.cuda.is_available())"
$SAM3_PYTHON -c "import torch; print('sam3 ok', torch.cuda.is_available())"
$SAM3D_PYTHON -c "import torch; print('sam3d ok', torch.cuda.is_available())"
$RETARGETING_PYTHON -c "import mujoco, warp; print('dai-retarget ok')"
$SPIDER_PYTHON -c "import spider, mujoco, mujoco_warp, warp; print('spider ok')"
```

## 13. Smoke demo

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

预期输出：

```text
experiments/smoke_v4/videos/v4_egoinfinity_full__triptych.mp4
experiments/smoke_v4/videos/v4_do_as_i_do_full__triptych.mp4
experiments/smoke_v4/reuse/reuse_manifest.json
```

## 14. Git hygiene

不要提交 `third_party/`、Open-AoE raw data、生成的 `experiments/`、
conda/venv 目录、Hugging Face cache、模型权重、生成的 `npz`、视频文件、
机器人 mesh 或大型仿真资产。

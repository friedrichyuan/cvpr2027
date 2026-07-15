# API 文档

语言： [English](API.md) | **中文**

导航： [README](README.zh-CN.md) | [安装](INSTALL.zh-CN.md) | [原理](PRINCIPLES.zh-CN.md)

本文档描述 repo 的脚本入口、输入输出和实验资产布局。命令默认从 repo root 运行：

```bash
cd "$AOE_RETARGET_LAB_ROOT"
source local_env.sh
```

所有生成结果必须写到：

```text
experiments/<run_name>/
```

## 1. Cell 命名

```text
traj_<egoinfinity|do_as_i_do>__hand_<aoe|estimated>__retarget_<egoinfinity|do_as_i_do|spider>
```

| 轴 | 选项 | 含义 |
| --- | --- | --- |
| `trajectory_6dof` | `egoinfinity`, `do_as_i_do` | object 6DoF、mesh、depth、overlay 来源 |
| `hand_source` | `aoe`, `estimated` | Open-AoE 手部数据，或所选管线估计出的手部动作 |
| `retargeting` | `egoinfinity`, `do_as_i_do`, `spider` | 机器人重定向和渲染后端 |

`hand_source` 属于共享的 retargeting 输入层，不属于某一个 retargeter。每个
cell 的 manifest 应记录真实使用的 hand source 资产。

## 2. 推荐主入口：完整 12 demos

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

该脚本会：

1. 先调用 `scripts/run_v4_two_full_pipelines.sh`，生成一个 source run。
2. 再调用 `scripts/reuse_v4_for_12_demos.py`，把中间结果展开成 12 个 cell。
3. 默认对 Do-as-I-Do trajectory 输入运行 SPIDER cell。
4. 将 triptych review 视频和 manifest 都写入 `experiments/`。

预期输出：

```text
experiments/<source_run_name>/videos/v4_egoinfinity_full__triptych.mp4
experiments/<source_run_name>/videos/v4_do_as_i_do_full__triptych.mp4
experiments/<source_run_name>/reuse/reuse_manifest.json
experiments/<matrix_run_name>/videos/<cell>__triptych.mp4
experiments/<matrix_run_name>/reuse_12_demo_manifest.json
experiments/<matrix_run_name>/cells/<cell>/manifest.json
```

如果已有 source run，不想重跑 SAM3D、EgoInfinity 或 Do-as-I-Do：

```bash
scripts/run_full_12_demos.sh \
  --source-run <source_run_name> \
  --matrix-run-name <matrix_run_name> \
  --task <task_name> \
  --hand-type bimanual \
  --spider-cuda-visible-devices 3
```

常用选项：

| 选项 | 含义 |
| --- | --- |
| `--source-run` | 跳过两条完整管线，复用 `experiments/<source_run>/` |
| `--matrix-run-name` | 12 demos 的输出 run |
| `--run-spider none|do_as_i_do|all` | 控制 SPIDER cell |
| `--duration <sec>` | 截断合成视频；`0` 表示使用原时长 |
| `--mode symlink|hardlink|copy` | 复用资产落盘方式 |
| `--force-spider` | 已有 SPIDER cell 也重新跑 |
| `--no-spider-fallback` | EgoInfinity+SPIDER 展示 cell 不复用 Do-as-I-Do SPIDER robot 视频 |

## 3. 仅运行两条完整管线

```bash
V4_RUN_NAME=<run_name> \
V4_EGO_VIDEO=/path/to/raw_video_undistorted.mp4 \
V4_EGO_CLIP_ID=<clip_id> \
V4_EGO_OBJECTS="bottle of vinegar" \
V4_EGO_START=0.0 \
V4_EGO_END=3.0 \
V4_DAI_RAW_DIR=/path/to/do_as_i_do_raw_dir \
V4_DAI_CLIP_DIR=/path/to/do_as_i_do_clip_dir \
V4_TASK=<task_name> \
V4_HAND_TYPE=bimanual \
MAIN_CUDA=0 \
SAM3_WORKER_CUDA=1 \
SAM3D_WORKER_CUDA=2 \
scripts/run_v4_two_full_pipelines.sh
```

该脚本会：

1. 启动 SAM3/SAM3.1 和 SAM3D worker。
2. 完整运行一次 EgoInfinity。
3. 完整运行一次 Do-as-I-Do，并走官方 Sharpa 入口。
4. 写入可复用资产和 reuse manifest。

重要变量：

| 变量 | 含义 |
| --- | --- |
| `V4_RUN_NAME` | `experiments/` 下的实验目录 |
| `V4_EGO_VIDEO` | EgoInfinity 使用的 AoE undistorted RGB mp4 |
| `V4_EGO_CLIP_ID` | 写入 EgoInfinity metadata 的 clip id |
| `V4_EGO_OBJECTS` | EgoInfinity object prompt |
| `V4_EGO_START`, `V4_EGO_END`, `V4_EGO_FPS` | EgoInfinity 截取时间和帧率 |
| `V4_DAI_RAW_DIR` | 完整 Do-as-I-Do reconstruction/retarget raw 目录 |
| `V4_DAI_CLIP_DIR` | Do-as-I-Do 可视化 clip 目录 |
| `V4_TASK` | 输出路径使用的稳定 task 名 |
| `V4_HAND_TYPE` | `right`, `left`, `bimanual` |
| `MAIN_CUDA` | 主流程 GPU |
| `SAM3_WORKER_CUDA` | segmentation worker GPU |
| `SAM3D_WORKER_CUDA` | SAM3D worker GPU |

预期输出：

```text
experiments/<run>/videos/v4_egoinfinity_full__triptych.mp4
experiments/<run>/videos/v4_do_as_i_do_full__triptych.mp4
experiments/<run>/reuse/reuse_manifest.json
experiments/<run>/run_env.txt
experiments/<run>/logs/
```

## 4. EgoInfinity 6DoF + G1 retargeting

```bash
scripts/run_egoinfinity_retarget.sh \
  --clip-dir /path/to/raw_video_undistorted.mp4 \
  --clip-id <clip_id> \
  --objects "<object prompt>" \
  --start <sec> \
  --end <sec> \
  --fps 15 \
  --run-name <run_name> \
  --robot g1
```

输入：

```text
AoE undistorted RGB mp4
object text prompt
start/end/fps
```

常用环境变量：

| 变量 | 含义 |
| --- | --- |
| `EGOINFINITY_OBJECT_SELECTION_MODE=all|best` | 保留所有稳定实例，或选择一个实例 |
| `EGOINFINITY_TARGET_POINT=x,y` | 同 prompt 多物体时的像素提示 |
| `EGOINFINITY_MAX_CENTROID_JUMP_PX` | object 连续性阈值 |
| `EGOINFINITY_POST_SCALE_SANITY=0|1` | 是否运行 scale sanity |
| `EGOINFINITY_SCALE_SANITY_THRESHOLD` | mask/depth 尺度 sanity 阈值 |
| `EGOINFINITY_RESET_OUTPUT=1` | 删除当前 clip cache 并重跑 |
| `SAM3_WORKER_SOCKET`, `SAM3D_WORKER_SOCKET` | 复用外部 worker |

主要输出：

```text
experiments/<run>/intermediates/egoinfinity/clip/pipeline_result.pkl.gz
experiments/<run>/intermediates/egoinfinity/clip/mask_overlay.mp4
experiments/<run>/intermediates/egoinfinity/clip/rgb_mesh_overlay.mp4
experiments/<run>/intermediates/egoinfinity/clip/mesh_pure_camera.mp4
experiments/<run>/intermediates/egoinfinity/clip/retarget_samples/depth.mp4
experiments/<run>/intermediates/egoinfinity/clip/retarget/g1/robot_sim.mp4
experiments/<run>/intermediates/egoinfinity/clip/object_filter_report.json
experiments/<run>/logs/retarget_egoinfinity_g1.log
```

最终 robot render 使用 `retarget/g1/robot_sim.mp4`。不要把 input visualization、
fingertip marker 或 skeleton debug view 当作最终 robot 输出。

## 5. Do-as-I-Do 6DoF 准备

```bash
scripts/prepare_do_as_i_do_trajectory_6dof.sh \
  --run-name <run_name> \
  --task <task_name> \
  --raw-dir <complete_do_as_i_do_raw_dir>
```

raw 目录必须包含：

```text
config.json
gravity.json
obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
video_segmentation/masks/frame_*_masks/<object>/<object>.obj
*/all_hand_meshes.npz
```

输出：

```text
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/raw_dir
experiments/<run>/assets/trajectory_6dof/do_as_i_do/<task>/object_6dof.npz
experiments/<run>/assets/trajectory_6dof/do_as_i_do/<task>/object_meshes/visual.obj
experiments/<run>/assets/trajectory_6dof/do_as_i_do/<task>/source_trajectory_keypoints.npz
experiments/<run>/assets/trajectory_6dof/do_as_i_do/<task>/task_info.json
experiments/<run>/logs/trajectory_6dof_do_as_i_do.log
```

生成 Do-as-I-Do overlay/depth/纯 mesh 诊断：

```bash
$RETARGETING_PYTHON scripts/materialize_do_as_i_do_visuals.py \
  --run-name <run_name> \
  --clip-dir <do_as_i_do_clip_dir> \
  --task <task_name> \
  --fps 15
```

常见诊断输出：

```text
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/mask_overlay.mp4
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/mesh_overlay.mp4
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/depth.mp4
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/mesh_pure_camera.mp4
```

## 6. Do-as-I-Do 官方 Sharpa retargeting

```bash
scripts/run_do_as_i_do_official_retarget.sh \
  --raw-dir <complete_do_as_i_do_raw_dir> \
  --run-name <run_name> \
  --task <task_name> \
  --trajectory-6dof do_as_i_do \
  --hand-source estimated \
  --hand-type bimanual \
  --robot-type sharpa \
  --max-sim-steps -1
```

主上游入口：

```text
third_party/do-as-i-do/retargeting/launch.py
```

不要把 `decompose_mesh`、`generate_scene`、`solve_ik`、`optimize_physics`
作为主路径逐个 wrapper 调用。首次完整跑可按需使用 `--force`；快速复跑时不要
强制全链路重跑。

输出：

```text
experiments/<run>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs/sharpa/<hand_type>/<task>/0/scene.xml
experiments/<run>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs/sharpa/<hand_type>/<task>/0/trajectory_mjwp.npz
experiments/<run>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs/sharpa/<hand_type>/<task>/0/visualization_mjwp.mp4
experiments/<run>/logs/do_as_i_do_official_retarget.log
```

如果官方入口没有输出 mp4，wrapper 会从：

```text
scene.xml + trajectory_mjwp.npz
```

调用 `scripts/render_mujoco_trajectory.py` 离屏渲染。

## 7. SPIDER / XHand retargeting

```bash
scripts/run_spider_retarget.sh \
  --run-name <run_name> \
  --trajectory-6dof do_as_i_do \
  --hand-source estimated \
  --task <task_name> \
  --hand-type bimanual \
  --robot-type xhand \
  --max-sim-steps -1 \
  --num-samples 1024 \
  --max-num-iterations 16
```

必需输入资产：

```text
experiments/<run>/assets/trajectory_6dof/<pipeline>/<task>/source_trajectory_keypoints.npz
experiments/<run>/assets/trajectory_6dof/<pipeline>/<task>/object_meshes/visual.obj
```

wrapper 走官方 SPIDER 路径：

```text
spider/preprocess/generate_xml.py
spider/preprocess/ik_fast.py
examples/run_mjwp.py
```

输出：

```text
experiments/<run>/intermediates/retargeting/spider/<cell>/dataset/
experiments/<run>/intermediates/retargeting/spider/<cell>/spider_input_manifest.json
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/scene.xml
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/trajectory_kinematic.npz
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/trajectory_mjwp.npz
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/visualization_ik.mp4
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/visualization_mjwp.mp4
```

快速 smoke 参数：

```bash
SPIDER_DEVICE=cuda:0 \
scripts/run_spider_retarget.sh \
  --run-name <run_name> \
  --trajectory-6dof do_as_i_do \
  --hand-source estimated \
  --task <task_name> \
  --hand-type bimanual \
  --robot-type xhand \
  --max-sim-steps 3 \
  --num-samples 128 \
  --max-num-iterations 4
```

## 8. 资产索引和 triptych 合成

把 cell 资产整理到稳定 review 路径：

```bash
$EGOINFINITY_PYTHON scripts/index_cell_assets.py \
  --run-name <run_name> \
  --trajectory-6dof <egoinfinity|do_as_i_do> \
  --hand-source <aoe|estimated> \
  --retargeting <egoinfinity|do_as_i_do|spider> \
  --task <task_name> \
  --hand-type <right|left|bimanual>
```

输出：

```text
experiments/<run>/cells/<cell>/overlay.mp4
experiments/<run>/cells/<cell>/depth.mp4
experiments/<run>/cells/<cell>/robot.mp4
experiments/<run>/cells/<cell>/manifest.json
experiments/<run>/assets/cells/<cell>/asset_manifest.json
```

合成同步视频：

```bash
$EGOINFINITY_PYTHON scripts/compose_triptych.py \
  --overlay experiments/<run>/cells/<cell>/overlay.mp4 \
  --depth experiments/<run>/cells/<cell>/depth.mp4 \
  --robot experiments/<run>/cells/<cell>/robot.mp4 \
  --output experiments/<run>/videos/<cell>__triptych.mp4 \
  --duration 0
```

## 9. 资产布局

```text
experiments/<run>/
  run_env.txt
  logs/
  intermediates/
    egoinfinity/clip/
    trajectory_6dof/
      egoinfinity/
      do_as_i_do/
    retargeting/
      egoinfinity/<cell>/
      do_as_i_do/<cell>/
      spider/<cell>/
  assets/
    trajectory_6dof/
      egoinfinity/<task>/
      do_as_i_do/<task>/
    cells/<cell>/
  cells/<cell>/
    overlay.mp4
    depth.mp4
    robot.mp4
    manifest.json
  videos/
    <cell>__triptych.mp4
  reuse/
    reuse_manifest.json
```

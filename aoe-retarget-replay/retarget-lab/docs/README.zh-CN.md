# AoE Retarget Lab

语言： [English](../README.md) | **中文**

AoE Retarget Lab 用来把 Open-AoE 第一视角 RGB 视频接入两个物体 6DoF
重建管线和三个机器人重定向/渲染后端。主要检查结果是一段同步视频：

```text
原始 RGB + 重建 mesh overlay
深度 / 输入几何
机器人 retargeting 渲染
```

所有新生成的资产都应写入 `experiments/<run_name>/`。不要把 mesh、轨迹、
视频、日志或 manifest 写回 Open-AoE 数据集目录或第三方源码目录。

## 文档入口

| 文档 | English | 中文 |
| --- | --- | --- |
| README | [../README.md](../README.md) | 本文档 |
| 安装 | [INSTALL.md](INSTALL.md) | [INSTALL.zh-CN.md](INSTALL.zh-CN.md) |
| 脚本 API | [API.md](API.md) | [API.zh-CN.md](API.zh-CN.md) |
| 原理介绍 | [PRINCIPLES.md](PRINCIPLES.md) | [PRINCIPLES.zh-CN.md](PRINCIPLES.zh-CN.md) |

建议先看 [安装文档](INSTALL.zh-CN.md)，再看 [API 文档](API.zh-CN.md)。
管线拆分、尺度和可视化约定见 [原理介绍](PRINCIPLES.zh-CN.md)。

## 项目结构

```text
scripts/                     可运行的 shell 和 Python 入口
scripts/setup_third_party.sh 本地拉取第三方上游包的脚本
src/aoe_retarget_lab/        AoE 适配器和工具
configs/                     示例配置
experiments/                 本地实验输出
docs/                        中英双语文档
third_party/                 setup 后生成的本地上游 clone，已 gitignore
```

这个 public 导入包不会 vendor EgoInfinity、Do-as-I-Do、SPIDER、机器人资产、
模型权重、数据集或生成视频。clone 后运行 `scripts/setup_third_party.sh`，
把第三方包拉到已 gitignore 的 `third_party/`。

## 管线矩阵

项目有三条轴：

```text
trajectory_6dof = egoinfinity | do_as_i_do
hand_source     = aoe | estimated
retargeting     = egoinfinity | do_as_i_do | spider
```

完整比较矩阵是：

```text
2 trajectory_6dof x 2 hand_source x 3 retargeting = 12 cells
```

cell 名称固定为：

```text
traj_<trajectory_6dof>__hand_<hand_source>__retarget_<retargeting>
```

## 快速开始

先完成环境安装、拉取第三方包，并配置外部资产路径：

```bash
bash scripts/setup_third_party.sh
source local_env.sh
```

推荐运行 full-12 主入口。它先完整跑一次 EgoInfinity 和一次 Do-as-I-Do，
再复用这两条大链路的中间结果，合成 12 个对比 cell。

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

预期输出：

```text
experiments/<source_run_name>/videos/v4_egoinfinity_full__triptych.mp4
experiments/<source_run_name>/videos/v4_do_as_i_do_full__triptych.mp4
experiments/<source_run_name>/reuse/reuse_manifest.json
experiments/<matrix_run_name>/videos/<cell>__triptych.mp4
experiments/<matrix_run_name>/reuse_12_demo_manifest.json
experiments/<matrix_run_name>/cells/
experiments/<matrix_run_name>/assets/
```

## Demo 验收规则

- overlay 行必须是原始 RGB 加重建 hand mesh 和 object mesh/point cloud，
  不能只用 segmentation mask、骨架或 fingertips。
- robot 行必须显示机器人本体和灵巧手：EgoInfinity/G1、Do-as-I-Do/Sharpa
  或 SPIDER/XHand。
- 重定向后的轨迹、mesh、`npz`、`scene.xml`、manifest、日志和视频都必须归档在
  `experiments/<run_name>/`。
- Do-as-I-Do retargeting 应使用官方入口
  `third_party/do-as-i-do/retargeting/launch.py`。
- 不要通过硬缩放或手动缩小物体来“修好看”。应定位 mask、mesh 重建、
  6DoF/layout scale 和 retarget adapter 坐标变换。

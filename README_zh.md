# Open-AoE

<p align="center">
  <strong>面向具身智能的开源第一视角数据与工具链</strong>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

> [!IMPORTANT]
> Open-AoE-2000H 数据集及配套技术报告正在准备公开发布。本仓库已经提供开源工具链与数据格式文档；数据下载和正式引用链接将在发布定稿后补充。

<p align="center">
  <img src="docs/fig1-open-aoe-overview.png" width="100%" alt="Open-AoE 数据集、处理流水线与开源工具链总览">
</p>

## 项目概览

Open-AoE 是一个面向社区的大规模真实世界第一视角操作数据项目，全部使用消费级智能手机采集。计划发布的数据包含约 **2,000 小时**第一人称人类操作视频，并提供时间同步的手部运动、相机运动与原子动作标注。

Open-AoE 不止发布视频与标签，还提供一条可复现的 data-to-model 路径：同一段同步数据可以被转换为可检查的可视化、可复用的手物资产、面向机器人本体的运动、robotized video，以及适配不同模型的训练接口。

| 数据规模 | 贡献者 | 设备类型 | 场景 | 任务 |
|---:|---:|---:|---:|---:|
| 约 2,000 小时 | 768 | 200 | 500 | 10,000+ |

### 为什么选择 Open-AoE？

- **智能手机优先。** 相比专用头显或仅依赖机器人的采集方式，消费级设备更容易扩展到真实日常场景。
- **面向操作学习的同步信号。** RGB 视频与相机标定及轨迹、MANO 手部重建、有效性标记和原子动作标注保持时间对齐。
- **完整工具链。** 仓库覆盖数据检查、4D 重建、人机重映射、机器人视频叠加、动作转换和下游训练配方。
- **模块化设计。** 每个组件都可以独立使用；不同下游方法选择与自身任务匹配的表示，而不是被迫使用单一固定格式。
- **可持续社区扩展。** 新机器人本体、重建后端和训练配方可以作为自包含子项目持续加入。

## Open-AoE-2000H 数据集

每个发布片段都是一条时间同步的多模态第一视角操作记录。

| 信号 | 主要文件 | 提供内容 |
|---|---|---|
| 原始与去畸变 RGB | `raw_video.mp4`、`raw_video_undistorted.mp4` | 第一人称视觉观测与标定后视频 |
| 相机元数据 | `video_info.json`、`undistorted_video_info.json` | 设备信息、内参、畸变、分辨率与帧率 |
| 相机运动 | `camera_traj.npz` 及 `hands.npz` 中的变换 | metric-scale 6-DoF 相机轨迹与世界/相机坐标变换 |
| 手部重建 | `hands.npz` | 双手逐帧 MANO 姿态、形状、根节点变换与有效性 |
| 原子动作 | `ego_action_annotation.json` | 时间对齐的动作片段、动词、物体、左右手和描述 |

完整样本目录、字段定义、坐标系约定和验证说明参见 [Open-AoE-2000H 数据规格](open-aoe-2000h/README.md)。

### 数据处理与质量控制

<p align="center">
  <img src="docs/fig2-data-pipline.png" width="100%" alt="Open-AoE 端侧采集、离线处理、重建标注和质量控制流水线">
</p>

发布流水线包含四个阶段：

1. **端侧采集控制**在上传前检查手部可见性、佩戴规范、光照、运动质量和设备状态。
2. **离线质量检查与场景标注**过滤无效或敏感内容，统一帧率，切分视频并生成场景/任务元数据。
3. **重建与标注**估计相机轨迹、重建 MANO 双手并生成原子动作片段。
4. **质量检查与交付**依次执行完整性、正确性和时序一致性检查，并加入人工复核。

## Open-AoE 工具链

工具链将同一条 Open-AoE 同步片段投射到不同机器人学习工作流需要的表示空间。

| 组件 | 作用 | 入口 |
|---|---|---|
| **AoE-Visualization** | 在一个 review video 中检查 RGB、MANO mesh/keypoint、相机运动、动作标注与时间轴 | [`aoe-visualization/`](aoe-visualization/) |
| **AoE-Retarget-Replay** | 重建手物资产，将人类运动映射到不同机器人本体，验证/渲染轨迹并合成 robotized video | [`aoe-retarget-replay/`](aoe-retarget-replay/) |
| **AoE-Training-Ready** | 将同步 AoE 信号转换为模型所需的 state/action 语义与可复现训练配方 | [`aoe-training-ready/`](aoe-training-ready/) |

### AoE-Visualization

AoE-Visualization 为每个样本生成一个端到端复核视频，在去畸变视频上同时展示 MANO 手部 mesh、21 点骨架、未来手腕轨迹、原子动作信息、世界坐标 3D 视图和时间轴。

```bash
cd aoe-visualization
pip install -r requirements.txt

# 单个样本
python visualize.py --sample /path/to/open_aoe_sample

# 样本目录
python visualize.py --data_dir /path/to/open_aoe_data --output_dir ./output
```

渲染依赖和输出说明参见 [AoE-Visualization 使用指南](aoe-visualization/README.md)。

### Reconstruction and Retargeting

<p align="center">
  <img src="docs/fig3-reconstruct-retarget.png" width="100%" alt="Open-AoE 4D 重建、运动重映射和机器人叠加三条路线">
</p>

重建与重映射工具链提供三类互补输出：**4D 手物表示**、**机器人可用运动**，以及将机器人渲染结果合成回真实场景的 **robotized video**。

| 子项目 | 机器人/方法覆盖 | 主要能力 |
|---|---|---|
| [**Phantom**](aoe-retarget-replay/phantom/) | Unitree G1 + Dex3 / Inspire | 臂部 IK、灵巧手重映射、MuJoCo 可视化和 SAM2 + E2FGVI 机器人叠加 |
| [**Retarget Galbot**](aoe-retarget-replay/retarget_galbot/) | Galbot / Galaxea 双臂平台 | Palm-to-TCP Pinocchio IK、平行夹爪映射、MuJoCo egoview 合成和 LeRobot/Rerun 导出 |
| [**AoE Retarget Lab**](aoe-retarget-replay/retarget-lab/) | EgoInfinity/G1、Do-as-I-Do/Sharpa、SPIDER/XHand | 外部方法集成、6-DoF 重建适配和 12-cell 对比矩阵 |

Retarget Lab 不会把第三方仓库、模型权重、机器人资产或生成视频直接 vendor 到本仓库；使用前请按照其安装文档在本地获取并配置上游项目。

### AoE-Training-Ready

<p align="center">
  <img src="docs/fig4-training-ready.png" width="100%" alt="面向 VLA、世界动作模型和世界模型的 Open-AoE Training-Ready 标注谱系">
</p>

Training-Ready 将转换理解为**动作语义适配**，而不是简单的文件格式搬运。同一段数据可以被表示为密集 MANO state/action、面向机器人手或夹爪的动作、手部加相机动态，或者 latent/weak action。

| 模型家族 | 已包含的集成配方 |
|---|---|
| VLA policy | VITRA、GR00T N1.7、H-RDT、ACT、Diffusion Policy、π0.5、SmolVLA |
| World/Video Action Model | DreamZero、LingBot-VA、Ctrl-World、iVideoGPT |
| Latent Action 与 World Model | GenieRedux、LAOM、AdaWorld、DreamDojo |

每个配方独立维护转换或启动脚本、文档，以及必要时使用的上游补丁。请从 [Training-Ready 配方索引](aoe-training-ready/README.md) 和统一的 [动作规格](aoe-training-ready/ACTION_SPEC.md) 开始。

## 快速开始

### 1. 克隆工具链

```bash
git clone https://github.com/woxue/Open-AoE-dev.git
cd Open-AoE-dev
```

### 2. 按需准备 MANO 模型

本仓库不分发 MANO 模型文件。请先在 [MANO 官网](https://mano.is.tue.mpg.de/) 注册并下载 `MANO_RIGHT.pkl` 和 `MANO_LEFT.pkl`，再复制到所有子项目共享的目录：

```bash
bash assets/mano/download_mano.sh \
  ~/Downloads/MANO_RIGHT.pkl \
  ~/Downloads/MANO_LEFT.pkl
```

### 3. 选择工作流

| 目标 | 从这里开始 |
|---|---|
| 了解发布样本格式 | [`open-aoe-2000h/README.md`](open-aoe-2000h/README.md) |
| 渲染和检查样本 | [`aoe-visualization/README.md`](aoe-visualization/README.md) |
| 运动重映射或机器人视频叠加 | [`aoe-retarget-replay/README.md`](aoe-retarget-replay/README.md) |
| 转换为模型训练数据 | [`aoe-training-ready/README.md`](aoe-training-ready/README.md) |

## 仓库结构

```text
Open-AoE-dev/
├── docs/                    # 技术报告框架图
├── open-aoe-2000h/         # 数据格式与使用文档
├── aoe-visualization/      # 同步数据复核与渲染
├── aoe-retarget-replay/    # 重建、重映射、回放与机器人叠加
│   ├── phantom/
│   ├── retarget_galbot/
│   └── retarget-lab/
├── aoe-training-ready/     # 模型适配、转换脚本、启动器与补丁
├── assets/mano/            # MANO 共享配置脚本；模型文件不纳入 Git
├── CONTRIBUTING.md
├── LEGAL.md
├── LICENSE
└── README_old.md           # 归档的协作/进展记录版 README
```

## 发布资源

- **Open-AoE-2000H 数据集：** 即将发布
- **技术报告：** 即将发布
- **AoE 采集应用：** 发布信息即将补充
- **详细数据规格：** [open-aoe-2000h/README.md](open-aoe-2000h/README.md)

## 参与贡献

我们欢迎社区贡献新的机器人本体、重建或重映射后端、可视化能力、数据转换器和训练配方。提交 PR 前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。每个集成应保持自包含并说明外部依赖；请勿提交数据集、模型权重、MANO 文件或许可证不兼容的第三方源码。

## 许可证与第三方组件

本仓库原创代码以 [Apache License 2.0](LICENSE) 发布。数据集分发条款、模型权重、机器人资产和第三方组件可能采用不同许可证；重新分发或商业使用前请阅读 [LEGAL.md](LEGAL.md) 和对应子项目 README。

## 引用

Open-AoE 技术报告发布后，我们会在此补充正式 BibTeX。

## 致谢

Open-AoE 由 AoE 社区共同建设，并集成了众多开源项目的思想与接口。感谢所有数据贡献者、工具链贡献者、维护者和上游研究团队。组件来源和许可证声明统一维护在 [LEGAL.md](LEGAL.md) 中。

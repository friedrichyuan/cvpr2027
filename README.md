# Open-AoE

**Open-AoE: The First Open-Source Egocentric Dataset with a Complete Data-to-Model Toolchain, Collected Entirely from Consumer Smartphones**

延续 AoE (Always-on Egocentric) CVPR Workshop 2026 的工作，Open-AoE 开源 2000 小时 ego-centric 操作数据及完整工具链，打通**原始数据→可视化→机器人重映射→模型训练**的最后一公里。

---

## 项目结构

```
Open-AoE/
├── README.md                ← 项目总览
├── CONTRIBUTING.md          ← 贡献指南
├── CLAUDE.md                ← AI 协作规范
├── LICENSE                  ← Apache 2.0
├── LEGAL.md                 ← 第三方依赖许可证声明
├── assets/                  ← 公共资源
│   └── mano/                ← MANO 模型下载脚本
├── open-aoe-2000h/          ← 数据集文档 / Dataset
├── aoe-visualization/       ← 数据可视化 / Visualization
├── aoe-retarget-replay/     ← Human-to-Robot 重映射 / 人机重映射
│   └── phantom/             ← [Phantom](aoe-retarget-replay/phantom/) (G1 + Dex3/Inspire)
└── aoe-training-ready/      ← 模型训练格式转换 / Training-Ready
    ├── vitra/               ← [VITRA](aoe-training-ready/vitra/)
    ├── gr00t_n1d7/          ← [GR00T N1.7](aoe-training-ready/gr00t_n1d7/)
    ├── H-RDT/               ← [H-RDT](aoe-training-ready/H-RDT/)
    ├── dreamzero/           ← [DreamZero](aoe-training-ready/dreamzero/)
    ├── lingbot-va/          ← [LingBot-VA](aoe-training-ready/lingbot-va/)
    ├── Ctrl-World/          ← [Ctrl-World](aoe-training-ready/Ctrl-World/)
    ├── lerobot/             ← [LeRobot](aoe-training-ready/lerobot/) (ACT / DP / pi0.5)
    ├── smolvla/             ← [SmolVLA](aoe-training-ready/smolvla/)
    ├── ivideogpt/           ← [iVideoGPT](aoe-training-ready/ivideogpt/)
    ├── genie-redux/         ← [GenieRedux](aoe-training-ready/genie-redux/)
    ├── laom/                ← [laom (LAOM)](aoe-training-ready/laom/)
    ├── adaworld/            ← [AdaWorld](aoe-training-ready/adaworld/)
    └── dreamdojo/           ← [DreamDojo](aoe-training-ready/dreamdojo/)
```

## 集成计划

> 状态说明：✅ 已发布 &nbsp;|&nbsp; 🔵 规划中

### 数据集

| 交付物 | 说明 | 状态 | 贡献 |
|--------|------|------|------|
| **Open-AoE-2000H** | 2000 小时 ego-centric 操作视频数据集，包含 MANO 手部重建、相机轨迹、中英双语原子动作描述，通过 HuggingFace 分发 | 🔵 规划中 | — |

### 可视化

| 方法 | 说明 | 状态 | 贡献 |
|------|------|------|------|
| [**AoE-Visualization**](aoe-visualization/) | 将 MANO 手部重建结果叠加到去畸变视频上，同时展示动作标注信息面板和 3D 世界帧，为每个样本生成端到端复核视频 | ✅ 已发布 | [@huaijin2787](https://github.com/huaijin2787) |

### 人机重映射

| 方法 | 说明 | 状态 | 贡献 |
|------|------|------|------|
| [**Phantom**](aoe-retarget-replay/phantom/) | 通用人机动作重映射框架 (G1 + Dex3/Inspire)，支持 MuJoCo 仿真验证和真机回放 | ✅ 已发布 | [@yfan-yang](https://github.com/yfan-yang) [@woxue](https://github.com/woxue) |
| **SPIDER** | Object 6-DoF Trajectory 估计管线 | 🔵 规划中 | [@jiadong5](https://github.com/jiadong5) |
| **EgoInfinity** | Object 6-DoF Trajectory 估计管线 + 小规模数据重定向 | 🔵 规划中 | [@jiadong5](https://github.com/jiadong5) |
| **Do as I do** | 动作重映射方法复现 | 🔵 规划中 | [@jiadong5](https://github.com/jiadong5) |
| **银河通用真机** | Retarget 到银河通用真机 | 🔵 规划中 | [@liuyou1103](https://github.com/liuyou1103) |

### 模型训练

#### VLA（Vision-Language-Action）

| 方法 | 说明 | 状态 | 贡献 |
|------|------|------|------|
| [**VITRA**](aoe-training-ready/vitra/) | 视觉轨迹推理与动作预测 | ✅ 已发布 | [@woxue](https://github.com/woxue) |
| [**GR00T N1.7**](aoe-training-ready/gr00t_n1d7/) | NVIDIA 通用机器人基础模型 (sharpa + gripper 模式) | ✅ 已发布 | [@reallm](https://github.com/reallm) |
| [**H-RDT**](aoe-training-ready/H-RDT/) | 双手操作扩散策略 (48D action) | ✅ 已发布 | [@zhaowenZhou](https://github.com/zhaowenZhou) |
| [**ACT**](aoe-training-ready/lerobot/) | Action Chunking Transformer，通过 LeRobot 集成 | ✅ 已发布 | [@William-wAng618](https://github.com/William-wAng618) [@ChaduCheng](https://github.com/ChaduCheng) |
| [**DP**](aoe-training-ready/lerobot/) | Diffusion Policy，通过 LeRobot 集成 | ✅ 已发布 | [@William-wAng618](https://github.com/William-wAng618) [@ChaduCheng](https://github.com/ChaduCheng) |
| [**pi0.5**](aoe-training-ready/lerobot/) | π0.5 物理智能模型，通过 LeRobot 集成 | ✅ 已发布 | [@William-wAng618](https://github.com/William-wAng618) [@ChaduCheng](https://github.com/ChaduCheng) |
| [**DreamZero**](aoe-training-ready/dreamzero/) | WAM/VAM 视频动作模型 | ✅ 已发布 | [@William-wAng618](https://github.com/William-wAng618) [@ChaduCheng](https://github.com/ChaduCheng) |
| [**LingBot-VA**](aoe-training-ready/lingbot-va/) | 灵巧手视觉动作模型 | ✅ 已发布 | [@William-wAng618](https://github.com/William-wAng618) [@ChaduCheng](https://github.com/ChaduCheng) |
| [**iVideoGPT**](aoe-training-ready/ivideogpt/) | 视频生成式世界模型用于动作预测 | ✅ 已发布 | [@CharlesPikachu](https://github.com/CharlesPikachu) [@ys-feng](https://github.com/ys-feng) |
| [**GenieRedux**](aoe-training-ready/genie-redux/) | 探索驱动的生成式交互环境 | ✅ 已发布 | [@CharlesPikachu](https://github.com/CharlesPikachu) [@ys-feng](https://github.com/ys-feng) |
| [**SmolVLA**](aoe-training-ready/smolvla/) | 轻量级视觉-语言-动作模型 (微调) | ✅ 已发布 | [@CharlesPikachu](https://github.com/CharlesPikachu) [@ys-feng](https://github.com/ys-feng) |

#### World Model

| 方法 | 说明 | 状态 | 贡献 |
|------|------|------|------|
| [**Ctrl-World**](aoe-training-ready/Ctrl-World/) | 可控世界模型，从手部动作预测未来视频帧 | ✅ 已发布 | [@William-wAng618](https://github.com/William-wAng618) [@ChaduCheng](https://github.com/ChaduCheng) |
| [**LAOM**](aoe-training-ready/laom/) | 大规模动作观测模型 | ✅ 已发布 | [@CharlesPikachu](https://github.com/CharlesPikachu) [@ys-feng](https://github.com/ys-feng) |
| [**DreamDojo**](aoe-training-ready/dreamdojo/) | 世界模型驱动的机器人技能学习 — **zero-shot preview only**（数据管道 + 推理已通，post-train 待 8×H100） | 🔵 规划中 | [@CharlesPikachu](https://github.com/CharlesPikachu) [@ys-feng](https://github.com/ys-feng) |
| [**AdaWorld**](aoe-training-ready/adaworld/) | 自适应世界模型 | ✅ 已发布 | [@CharlesPikachu](https://github.com/CharlesPikachu) [@ys-feng](https://github.com/ys-feng) |

## 核心差异化

- **成本**: 仅需一台消费级手机即可采集数据，单人硬件准备成本 <$20（vs 竞品 $300–$3,500）
- **工具链**: 唯一同时提供 Visualization + Retarget-Replay + Training-Ready 的开源数据集
- **可参与性**: 任何人用自己的手机即可贡献数据

## 前置准备

Open-AoE 中的多个工具需要 MANO 手部模型进行渲染和重映射。MANO 模型需要用户在官网注册后下载，详见 [MANO 许可](https://mano.is.tue.mpg.de/license.html)。

```bash
# 1. 注册并下载 MANO 模型: https://mano.is.tue.mpg.de/
# 2. 运行下载脚本，将模型复制到共享目录
bash assets/mano/download_mano.sh ~/Downloads/MANO_RIGHT.pkl ~/Downloads/MANO_LEFT.pkl
```

完成后，所有子项目将自动发现 MANO 模型。你也可以在具体子项目中根据提示单独准备。

## 快速开始

### 数据访问 / Dataset

数据集托管在 HuggingFace：

> 🔗 [数据集链接]（待发布）

### 可视化 / Visualization

```bash
cd aoe-visualization
pip install -r requirements.txt
python visualize.py --data <path_to_aoe_data>
```

详见 [aoe-visualization/README.md](aoe-visualization/README.md)

### 模型训练 / Training-Ready

```bash
# VITRA 训练配方
cd aoe-training-ready/vitra
# 详见 README.md

# GR00T N1.7 训练配方
cd aoe-training-ready/gr00t_n1d7
# 详见 README.md

# H-RDT 训练配方
cd aoe-training-ready/H-RDT
# 详见 README.md

# DreamZero 训练配方
cd aoe-training-ready/dreamzero
# 详见 README_OPEN_AOE.md

# LingBot-VA 训练配方
cd aoe-training-ready/lingbot-va
# 详见 README_OPEN_AOE.md

# Ctrl-World 训练配方
cd aoe-training-ready/Ctrl-World
# 详见 README_OPEN_AOE.md

# LeRobot 训练配方 (ACT / DP / pi0.5)
cd aoe-training-ready/lerobot
# 详见 README_OPEN_AOE.md

# 世界模型 / VLA 训练配方 (SmolVLA / iVideoGPT / GenieRedux / laom / AdaWorld / DreamDojo)
cd aoe-training-ready/ivideogpt
# 详见 README_OPEN_AOE.md
```

详见 [aoe-training-ready/README.md](aoe-training-ready/README.md)

### 人机重映射 / Retarget-Replay

```bash
# Phantom 重映射方法 (G1 + Dex3/Inspire)
cd aoe-retarget-replay/phantom
# 详见 README.md
```

## 致谢

Open-AoE 建立在以下优秀开源项目之上，感谢所有贡献者：

| 项目 | 用途 | 许可证 |
|------|------|--------|
| [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 模型训练框架 | Apache 2.0 |
| [H-RDT](https://github.com/HongzheBi/H_RDT) | 双手操作扩散策略 | Apache 2.0 |
| [Phantom](https://github.com/MarionLepert/phantom) | 人机动作重映射 | MIT |
| [VITRA](https://github.com/microsoft/VITRA) | 视觉轨迹推理 | MIT |
| [dex-retargeting](https://github.com/dex-retargeting/dex-retargeting) | 手指重映射 | MIT |
| [MANO](https://mano.is.tue.mpg.de/) | 手部模型 | MANO License |
| [HaWoR](https://github.com/ThunderVVV/HaWoR) | 手部重建 | CC-BY-NC-ND 4.0 |
| [iVideoGPT](https://github.com/thuml/iVideoGPT) | 动作条件世界模型 | MIT |
| [GenieRedux](https://github.com/insait-institute/GenieRedux) | 潜动作生成式世界模型 (Genie) | MIT |
| [LeRobot / SmolVLA](https://github.com/huggingface/lerobot) | VLA 策略训练 | Apache 2.0 |
| [LAOM](https://github.com/dunnolab/laom) | 潜动作学习（带监督） | Apache 2.0 |
| [AdaWorld](https://github.com/Little-Podi/AdaWorld) | 潜动作世界模型 | Apache 2.0 |
| [DreamDojo](https://github.com/NVIDIA/DreamDojo) | 世界模型 (Cosmos-Predict2.5) | Apache 2.0 |

## 相关链接

- **技术报告**: [arXiv 链接]（待发布）
- **数据集**: [HuggingFace 链接]（待发布）
- **AoE 论文** (CVPR Workshop 2026): [链接]（待发布）
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **第三方许可**: [LEGAL.md](LEGAL.md)

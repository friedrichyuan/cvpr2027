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
    └── H-RDT/               ← [H-RDT](aoe-training-ready/H-RDT/)
```

## 开源交付物

| 交付物 | 形态 | 价值锚点 | 状态 |
|--------|------|---------|------|
| **Open-AoE-2000H** | 2000H 视频 + 原子动作描述(双语) + MANO 标注 + 相机轨迹 | 1000H 是加入基模 pretrain 的入门门槛 | 🟡 数据筛选中 |
| **AoE-Visualization** | 数据可视化与渲染 / Visualization | 方便使用者进行数据洞察、过滤、对比 | ✅ 已发布 |
| **AoE-Retarget-Replay** | 人机动作重映射 + 真机回放 | 对有真机的团队，直接 replay 到 G1 等机器人 | ✅ 已发布（[Phantom](aoe-retarget-replay/phantom/)，规划 [AGILE]() / [SPIDER]()） |
| **AoE-Training-Ready** | 模型训练格式转换 + 多模型 recipe | 打通主流 VLA 模型训练 | ✅ 已发布（[VITRA](aoe-training-ready/vitra/)、[GR00T N1.7](aoe-training-ready/gr00t_n1d7/)、[H-RDT](aoe-training-ready/H-RDT/)） |

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

## 相关链接

- **技术报告**: [arXiv 链接]（待发布）
- **数据集**: [HuggingFace 链接]（待发布）
- **AoE 论文** (CVPR Workshop 2026): [链接]（待发布）
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **第三方许可**: [LEGAL.md](LEGAL.md)
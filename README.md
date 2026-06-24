# Open-AoE

**Open-AoE: The First Open-Source Egocentric Dataset with a Complete Data-to-Model Toolchain, Collected Entirely from Consumer Smartphones**

延续 AoE (Always-on Egocentric) CVPR Workshop 2026 的工作，Open-AoE 开源 2000 小时 ego-centric 操作数据及完整工具链，打通**原始数据→可视化→机器人重映射→模型训练**的最后一公里。

---

## 项目结构

```
Open-AoE/
├── README.md                   ← 项目总览
├── CONTRIBUTING.md             ← 贡献指南
├── LICENSE                     ← Apache 2.0
├── LEGAL.md                    ← 第三方依赖许可证声明
│
├── assets/                     ← 公共资源
│   └── mano/                   ← MANO 模型下载脚本
│
├── release/                    ← 对外发布物
│   ├── open-aoe-2000h/         ← 数据集文档与使用指南
│   ├── aoe-visualization/      ← 数据可视化工具
│   ├── aoe-retarget-replay/    ← Human-to-Robot 重映射工具
│   └── aoe-training-ready/     ← 模型训练格式转换工具
│
└── CLAUDE.md                   ← AI 协作规范
```

## 开源交付物

| 交付物 | 形态 | 价值锚点 | 状态 |
|--------|------|---------|------|
| **Open-AoE-2000H** | 2000H 视频 + 原子动作描述(双语) + MANO 标注 + 相机轨迹 | 1000H 是加入基模 pretrain 的入门门槛 | 🟡 数据筛选中 |
| **AoE-Visualization** | 数据浏览器与渲染工具 | 方便使用者进行数据洞察、过滤、对比 | ✅ 已发布 |
| **AoE-Retarget-Replay** | 人机动作重映射 + 真机回放 (Phantom) | 对有真机的团队，直接 replay 到 G1 | ✅ 已发布 |
| **AoE-Training-Ready** | LeRobot 格式转换器 + GR00T / VITRA training recipe | 打通主流 VLA 模型训练 | ✅ 已发布 |

## 核心差异化

- **成本**: <$20/人（vs 竞品 $300–$3,500）
- **工具链**: 唯一同时提供 Visualization + Retarget-Replay + Training-Ready 的开源数据集
- **可参与性**: 任何人用自己的手机即可贡献数据
- **双语标注**: 中英文原子动作描述

## 快速开始

### 数据访问

数据集托管在 HuggingFace：

> 🔗 [数据集链接]（待发布）

### 可视化

```bash
cd release/aoe-visualization
pip install -r requirements.txt
python visualize.py --data <path_to_aoe_data>
```

详见 [release/aoe-visualization/README.md](release/aoe-visualization/README.md)

### 模型训练

```bash
# GR00T N1.7 训练配方
cd release/aoe-training-ready/gr00t_n1d7
# 详见 README.md

# VITRA 训练配方
cd release/aoe-training-ready/vitra
# 详见 README.md
```

详见 [release/aoe-training-ready/README.md](release/aoe-training-ready/README.md)

### 人机重映射

```bash
cd release/aoe-retarget-replay/phantom
# 详见 README.md
```

## 相关链接

- **技术报告**: [arXiv 链接]（待发布）
- **数据集**: [HuggingFace 链接]（待发布）
- **AoE 论文** (CVPR Workshop 2026): [链接]（待发布）
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
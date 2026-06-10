# Open-AoE

**Open-AoE: The First Open-Source Egocentric Dataset with a Complete Data-to-Model Toolchain, Collected Entirely from Consumer Smartphones**

延续 AoE (Always-on Egocentric) CVPR Workshop 2026 的工作，Open-AoE 开源 2000 小时 ego-centric 操作数据及完整工具链，打通**原始数据→可视化→机器人重映射→模型训练**的最后一公里。

---

## 项目总览

```
Open-AoE/
├── README.md              ← 你在这里
├── CLAUDE.md              ← AI 协作规范（Copilot / Agent 读这个文件）
├── STORY.md               ← 行业叙事：为什么做、差异化定位、价值锚点
├── PROJECT.md             ← 多人分工与里程碑追踪
│
├── release/               ← 对外发布物（代码 + 数据）
│   ├── open-aoe-2000h/    ← 开源数据集（2000H 视频 + 标注）
│   ├── aoe-visualization/ ← 数据可视化工具（Rerun）
│   ├── aoe-retarget-replay/ ← Human-to-Robot 重映射工具
│   └── aoe-training-ready/  ← 模型训练格式转换工具
│
├── report/                ← 技术报告（LaTeX）
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── tables/
│
└── inner/                 ← 内部工作区（不对外发布）
    ├── drafts/            ← 草案与规划文档
    ├── research/          ← 调研报告
    ├── sample_data/       ← 样例数据
    ├── prompts/           ← AI 协作 prompt 历史
    └── references/        ← 参考论文与资料
```

## 开源交付物矩阵

| 交付物 | 形态 | 价值锚点 | 状态 |
|--------|------|---------|------|
| **Open-AoE-2000H** | 2000H 视频 + 原子动作描述(双语) + MANO 标注 + 相机轨迹 | 1000H 是被加入基模 pretrain 的入门门槛 | 🟡 数据筛选中 |
| **AoE-Visualization** | 基于 Rerun 的数据浏览器 | 方便使用者进行数据洞察、过滤、对比 | 🔴 未开始 |
| **AoE-Retarget-Replay** | `aoe_retarget_replay.py` + 真机回放 SDK | 对有真机的团队，直接 replay 到 G1 | 🟡 核心脚本 80% |
| **AoE-Training-Ready** | LeRobot 格式转换器 + 多模型 training recipe | 打通 π₀.₅ / GR00T / fastWAM 等 | 🟡 fastWAM 已验证 |
| **Tech Report** | 4 页短报告（或 AoE v3 章节） | 数据分布洞察 + 数据质量实验 | 🔴 未开始 |

## 核心差异化

- **成本**: <$20/人（vs 竞品 $300–$3,500）
- **工具链**: 唯一同时提供 Visualization + Retarget-Replay + Training-Ready 的开源数据集
- **可参与性**: 任何人用自己的手机即可贡献数据
- **双语标注**: 中英文原子动作描述

## 快速开始

```bash
# 1. 浏览样例数据
ls inner/sample_data/

# 2. 查看数据格式
cat inner/sample_data/AoE_dataset_README.md

# 3. 查看调研报告
ls inner/research/
```

## 目标里程碑

- **2026-07 WAIC**: Open-AoE 首发（数据 + 工具链 + Tech Report）
- **2026-09 外滩大会**: 社区反馈迭代版

## 相关链接

- AoE 论文 (CVPR Workshop 2026): `../Paper_Notes/AoE_CVPRW_2026/`
- 项目协作: [PROJECT.md](PROJECT.md)
- AI 协作规范: [CLAUDE.md](CLAUDE.md)
- 行业叙事: [STORY.md](STORY.md)

# CLAUDE.md — Open-AoE AI 协作规范

> 本文件供 AI Agent（Claude / Copilot / Cursor 等）在参与 Open-AoE 项目时阅读，用于快速理解项目上下文并遵守协作约定。

---

## 项目一句话定义

**Open-AoE** 是一个基于消费级智能手机的大规模 ego-centric 操作数据集开源项目，包含 2000 小时数据和完整的 data-to-model 工具链（可视化 / human-to-robot 重映射 / 模型训练格式转换）。

## 关键背景

- **前序工作**: AoE (Always-on Egocentric), CVPR Workshop 2026, 论文位于 `../Paper_Notes/AoE_CVPRW_2026/`
- **DDL**: 2026-07 中下旬 WAIC 首发，2026-09 外滩大会迭代版
- **定位**: "The First Open-Source Egocentric Dataset with a Complete Data-to-Model Toolchain, Collected Entirely from Consumer Smartphones"

## 目录结构约定

```
Open-AoE/
├── README.md          # 项目总览，对外可见
├── CLAUDE.md          # AI 协作规范（你正在读的文件）
├── STORY.md           # 行业叙事，保障代码/数据和叙事不脱钩
├── PROJECT.md         # 多人分工与里程碑
├── release/           # 对外发布物，代码质量要求最高
│   ├── open-aoe-2000h/
│   ├── aoe-visualization/
│   ├── aoe-retarget-replay/
│   └── aoe-training-ready/
├── report/            # 技术报告 LaTeX 源码
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── tables/
└── inner/             # 内部工作区，不对外发布
    ├── drafts/        # 草案、规划文档
    ├── research/      # 调研报告（3份deep-research报告在此）
    ├── sample_data/   # AoE 样例数据
    ├── prompts/       # AI 协作 prompt 历史
    └── references/    # 参考论文与资料
```

## AI 协作规则

### 写入文件前必须确认

1. **release/ 下的代码** — 需要可运行、有 docstring、有类型注解、有 README。这是对外发布的代码，质量标准最高。
2. **report/ 下的 LaTeX** — 需要可编译。修改前先阅读 `report/main.tex` 了解当前结构。
3. **inner/ 下的文档** — 内部文档，可以较自由地写，但需遵循命名约定：`YYYYMMDD_描述.md`。
4. **顶层 .md 文件** — 修改 STORY.md / PROJECT.md 前先阅读全文，追加而非覆盖。

### 命名约定

- 调研报告: `inner/research/ReportN_主题_YYYYMMDD/research_report_YYYYMMDD_slug.md`
- 草案: `inner/drafts/YYYYMMDD_描述.md`
- release 子项目: 各自维护独立的 `README.md` + `requirements.txt`（或 `pyproject.toml`）
- LaTeX sections: `report/sections/N_section_name.tex`

### 信息查找优先级

当需要了解项目信息时，按以下优先级查找：

1. **STORY.md** — 项目叙事、差异化定位、价值锚点
2. **PROJECT.md** — 当前分工、进度、阻塞项
3. **inner/drafts/** — 规划草案（`[草案] Open-AoE.md` 包含完整的开源计划矩阵）
4. **inner/research/** — 三份调研报告：
   - Report1: 开源数据集对比维度 + 技术报告规范 + Related Work
   - Report2: Human-to-Robot retarget 方法选型（GMR/dex-retargeting/SPIDER等）
   - Report3: 主流开源模型选型（π₀.₅/GR00T/RDT-1B/LeRobot等）
5. **AoE 论文原文** — `../Paper_Notes/AoE_CVPRW_2026/`

### 禁止事项

- **不要修改 inner/sample_data/ 中的数据文件**（这些是 AoE 采集的原始样例）
- **不要在 release/ 中放置草稿或实验性代码**（先在 inner/ 中验证）
- **不要在一次提交中同时大幅修改多个顶层 .md 文件**（逐个更新，避免冲突）
- **不要擅自更改 28D joint space 定义**: `[L_ARM(7), R_ARM(7), L_HAND(6), R_HAND(6), PAD(2)]`

## AoE 数据格式速查

每个 segment 目录结构:

```
raw_{collector_id}_seg_{segment_id}/
├── raw_video.mp4                        # 去畸变后的 ego-centric 视频
├── video_info.json                      # 设备信息、相机内参、分辨率、帧率
├── ego_annotation/
│   ├── ego_action_annotation.json       # 原子动作切分 + 双语 caption
│   └── ...
└── ego_process/
    ├── hands.npz                        # MANO 手部重建（HaWoR）
    ├── camera_traj.npz                  # 相机轨迹（MegaSAM）
    └── ...
```

## 关键技术参数

| 参数 | 值 |
|------|-----|
| 开源数据规模 | 2000 小时 |
| 采集设备 | 消费级智能手机（多机型，Android/iOS）|
| 单人成本 | < $20 |
| 手部标注 | MANO (HaWoR) + 21-joint keypoints |
| 动作标注 | VLM 自动切分（Qwen3-VL）+ 人工复审 |
| 语言标注 | 中英双语原子动作描述 |
| 目标 joint space | 28D: [L_ARM(7), R_ARM(7), L_HAND(6), R_HAND(6), PAD(2)] |
| 已验证真机 | Unitree G1 + Inspire 5-fingered hands |
| 已验证模型 | GR00T N1.5 + FLARE, fastWAM multihead |
| 数据格式中枢 | LeRobot v2.1 |

## 竞品速查

| 数据集 | 规模 | 设备 | 成本 | 工具链 |
|--------|------|------|------|--------|
| EgoDex (Apple) | 829H | Vision Pro | $3,500 | ✗ |
| EgoLive (JD) | 1,680H | 自研头戴 | 自研 | ✗ |
| OpenEgo (UT Dallas) | 1,107H | 混合 | N/A | ✗ |
| EgoScale (NVIDIA) | 20K+H | MANUS等 | 高 | ✗ |
| **Open-AoE** | **2,000H** | **手机** | **<$20** | **✓ 完整** |

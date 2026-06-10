# PROJECT.md — Open-AoE 多人分工与协作

> 本文件追踪 Open-AoE 项目的分工、进度和阻塞项。
> 更新原则：**谁完成了什么，就由谁更新对应行**。AI Agent 只更新自己参与的任务。

---

## 一、里程碑

| 里程碑 | 目标日期 | 交付物 | 状态 |
|--------|---------|--------|------|
| **M0: 脚手架搭建** | 2026-06-08 | 项目结构 + 调研报告 ×3 | ✅ 完成 |
| **M1: 数据筛选打包** | 2026-06-下旬 | 2000H 数据 + 质量门限定义 | 🔴 未开始 |
| **M2: 工具链 MVP** | 2026-07-上旬 | Visualization + Retarget-Replay + Training-Ready 各一个可运行 demo | 🔴 未开始 |
| **M3: Tech Report 初稿** | 2026-07-中旬 | 4 页短报告 LaTeX，含数据分布 + 实验 | 🔴 未开始 |
| **M4: WAIC 发布** | 2026-07-中下旬 | 全部交付物上线 GitHub + HuggingFace | 🔴 未开始 |
| **M5: 外滩大会迭代** | 2026-09-中旬 | 社区反馈修复 + 新模型适配 | 🔴 未开始 |

---

## 二、工作包分解（WBS）

### WP1: 数据筛选与打包（Open-AoE-2000H）

| 任务 | 负责人 | 状态 | 备注 |
|------|--------|------|------|
| 定义筛选门限（pred_valid 覆盖率 + IK failure rate + 相机轨迹连续性） | TBD | 🔴 | 参考 `inner/drafts/[草案] Open-AoE.md` |
| 执行数据筛选脚本 | TBD | 🔴 | — |
| 数据质量人工抽检（5% 采样） | TBD | 🔴 | — |
| 数据打包（MP4 + JSON + NPZ 格式） | TBD | 🔴 | — |
| 撰写数据集 README 和 DataCard | TBD | 🔴 | 参考 `inner/sample_data/AoE_dataset_README.md` |
| 上传 HuggingFace Hub | TBD | 🔴 | — |

### WP2: AoE-Visualization

| 任务 | 负责人 | 状态 | 备注 |
|------|--------|------|------|
| 技术选型（Rerun SDK） | TBD | 🔴 | — |
| 实现视频 + MANO 叠加可视化 | TBD | 🔴 | — |
| 实现相机轨迹 3D 可视化 | TBD | 🔴 | — |
| 实现原子动作 timeline 可视化 | TBD | 🔴 | — |
| 打包为 pip 可安装工具 | TBD | 🔴 | — |
| 撰写 README + demo 截图 | TBD | 🔴 | — |

### WP3: AoE-Retarget-Replay

| 任务 | 负责人 | 状态 | 备注 |
|------|--------|------|------|
| 沿用 `ant_aoe_retarget_to_g1.py` 臂部 retarget | TBD | 🟡 80% | 已有核心脚本 |
| 集成 dex-retargeting 做手指映射 | TBD | 🔴 | 需为 Inspire Hand 编写 URDF 配置 |
| `triple_play.py` MuJoCo 可视化产品化 | TBD | 🟡 | 已有核心脚本 |
| 批量处理能力（2000H 数据） | TBD | 🔴 | — |
| 真机 Replay 视频录制 | TBD | 🔴 | G1 实物演示 |
| 撰写 README + 使用教程 | TBD | 🔴 | — |

### WP4: AoE-Training-Ready

| 任务 | 负责人 | 状态 | 备注 |
|------|--------|------|------|
| 完善 `retarget_npz_to_lerobot.py`（LeRobot v2.1） | TBD | 🟡 60% | 核心格式转换器 |
| fastWAM training recipe 文档化 | TBD | 🟡 | 已验证 Loss 0.523 |
| π₀.₅ (OpenPI) 适配器 + recipe | TBD | 🔴 | 参考 Report3 |
| GR00T N1.7 适配器 + recipe | TBD | 🔴 | 已有 FLARE 验证基础 |
| RDT-1B 适配器 + recipe | TBD | 🔴 | 可选，双臂场景 |
| 撰写 README + 模型选择指南 | TBD | 🔴 | — |

### WP5: Tech Report

| 任务 | 负责人 | 状态 | 备注 |
|------|--------|------|------|
| LaTeX 模板搭建 | TBD | 🔴 | `report/main.tex` |
| Related Work 撰写 | TBD | 🟡 | Report1 已有初稿 |
| 数据分布可视化（场景/动作/时间/机型） | TBD | 🔴 | — |
| 数据集对比表格（Table 1 + Table 2） | TBD | 🟡 | Report1 已有设计方案 |
| 下游实验（Data Recipe 消融） | TBD | 🟡 | 已有 G1 实验数据 |
| 论文编译与校对 | TBD | 🔴 | — |

---

## 三、依赖关系

```
WP1(数据筛选) ──→ WP4(Training-Ready) ──→ WP5(Tech Report 实验部分)
      │                                         ↑
      └──→ WP2(Visualization) ──→ WP5(数据分布可视化)
      │
      └──→ WP3(Retarget-Replay) ──→ WP5(Retarget 验证实验)
```

**关键路径**: WP1 → WP4 → WP5（数据筛选 → 模型训练 → 实验报告）

---

## 四、协作约定

### 分支管理（当项目转为 Git 仓库后）

- `main`: 稳定版本，仅通过 PR 合入
- `dev/*`: 各人开发分支
- `release/*`: 发布准备分支

### 沟通渠道

- **日常同步**: 更新本文件对应任务行
- **技术决策**: 记录在 `inner/drafts/YYYYMMDD_决策_描述.md`
- **叙事调整**: 更新 `STORY.md` 第六节"叙事迭代日志"

### 质量标准

| 区域 | 标准 |
|------|------|
| `release/` 代码 | 有 README、有 docstring、有类型注解、可 pip install |
| `report/` LaTeX | 可 `pdflatex` 编译通过 |
| `inner/` 文档 | 文件名含日期，有标题和目的说明 |

---

## 五、风险登记

| 风险 | 影响 | 缓解措施 | Owner |
|------|------|---------|-------|
| 数据筛选门限定义不清 | 阻塞 WP1-WP5 全链路 | 尽快与数据团队对齐，先用 sample_data 验证 | TBD |
| Inspire Hand 无 URDF 配置 | WP3 手指 retarget 无法完成 | 联系硬件团队获取 URDF，或自行测量建模 | TBD |
| WAIC DDL 紧张（~6 周） | 可能无法完成全部工具链 | 优先级排序：fastWAM(已验证) > GR00T > π₀.₅ | TBD |
| 数据质量被社区质疑 | 口碑风险 | 多轮人工复审 + 真机 Replay 视频背书 | TBD |

---

## 六、变更日志

### 2026-06-08 项目初始化

- 搭建项目脚手架（README / CLAUDE / STORY / PROJECT）
- 完成三份 deep-research 调研报告
- 迁移已有文件到新目录结构
- 确定五个工作包（WP1-WP5）和关键路径

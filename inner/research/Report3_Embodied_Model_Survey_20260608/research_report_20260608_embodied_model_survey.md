# 具身智能主流开源模型调研报告：AoE-Training-Ready工具链方案

**研究日期**: 2026-06-08  
**研究模式**: Deep Research  
**研究目的**: 为AoE-Training-Ready工具链选型提供方案参考，评估可集成的主流开源模型

---

## 执行摘要

本报告系统调研了2024-2026年间具身智能领域的主流开源模型，从**VLA基础模型**、**扩散策略模型**、**数据格式与训练框架**三个维度出发，结合GitHub Stars和论文引用量评估各模型的社区影响力与集成价值。基于AoE数据的特点（ego-centric视频 + MANO手部 + 28D joint space + LeRobot v2.1格式），我们推荐AoE-Training-Ready工具链采用**分层集成策略**：以LeRobot为数据格式标准层，优先打通π₀.₅和GR00T N1.5/N1.7两大旗舰模型，同时提供Diffusion Policy/RDT-1B等经典模型的training recipe。

---

## 一、模型全景与影响力排名

### 1.1 GitHub Stars排名（截至2026-06-08）

| 排名 | 模型/框架 | Stars | 机构 | 类别 | AoE集成优先级 |
|------|-----------|-------|------|------|--------------|
| 1 | **LeRobot** | 24,781 | Hugging Face | 训练框架 | 🔴 基础设施 |
| 2 | **OpenPI (π₀.₅)** | 12,220 | Physical Intelligence | VLA模型 | 🔴 最高优先 |
| 3 | **Isaac-GR00T** | 7,280 | NVIDIA | VLA模型 | 🔴 最高优先 |
| 4 | **OpenVLA** | 6,381 | Stanford/Berkeley | VLA模型 | 🟠 高优先 |
| 5 | **Diffusion Policy** | 4,246 | Columbia/Stanford | 策略模型 | 🟠 高优先 |
| 6 | **ACT** | 1,986 | Stanford (ALOHA) | 策略模型 | 🟡 中优先 |
| 7 | **RDT-1B** | 1,717 | 清华大学 | 扩散基础模型 | 🟠 高优先 |
| 8 | **Octo** | 1,666 | Berkeley | 策略模型 | 🟡 中优先 |
| 9 | **3D-Diffusion-Policy** | 1,376 | Stanford | 策略模型 | 🟡 中优先 |
| 10 | **HPT** | 537 | MIT | 预训练框架 | ⚪ 参考 |
| 11 | **CogACT** | 430 | Microsoft | VLA模型 | ⚪ 参考 |

### 1.2 技术路线分类

**第一梯队：VLA旗舰模型（Vision-Language-Action）**
- π₀ / π₀.₅ / π₀.₆（Physical Intelligence）
- GR00T N1 / N1.5 / N1.7（NVIDIA）
- OpenVLA / OpenVLA v2（Stanford/Berkeley）

**第二梯队：扩散策略基础模型**
- Diffusion Policy（Columbia/Stanford）
- RDT-1B（清华大学）
- 3D-Diffusion-Policy（Stanford）

**第三梯队：轻量策略模型**
- ACT（Stanford ALOHA）
- Octo（Berkeley）
- CogACT（Microsoft）
- HPT（MIT）

**基础设施层**
- LeRobot（Hugging Face）— 数据格式 + 训练框架
- Open X-Embodiment — 大规模数据集标准

---

## 二、核心模型深度评估

### 2.1 π₀.₅ / OpenPI（Physical Intelligence）

| 指标 | 值 |
|------|-----|
| **GitHub Stars** | **12,220** ⭐ |
| **发表** | 2025 |
| **参数量** | ~3B（VLM backbone + Action Expert） |
| **代码** | github.com/Physical-Intelligence/openpi |

**架构概述**：
- **VLM Backbone**：基于Gemma模型 + 视觉编码器，继承互联网级别的语义知识
- **Action Expert**：+3亿参数的专用动作生成模块
- **输出机制**：Flow Matching（流匹配）生成高频动作序列，支持50Hz控制频率
- **训练框架**：支持异构数据co-training（不同机器人、不同任务）

**数据格式要求**：
- 支持LeRobot v2格式
- 支持RLDS格式（Open X-Embodiment）
- 需要：图像观测 + 语言指令 + 动作序列（joint positions/velocities）
- OpenPI已提供DROID全量数据训练指导

**与AoE数据的适配分析**：
- ✅ AoE的ego视频可作为图像观测
- ✅ AoE的双语caption可作为语言指令
- ✅ AoE的28D joint space可转换为动作序列
- ⚠️ 需要通过retarget将MANO手部转换为目标机器人的joint space
- ⚠️ π₀.₅的预训练主要在robot data上，需要adapter处理ego视角差异

**集成方案**：
```
AoE数据 → retarget_to_robot_joints → LeRobot v2格式 → OpenPI训练脚本
```
- 核心转换：`retarget_npz_to_lerobot.py`（已存在）→ 适配OpenPI的数据加载器
- 预计工作量：1-2周（主要是数据格式对齐和训练配置调参）
- **推荐优先级：🔴 最高优先**

---

### 2.2 GR00T N1 / N1.5 / N1.7（NVIDIA）

| 指标 | 值 |
|------|-----|
| **GitHub Stars** | **7,280** ⭐ |
| **发表** | GTC 2025 / 2026 |
| **参数量** | 2B（GR00T-N1-2B）|
| **代码** | github.com/NVIDIA/Isaac-GR00T |

**架构概述**：
- 全球首个开源人形机器人基础模型
- 支持多模态输入（视频 + 语言 + 本体感知）
- 跨本体架构设计（cross-embodiment）
- 版本演进：N1 → N1.5（架构+数据改进）→ N1.6 → N1.7（最新）

**数据格式要求**：
- 使用Isaac-GR00T自定义的数据格式
- 支持LeRobot格式转换
- 需要：多视角图像 + 语言指令 + 关节状态/动作
- 提供了详细的自定义数据集训练文档

**与AoE数据的适配分析**：
- ✅ AoE论文已在G1 + GR00T N1.5 + FLARE上验证（SR: 45%→95%）
- ✅ 数据格式转换路径已验证
- ✅ FLARE框架支持跨本体微调
- ⚠️ GR00T更偏向humanoid场景，需要适配ego视角
- ✅ N1.7最新版已开源，社区活跃

**集成方案**：
```
AoE数据 → retarget_to_g1_joints → Isaac-GR00T数据格式 → GR00T微调
```
- 核心依赖：`ant_aoe_retarget_to_g1.py`（已完成80%）+ FLARE框架
- 已验证：fastWAM multihead (Step 100, Loss 0.523)
- **推荐优先级：🔴 最高优先**（已有验证基础）

---

### 2.3 OpenVLA / OpenVLA v2（Stanford/Berkeley）

| 指标 | 值 |
|------|-----|
| **GitHub Stars** | **6,381** ⭐ |
| **发表** | CoRL 2024 |
| **参数量** | 7B |
| **代码** | github.com/openvla/openvla |

**架构概述**：
- 首个大规模开源VLA模型
- 基于Prismatic VLM + 动作token化
- 在Open X-Embodiment 970K轨迹上预训练
- 支持少量数据微调（few-hundred demonstrations）

**数据格式要求**：
- RLDS格式（TensorFlow Datasets）
- 需要：RGB图像 + 语言指令 + 离散化动作token
- 支持单臂/双臂多种动作空间

**与AoE数据的适配分析**：
- ✅ OpenVLA的7B规模使其具备强大的泛化能力
- ✅ 支持从少量数据微调
- ⚠️ 动作token化需要仔细设计（离散化精度 vs 连续控制）
- ⚠️ RLDS格式转换需要额外工作
- OpenVLA v2（2025）改进了动作表示和训练效率

**集成方案**：
```
AoE数据 → retarget_to_robot_joints → RLDS格式 → OpenVLA微调
```
- 预计工作量：2-3周
- **推荐优先级：🟠 高优先**

---

### 2.4 Diffusion Policy（Columbia/Stanford）

| 指标 | 值 |
|------|-----|
| **GitHub Stars** | **4,246** ⭐ |
| **发表** | RSS 2023 / IJRR 2024 |
| **代码** | github.com/real-stanford/diffusion_policy |

**架构概述**：
- 开创性地将扩散模型引入机器人策略学习
- 两种架构：CNN-based 和 Transformer-based
- 条件去噪扩散过程生成多步动作
- 在多个benchmark上达到SOTA

**数据格式要求**：
- 自定义Zarr格式
- 需要：观测图像 + 低维状态 + 动作序列
- 支持action chunking

**与AoE数据的适配分析**：
- ✅ Diffusion Policy是行业经典基线，广泛用于对比实验
- ✅ 数据格式简单，易于转换
- ✅ 在LeRobot中已集成Diffusion Policy训练
- ⚠️ 不支持语言条件（纯视觉策略）
- ⚠️ 需要per-task训练，无法利用AoE的多任务多样性

**集成方案**：通过LeRobot统一集成
- **推荐优先级：🟠 高优先**（作为baseline）

---

### 2.5 RDT-1B（清华大学）

| 指标 | 值 |
|------|-----|
| **GitHub Stars** | **1,717** ⭐ |
| **发表** | ICLR 2025 |
| **参数量** | 1.2B |
| **代码** | github.com/thu-ml/RoboticsDiffusionTransformer |

**架构概述**：
- 最大的扩散基础模型（1.2B参数）
- 在1M+多机器人episode上预训练
- 专为双臂操作设计
- 支持语言条件和多模态输入

**数据格式要求**：
- 自定义格式，提供了数据转换工具
- 支持HuggingFace Dataset格式
- 需要：多视角图像 + 语言指令 + 双臂关节动作

**与AoE数据的适配分析**：
- ✅ RDT-1B专为双臂操作设计，与AoE的双手数据匹配
- ✅ 1.2B参数的预训练基础提供了强泛化能力
- ✅ 支持语言条件，可利用AoE的caption
- ✅ H-RDT变体已实现从人类视频学习
- ⚠️ 数据格式需要定制转换

**集成方案**：
```
AoE数据 → retarget → RDT数据格式 → RDT-1B微调
```
- H-RDT已验证ego视频→robot policy的可行性
- **推荐优先级：🟠 高优先**（双臂场景优先选择）

---

### 2.6 ACT — Action Chunking with Transformers

| 指标 | 值 |
|------|-----|
| **GitHub Stars** | **1,986** ⭐ |
| **发表** | RSS 2023 |
| **代码** | github.com/tonyzhaozh/act |

**架构概述**：
- ALOHA系统的核心算法
- CVAE + Transformer架构
- Action Chunking：预测动作序列而非单步动作
- 轻量高效，适合低成本硬件

**与AoE数据的适配性**：
- ✅ ACT在LeRobot中已集成
- ✅ 轻量级，训练门槛低
- ⚠️ 不支持语言条件
- ⚠️ 主要针对桌面双臂场景

**推荐优先级：🟡 中优先**（通过LeRobot统一集成）

---

### 2.7 Octo — Open-Source Generalist Robot Policy

| 指标 | 值 |
|------|-----|
| **GitHub Stars** | **1,666** ⭐ |
| **发表** | RSS 2024 |
| **参数量** | 27M / 93M |
| **代码** | github.com/octo-models/octo |

**架构概述**：
- Transformer-based diffusion policy
- 在Open X-Embodiment 800K轨迹上预训练
- 支持灵活的任务和观测定义
- 设计为通用的策略初始化模型

**与AoE数据的适配性**：
- ✅ Octo使用OXE标准格式，数据转换有先例
- ✅ 轻量级（93M参数），训练成本低
- ⚠️ 性能已被π₀.₅和GR00T超越
- ⚠️ 社区活跃度下降

**推荐优先级：🟡 中优先**（作为轻量baseline）

---

### 2.8 LeRobot — 数据格式标准与训练框架

| 指标 | 值 |
|------|-----|
| **GitHub Stars** | **24,781** ⭐ |
| **发表** | 2024-2026持续更新 |
| **代码** | github.com/huggingface/lerobot |

**这不是一个模型，而是整个生态的基础设施。**

**核心能力**：
- **统一数据格式**：LeRobotDataset（Parquet + MP4/images），v0.4.0支持OXE级别（>400GB）
- **内置策略**：ACT、Diffusion Policy、TDMPC、VQ-BeT等
- **训练Pipeline**：端到端的训练+评估+部署
- **HuggingFace Hub集成**：数据集托管和共享
- **跨机器人支持**：ALOHA、Koch、WidowX、Franka等

**与AoE的核心关系**：
- ✅ AoE已有`retarget_npz_to_lerobot.py`（LeRobot v2.1格式转换器）
- ✅ LeRobot是π₀.₅/OpenPI和GR00T的数据格式桥梁
- ✅ 上传到HuggingFace Hub即可被全球社区使用
- ✅ LeRobot v0.4.0的Chunked Episodes格式支持大规模数据

**定位**：AoE-Training-Ready的**数据格式标准层**，所有模型的数据转换都应以LeRobot格式为中转。

---

### 2.9 fastWAM（已验证）

| 指标 | 值 |
|------|-----|
| **验证状态** | ✅ 已验证（Step 100, Loss 0.523）|
| **格式** | LeRobot v2.1 |

**当前状态**：AoE已通过fastWAM multihead验证了数据→模型训练的完整流程。这是AoE-Training-Ready工具链的第一个成功案例，证明了28D joint space + LeRobot格式的可行性。

---

## 三、集成架构与路线图

### 3.1 AoE-Training-Ready分层架构

```
┌─────────────────────────────────────────────────────────┐
│              AoE-Training-Ready 工具链                     │
├─────────────────────────────────────────────────────────┤
│  Layer 3: 模型训练Recipes                                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐│
│  │ π₀.₅      │ │ GR00T     │ │ RDT-1B    │ │ DP/ACT  ││
│  │ Recipe    │ │ N1.7      │ │ Recipe    │ │ Recipe  ││
│  │           │ │ Recipe    │ │           │ │         ││
│  └───────────┘ └───────────┘ └───────────┘ └─────────┘│
├─────────────────────────────────────────────────────────┤
│  Layer 2: 模型格式适配器                                    │
│  ┌─────────────────┐  ┌──────────────────────────┐     │
│  │ lerobot_to_openpi│  │ lerobot_to_groot          │     │
│  │ (OpenPI适配)     │  │ (Isaac-GR00T适配)         │     │
│  └─────────────────┘  └──────────────────────────┘     │
│  ┌─────────────────┐  ┌──────────────────────────┐     │
│  │ lerobot_to_rlds  │  │ lerobot_to_rdt            │     │
│  │ (OpenVLA适配)    │  │ (RDT-1B适配)             │     │
│  └─────────────────┘  └──────────────────────────┘     │
├─────────────────────────────────────────────────────────┤
│  Layer 1: LeRobot数据格式标准层                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  retarget_npz_to_lerobot.py (已完成60%)           │   │
│  │  AoE NPZ → LeRobot v2.1 Dataset                  │   │
│  │  (Parquet + MP4, 28D joint space)                 │   │
│  └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Layer 0: AoE原始数据                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  MP4 + ego_action_annotation.json + hands.npz    │   │
│  │  + camera_traj.npz + video_info.json              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 集成优先级路线图

**Phase 1（MVP，1-2周）— 已验证模型完善**
- ✅ 完善`retarget_npz_to_lerobot.py`（当前完成度60%）
- ✅ 发布fastWAM训练recipe和训练日志
- ✅ 上传示例数据到HuggingFace Hub
- 交付物：`aoe_to_fastwam_recipe.py` + 训练配置 + README

**Phase 2（核心，2-4周）— 旗舰模型打通**
- 🔴 打通π₀.₅：编写`lerobot_to_openpi.py`适配器 + OpenPI微调recipe
- 🔴 打通GR00T N1.7：编写`lerobot_to_groot.py`适配器 + Isaac-GR00T微调recipe
- 交付物：每个模型的数据转换脚本 + 训练配置 + 训练日志 + README

**Phase 3（扩展，4-6周）— 更多模型支持**
- 🟠 打通RDT-1B：利用H-RDT的ego视频学习范式
- 🟠 打通Diffusion Policy / ACT：通过LeRobot内置支持
- 🟡 打通OpenVLA：RLDS格式转换
- 交付物：各模型的training recipe

**Phase 4（长期）— 持续跟进爆款模型**
- 建立模型适配CI/CD：新模型发布后2周内提供AoE training recipe
- 社区贡献机制：接受PR添加新模型支持
- 性能对比dashboard：不同模型在AoE数据上的训练效果对比

### 3.3 关键技术决策

**决策1：数据格式中枢选择**

| 方案 | 优势 | 劣势 |
|------|------|------|
| **LeRobot格式（推荐）** | 24.8K stars，行业标准，HuggingFace生态 | 需为每个模型写转换器 |
| 自定义Zarr格式 | EgoVerse采用，灵活 | 社区不通用 |
| RLDS格式 | OXE标准 | TF依赖重，社区收窄 |

**建议**：以LeRobot v2.1作为数据格式中枢，所有模型的适配都从LeRobot格式出发。

**决策2：首批打通模型优先级**

| 模型 | 理由 | 难度 |
|------|------|------|
| **π₀.₅** | 12.2K stars，行业最热，OpenPI已有DROID训练示例 | ★★★☆☆ |
| **GR00T N1.7** | 7.3K stars，AoE论文已验证，NVIDIA生态 | ★★☆☆☆ |
| **fastWAM** | 已验证，Loss 0.523 | ★☆☆☆☆（已完成）|
| **RDT-1B** | 1.7K stars，双臂专用，H-RDT支持ego视频 | ★★★☆☆ |

**决策3：ego视角到robot视角的处理策略**

| 方案 | 说明 | 适用模型 |
|------|------|---------|
| **动作空间retarget** | 仅转换动作序列，图像保持ego视角 | π₀.₅, GR00T |
| **图像域迁移** | Masquerade式的robot inpainting | VLA预训练 |
| **双流输入** | ego图像+retarget动作同时输入 | H-RDT, EgoMimic |

**建议**：Phase 2采用"动作空间retarget"方案（最简单），后续根据实验结果决定是否引入图像域迁移。

---

## 四、各模型数据需求速查表

| 模型 | 图像格式 | 动作格式 | 语言 | 本体感知 | 推荐数据量 |
|------|---------|---------|------|---------|-----------|
| π₀.₅ | RGB (224×224) | 连续joint pos/vel | ✓ | ✓ | 100-1000 episodes |
| GR00T N1.7 | RGB (多视角) | 连续joint pos | ✓ | ✓ | 50-500 episodes |
| OpenVLA | RGB (224×224) | 离散token | ✓ | ✗ | 100-500 episodes |
| RDT-1B | RGB (多视角) | 连续joint pos | ✓ | ✓ | 50-200 episodes |
| Diffusion Policy | RGB (任意) | 连续joint pos | ✗ | ✓ | 50-200 episodes |
| ACT | RGB (多视角) | 连续joint pos | ✗ | ✓ | 50-200 episodes |
| fastWAM | RGB | 连续28D joint | ✗ | ✓ | 100+ episodes |

**AoE数据覆盖分析**：
- ✅ RGB图像：ego视频提供
- ✅ 语言指令：双语caption提供
- ✅ 连续动作：28D joint space（通过retarget）
- ⚠️ 多视角：仅单一ego视角（可通过数据增强补充）
- ⚠️ 本体感知：需从retarget结果推导

---

## 五、竞品工具链对比

### 5.1 EgoVerse（GaTech-RL2）

EgoVerse是目前最接近AoE-Training-Ready定位的竞品：
- **七层架构**：Data Upload → Storage → Ingestion → Format → Transform → Loading → Training
- **支持模型**：HPT, ACT, Pi0.5, EgoBridge
- **数据格式**：自定义Zarr v3
- **规模**：250+分支，40+贡献者

**与AoE-Training-Ready的差异**：
- EgoVerse偏向研究平台，AoE偏向工具链
- EgoVerse使用自定义Zarr，AoE使用行业标准LeRobot
- EgoVerse已支持Pi0.5（通过OpenPi子模块），AoE应参考其集成方式
- AoE的优势：更低的使用门槛，更聚焦的工具链定位

### 5.2 Any4LeRobot（社区工具集）

github.com/Tavish9/any4lerobot — LeRobot的社区工具集合：
- 数据转换脚本
- 预处理工具
- 训练workflow helpers
- **可参考其数据转换的实现模式**

### 5.3 Dexbotic（开源VLA工具箱）

dexbotic.com — 开源VLA模型工具箱：
- 提供统一的VLA研究框架
- 支持多种模型的训练和评估
- **可参考其模型统一接口设计**

---

## 六、对外发布策略

### 6.1 发布物清单

| 组件 | 形态 | 说明 |
|------|------|------|
| `retarget_npz_to_lerobot.py` | Python脚本 | AoE NPZ → LeRobot v2.1格式 |
| `aoe_to_openpi_recipe/` | 目录 | π₀.₅微调配置+示例 |
| `aoe_to_groot_recipe/` | 目录 | GR00T N1.7微调配置+示例 |
| `aoe_to_fastwam_recipe/` | 目录 | fastWAM训练配置+示例（已验证）|
| `aoe_to_rdt_recipe/` | 目录 | RDT-1B微调配置+示例 |
| `benchmark_results/` | 目录 | 各模型在AoE数据上的训练结果 |
| HuggingFace Dataset | 在线 | Open-AoE LeRobot格式数据集 |
| README.md | 文档 | 快速开始+模型选择指南 |

### 6.2 社区运营策略

1. **首发效应**：WAIC发布时优先展示π₀.₅和GR00T的训练结果
2. **持续跟进**：建立"新模型适配"的社区贡献机制
3. **性能Dashboard**：公开各模型的训练loss曲线和真机评估结果
4. **GitHub Issue模板**：方便社区提交新模型适配请求

---

## 七、风险与局限

1. **模型API变动**：开源模型频繁更新，需要持续维护适配器
2. **计算资源**：π₀.₅和GR00T的微调需要GPU集群，可能限制社区使用
3. **ego-robot域差异**：ego视角数据直接训练robot policy的效果存在不确定性
4. **评估标准缺失**：缺乏统一的ego数据→robot policy的评估benchmark
5. **数据规模权衡**：2000H数据是否足够体现scaling效果？

---

## 参考文献

[1] Physical Intelligence. "π₀.₅: A VLA Model with Open-World Generalization." 2025. github.com/Physical-Intelligence/openpi (12,220 ⭐)

[2] NVIDIA. "Isaac GR00T N1.7: Open Foundation Model for Humanoid Robots." GTC 2025/2026. github.com/NVIDIA/Isaac-GR00T (7,280 ⭐)

[3] Kim et al. "OpenVLA: An Open-Source Vision-Language-Action Model." CoRL 2024. github.com/openvla/openvla (6,381 ⭐)

[4] Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023. github.com/real-stanford/diffusion_policy (4,246 ⭐)

[5] Zhao et al. "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)." RSS 2023. github.com/tonyzhaozh/act (1,986 ⭐)

[6] Liu et al. "RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation." ICLR 2025. github.com/thu-ml/RoboticsDiffusionTransformer (1,717 ⭐)

[7] Octo Model Team. "Octo: An Open-Source Generalist Robot Policy." RSS 2024. github.com/octo-models/octo (1,666 ⭐)

[8] Hugging Face. "LeRobot: End-to-End Robot Learning Framework." 2024-2026. github.com/huggingface/lerobot (24,781 ⭐)

[9] Ze et al. "3D-Diffusion-Policy (DP3)." github.com/YanjieZe/3D-Diffusion-Policy (1,376 ⭐)

[10] Wang et al. "HPT: Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers." NeurIPS 2024. github.com/liruiw/HPT (537 ⭐)

[11] Microsoft. "CogACT: A Foundational VLA Model." 2025. github.com/Microsoft/CogACT (430 ⭐)

[12] EgoVerse Team. "EgoVerse: A Multi-Embodiment Learning Platform." GaTech-RL2, 2026. github.com/GaTech-RL2/EgoVerse

[13] Bi et al. "H-RDT: Human Manipulation Enhanced Bimanual Robotic Diffusion Transformer." 2025.

[14] Zheng et al. "FLARE: Achieving Masterful Robot Policies with Large-Scale RL." 2025.

[15] Tavish. "Any4LeRobot: A Tool Collection for LeRobot." github.com/Tavish9/any4lerobot

[16] VLA Training Data Guide. Claru, 2026. claru.ai/vla-training-data-guide

[17] Feng et al. "From Human Videos to Robot Manipulation: A Survey." IJCAI 2026.

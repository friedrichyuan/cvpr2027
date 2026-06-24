# AoE Training-Ready: GR00T N1.7 Recipe

基于 AoE 手机采集的 egocentric 手部数据，完整实现从原始数据到 GR00T N1.7 预训练的自包含工具链。无需依赖外部 Isaac-GR00T 仓库。

```
AoE 原始数据                          训练好的模型
(MANO 参数 + ego 视频)                (GR00T N1.7 checkpoint)
         │                                    ▲
         ▼                                    │
 ┌───────────────────────────────────────────────────────────┐
 │  Step 0        Step 1         Step 2       Step 3         │
 │  MANO FK  ──► Retarget  ──► LeRobot  ──► 预训练          │
 │  (21关键点)   (Sharpa/夹爪)   (V2.1)      (GR00T N1.7)    │
 └───────────────────────────────────────────────────────────┘
```


## 环境要求

### 数据与模型

| 项目 | 说明 | 获取方式 |
|------|------|---------|
| AoE 数据集 | `poc_deliver/` 目录，包含所有 episode 原始数据 | 内部: 参见 PROJECT.md |
| GR00T-N1.7-3B | 基础模型权重 | 下载到 `/path/to/GR00T-N1.7-3B` |
| Cosmos-Reason2-2B | VLM 骨干网络 (Qwen3-VL)；提供 tokenizer 和图像处理器 | `huggingface-cli download nvidia/Cosmos-Reason2-2B`，或通过 `--vlm-model-name` 指定本地路径 |
| MANO 模型 | `MANO_LEFT.pkl` + `MANO_RIGHT.pkl` | 运行 `assets/mano/download_mano.sh` |
| Sharpa URDF | 机器人手描述文件（URDF + MJCF + STL 网格） | 已包含在 `scripts/urdf/sharpa-urdf-usd-xml/` |

> **关于 Cosmos-Reason2-2B**: GR00T N1.7 使用 `nvidia/Cosmos-Reason2-2B`（Qwen3-VL 架构）作为视觉语言骨干网络。训练代码启动时会加载该模型的 tokenizer 和图像处理器。首次运行将从 HuggingFace Hub 自动下载，也可预下载后通过 `--vlm-model-name` 指定本地路径：
> ```bash
> huggingface-cli download nvidia/Cosmos-Reason2-2B --local-dir /path/to/Cosmos-Reason2-2B
> # 然后: --vlm-model-name /path/to/Cosmos-Reason2-2B
> ```


## 快速开始：端到端流程

本节带你走完完整的流程——从 AoE 原始数据到启动训练。

### Step 0: 安装

```bash
cd release/aoe-training-ready/gr00t_n1d7
uv sync --python 3.10
source .venv/bin/activate

# 验证关键依赖
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import smplx; print('smplx OK')"
python -c "import casadi; print('CasADi OK')"
```

### Step 1: 数据转换

选择一种模式：**sharpa**（62D，用于灵巧手）或 **gripper**（20D，用于平行夹爪）。

#### Sharpa 模式（推荐）：

```bash
DATA_ROOT=/path/to/poc_deliver
OUTPUT=output

# 1a. MANO FK → 21 关键点（111 个 episode 约 2 分钟）
python scripts/convert_mano_to_keypoints.py \
    --data-root $DATA_ROOT \
    --output-dir $OUTPUT

# 1b. 关键点 → Sharpa 22-DoF 关节角（约 5 分钟，CasADi+IPOPT 优化）
python scripts/retarget_to_sharpa.py \
    --data-root $DATA_ROOT \
    --keypoints-dir $OUTPUT \
    --output-dir $OUTPUT

# 1c. 组装为 LeRobot V2.1 格式（约 1 分钟）
python scripts/convert_ego_to_lerobot.py --mode sharpa \
    --data-root $DATA_ROOT \
    --retarget-dir $OUTPUT \
    --output-dir $OUTPUT
```

#### Gripper 模式（简化替代方案）：

```bash
DATA_ROOT=/path/to/poc_deliver
OUTPUT=output

python scripts/convert_mano_to_keypoints.py \
    --data-root $DATA_ROOT --output-dir $OUTPUT

python scripts/retarget_to_gripper.py \
    --data-root $DATA_ROOT \
    --keypoints-dir $OUTPUT --output-dir $OUTPUT

python scripts/convert_ego_to_lerobot.py --mode gripper \
    --data-root $DATA_ROOT \
    --retarget-dir $OUTPUT --output-dir $OUTPUT
```

**Step 1 预期输出**（每个 episode）:
```
output/<episode>/
├── hands_keypoints.npz            # (2, T, 21, 3) 世界坐标 + 相机坐标关键点
├── hands_retargeted_sharpa.npz    # 每只手 (T, 22) 关节角
└── ego_sharpa_lerobotv21/         # LeRobot V2.1 数据集
    ├── data/chunk-000/episode_000000.parquet
    ├── videos/chunk-000/observation.images.ego_view/episode_000000.mp4
    └── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
```

### Step 2: 合并数据集

将各 episode 独立数据集合并为统一训练集：

```bash
python scripts/merge_lerobot_datasets.py \
    --input-dir $OUTPUT \
    --mode sharpa \
    --output-dir $OUTPUT/ego_sharpa_merged
```

**预期输出**:
```
output/ego_sharpa_merged/
├── data/chunk-000/episode_000000.parquet ... episode_000110.parquet
├── videos/chunk-000/observation.images.ego_view/episode_*.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
```

### Step 3: 验证数据（可选但推荐）

```bash
python scripts/validate_dataset.py $OUTPUT/ego_sharpa_merged
```

预期：所有检查通过，打印数据集摘要（episode 数、帧数、维度等）。

### Step 4: 启动训练

```bash
bash scripts/launch_pretrain.sh \
    --mode sharpa \
    --base-model /path/to/GR00T-N1.7-3B \
    --dataset $OUTPUT/ego_sharpa_merged \
    --vlm-model-name /path/to/Cosmos-Reason2-2B \
    --max-steps 1000 \
    --batch-size 32 \
    --lr 1e-4
```

**预期输出**:
```
output/sharpa_train_YYYYMMDD_HHMMSS/
├── train.log
├── config.json
├── checkpoint-*/
│   ├── model*.safetensors
│   ├── optimizer.pt
│   └── trainer_state.json
└── wandb_config.json（如启用 WandB）
```


## 训练配置

本 Recipe 对应 **EgoScale Stage I: 大规模人类数据预训练** —— 全部模型参数参与训练（VLM backbone + 视觉编码器 + Projector + DiT 动作头）。模型通过 flow-matching 动作预测从 egocentric 人手数据中学习通用操作先验。

### launch_pretrain.sh 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `sharpa` | `sharpa`（62D）或 `gripper`（20D）|
| `--base-model` | （必填）| GR00T-N1.7-3B 权重路径 |
| `--dataset` | （必填）| 合并后的 LeRobot 数据集路径 |
| `--vlm-model-name` | `nvidia/Cosmos-Reason2-2B` | VLM 骨干网络的 HuggingFace 模型 ID 或本地路径（tokenizer + 图像处理器）。使用本地路径可避免每次运行从 HuggingFace Hub 下载。 |
| `--output` | 自动生成 | checkpoint 输出目录 |
| `--max-steps` | 1000 | 总训练步数 |
| `--batch-size` | 32 | 全局 batch size（所有 GPU 合计）|
| `--lr` | 1e-4 | 学习率 |
| `--num-gpus` | 1 | GPU 数量（>1 时自动使用 torchrun）|
| `--deepspeed-stage` | 3 | DeepSpeed ZeRO 阶段。2=优化器分片, 3=参数+优化器+梯度全分片。消费级 GPU（如 RTX 4090）推荐 3。|
| `--no-gradient-checkpointing` | （默认开启）| 禁用梯度检查点。默认开启以节省显存（用计算换显存）。|
| `--save-steps` | （无）| 每 N 步保存一次 checkpoint |
| `--skip-weight-loading` | false | 从头训练（忽略基础模型权重）|

### 模式对比

| | Sharpa (62D) | Gripper (20D) |
|---|---|---|
| State 维度 | 62 | 20 |
| 手部表示 | 22-DoF Sharpa Wave 关节角 | 1D 开合标量 |
| Retarget 方法 | CasADi + IPOPT 非线性优化 | 拇指-食指距离归一化 |
| 使用场景 | 部署到 Sharpa Wave 灵巧手 | 部署到平行夹爪 |
| 预训练 Projector | 有（复用 GR00T index 26）| 无（需从头训练新 projector）|
| 推荐训练步数 | 500-1000（微调）| 2000+（从头训练）|

**State 布局**:
```
sharpa  (62D): [left_wrist_eef(9), right_wrist_eef(9), left_hand_joints(22), right_hand_joints(22)]
gripper (20D): [left_wrist_eef(9), right_wrist_eef(9), left_gripper(1), right_gripper(1)]
```

### 多 GPU 训练

```bash
bash scripts/launch_pretrain.sh \
    --mode sharpa \
    --base-model /path/to/GR00T-N1.7-3B \
    --dataset output/ego_sharpa_merged \
    --vlm-model-name /path/to/Cosmos-Reason2-2B \
    --num-gpus 4 \
    --batch-size 64 \
    --max-steps 2000
```

默认使用 `torchrun` + DeepSpeed ZeRO-3 + 梯度检查点。Batch size 会自动对齐 GPU 数量。

### WandB 集成

要启用 Weights & Biases 日志，通过额外参数移除 `--no-use-wandb` 标记：

```bash
bash scripts/launch_pretrain.sh \
    --mode sharpa \
    --base-model /path/to/GR00T-N1.7-3B \
    --dataset output/ego_sharpa_merged \
    -- --use-wandb --wandb-project aoe-gr00t
```

### 关键训练超参数（训练代码内）

| 参数 | 值 | 说明 |
|------|-----|------|
| action_horizon | 16 | 动作预测窗口（步数）|
| state_dropout_prob | 0.8 | 高 dropout → 模型更多依赖视觉+语言 |
| tune_llm | true | 预训练: 解冻 VLM backbone (Qwen3-VL) |
| tune_visual | true | 预训练: 解冻视觉编码器 (ViT) |
| tune_projector | true | 预训练: 解冻多模态 projector |
| tune_diffusion_model | true | 预训练: 解冻 DiT 动作头 |
| DeepSpeed | ZeRO-3 | 参数+优化器+梯度全分片 |
| gradient_checkpointing | true | 梯度检查点（用计算换显存）|

> **注意**: 四个 `tune_*` 标记全部开启 —— 这是全参数预训练（EgoScale Stage I），而非微调。下游微调（Stage III）时，应冻结 VLM+ViT，仅训练 projector+DiT。


## 数据转换详解

### MANO FK → 21 关键点（Step 0）

**脚本**: `scripts/convert_mano_to_keypoints.py`

将 MANO 参数化表示 → 显式 3D 关节位置，使用 `smplx.MANOLayer` 前向运动学：

```
MANO 参数 (pred_rot, pred_trans, pred_hand_pose, pred_betas)
    → smplx.MANOLayer (Linear Blend Skinning)
    → 778 mesh 顶点 + 21 关节位置
    → mano_to_openpose 索引映射
    → joints_world (2, T, 21, 3)
```

指尖关键点（索引 4, 8, 12, 16, 20）来自 mesh 特定顶点，而非 FK 链末端——这是 MANO 标准做法，精度受限于 mesh 重建质量（典型误差 5-10mm）。

**21 关键点顺序（OpenPose Hand）**:
```
 [0]     手腕 (Wrist)
 [1-4]   拇指: CMC → MCP → IP → TIP
 [5-8]   食指: MCP → PIP → DIP → TIP
 [9-12]  中指: MCP → PIP → DIP → TIP
[13-16]  无名指: MCP → PIP → DIP → TIP
[17-20]  小指: MCP → PIP → DIP → TIP
```

关节缩写：CMC=腕掌关节, MCP=掌指关节, PIP=近端指间关节, DIP=远端指间关节, IP=指间关节（仅拇指）, TIP=指尖（mesh 顶点）。

**输出**: `output/<episode>/hands_keypoints.npz`

| 字段 | Shape | 说明 |
|------|-------|------|
| `joints_world` | (2, T, 21, 3) | 21 关键点世界坐标 |
| `joints_cam` | (2, T, 21, 3) | 21 关键点相机坐标 |
| `pred_valid` | (2, T) | 有效帧标记 (1=有效, 0=手未检测到) |

### Retarget → Sharpa Wave 22-DoF（Step 1a）

**脚本**: `scripts/retarget_to_sharpa.py`

求解 Sharpa Wave 灵巧手的 22 个关节角度，使机器人指尖位置匹配人手指尖。使用 CasADi 符号框架 + IPOPT 非线性求解器。

**逐帧算法**：
```
输入: keypoints[t] (21, 3) 世界坐标

A. 提取 5 指尖相对手腕的位置
   tips = keypoints[[4,8,12,16,20]] - keypoints[0]    # (5, 3)

B. 构建手掌局部坐标系（对齐 Sharpa rest pose）
   Z轴 = normalize(middle_MCP - wrist)       # 手指方向
   Y轴 = normalize(ring_MCP - index_MCP)     # 侧向（左右手镜像）
   X轴 = Y × Z                               # 掌心法线
   R_hand = [X | Y | Z]

C. 旋转指尖到 Sharpa 局部坐标系
   tips_local = R_hand.T @ tips.T

D. 非线性优化 (CasADi + IPOPT)
   目标函数: min ||FK(q) - tips_local||²
   约束: q_min ≤ q ≤ q_max（关节限位）
   初始猜测: q_prev（warm start）
```

**为什么需要 Step B（坐标系对齐）？** Sharpa FK 计算的指尖位置在手掌自身的局部坐标系中（rest pose 下手指沿 +Z）。如果直接用世界坐标下的指尖位置作为目标，当人手旋转时（如掌心朝下），指尖方向不再沿 +Z，solver 无法匹配，导致巨大误差。

**左右手差异**: 左右手的 Sharpa URDF 有 Y 轴镜像差异。`_build_hand_frame` 对右手取 `index_MCP - ring_MCP`（与左手相反）以匹配各自 URDF 的侧向约定。

**Sharpa Wave 22-DoF 关节顺序**:
```
拇指  (5): CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP
食指  (4): MCP_FE, MCP_AA, PIP, DIP
中指  (4): MCP_FE, MCP_AA, PIP, DIP
无名指(4): MCP_FE, MCP_AA, PIP, DIP
小指  (5): CMC, MCP_FE, MCP_AA, PIP, DIP
```

FE=屈/伸 (Flexion/Extension), AA=展/收 (Abduction/Adduction)。

**Retarget 精度**:

| 指标 | 左手 | 右手 |
|------|------|------|
| 平均误差 | 4.4 mm | 2.2 mm |
| P95 误差 | 25.5 mm | 6.7 mm |

**输出**: `output/<episode>/hands_retargeted_sharpa.npz`

| 字段 | Shape | 说明 |
|------|-------|------|
| `left_sharpa_joints` | (T, 22) | 左手 22-DoF 关节角 (rad) |
| `right_sharpa_joints` | (T, 22) | 右手 22-DoF 关节角 (rad) |
| `pred_valid` | (2, T) | 有效帧标记 |

### Retarget → Gripper 开合度（Step 1b）

**脚本**: `scripts/retarget_to_gripper.py`

计算拇指尖与食指尖的 3D 距离，除以最大开合距离（几何计算），归一化到 [0, 1]：

```
gripper = ||thumb_tip - index_tip|| / sqrt(thumb_length² + index_length²)
    thumb_length = ||kp[4] - kp[1]||   (拇指 CMC → tip)
    index_length = ||kp[8] - kp[5]||   (食指 MCP → tip)
```

- 0.0 = 捏合（拇指接触食指）
- 1.0 = 全开（约 90° 张开角）

**输出**: `output/<episode>/hands_retargeted_gripper.npz`

| 字段 | Shape | 说明 |
|------|-------|------|
| `left_gripper` | (T, 1) | 左手开合度 [0, 1] |
| `right_gripper` | (T, 1) | 右手开合度 [0, 1] |
| `pred_valid` | (2, T) | 有效帧标记 |

### Wrist EEF 9D（两种模式共享）

从 `hands.npz` 的 `pred_rot` / `pred_trans` 直接提取：

```
EEF 9D = [x, y, z, rot6d(6)]
    rot6d = rotation_matrix[:2, :].flatten()  (前两行展平)
```

### LeRobot V2 转换（Step 2）

**脚本**: `scripts/convert_ego_to_lerobot.py`

将处理后的数据转换为 GR00T N1.7 数据加载器兼容的 LeRobot V2.1 格式：
- 按 `pred_valid` 掩码将 episode 切分为连续有效段
- 视频从 30fps 降采样到 15fps
- 计算 action 为相邻 state 的差值
- 应用 action chunking（horizon=16 步）
- 写入每 episode 独立数据集：parquet + 视频符号链接 + 元数据 JSON


## AoE 数据格式参考

每个 AoE episode 目录结构：

```
raw_{collector_id}_seg_{segment_id}/
├── ego_process/ego_hands_reconstruction/
│   └── hands.npz               ← MANO 手部重建参数（源数据）
├── ego_process/ego_undistorted_video/
│   └── raw_video_undistorted.mp4
└── ego_annotation/
    └── ego_action_annotation.json
```

### hands.npz 字段

| 字段 | Shape | 说明 |
|------|-------|------|
| `pred_rot` | (2, T, 3) | 手腕全局旋转 (axis-angle, 世界坐标系), [0]=左 [1]=右 |
| `pred_trans` | (2, T, 3) | 手腕位置 (米, 世界坐标系) |
| `pred_hand_pose` | (2, T, 45) | 15 个手指关节旋转 (axis-angle, 15×3=45D) |
| `pred_betas` | (2, T, 10) | MANO 形状参数 (手型大小) |
| `pred_valid` | (2, T) | 有效帧标记 (1=有效, 0=手未检测到) |
| `R_c2w` / `t_c2w` | (T,3,3) / (T,3) | 相机→世界坐标变换 |
| `R_w2c` / `t_w2c` | (T,3,3) / (T,3) | 世界→相机坐标变换 |
| `focal` | scalar | 相机焦距 (pixels) |

### 数据集统计（poc_deliver）

| 指标 | 值 |
|------|-----|
| Episode 总数 | 111 |
| 总帧数 | 317,553 |
| 总时长 | 2.94 小时 (30fps) |
| 单集时长 | 18.5s ~ 255.6s（均值 95.4s）|
| 左手有效率 | 98.05% |
| 右手有效率 | 96.77% |
| 场景数 | 18 种 |
| 动作动词 | 97 种 |
| 标注段总数 | 1,338 |


## 可视化验证

### Sharpa AR Overlay（MuJoCo mesh + 关键点叠加到 ego 视频）

```bash
python scripts/visualize_sharpa_overlay_3d.py \
    --data-root /path/to/poc_deliver \
    --keypoints-dir output --retarget-dir output \
    --output-dir output --max-episodes 2
```

### Gripper Overlay（拇指-食指连线 + 开合度数值）

```bash
python scripts/visualize_gripper_overlay.py \
    --data-root /path/to/poc_deliver \
    --keypoints-dir output --retarget-dir output \
    --output-dir output --max-episodes 2
```

输出视频保存在 `output/<episode>/vis_retargeted_*_overlay.mp4`。


## 常见问题

### CasADi / IPOPT 安装失败

CasADi 在多数平台上自带 IPOPT。如果 `pip install casadi` 失败：
```bash
conda install -c conda-forge casadi   # 通过 conda 安装
```

### 训练时 CUDA 显存不足

- 减小 `--batch-size`（尝试 8 或 4）
- 使用多 GPU：`--num-gpus 2`（或更多）
- 使用 ZeRO-3（默认已开启）：`--deepspeed-stage 3`
- 确保梯度检查点开启（默认已开启）：不要传 `--no-gradient-checkpointing`

### "ERROR: Dataset meta/ not found"

忘记了合并步骤。运行：
```bash
python scripts/merge_lerobot_datasets.py \
    --input-dir output --mode sharpa --output-dir output/ego_sharpa_merged
```

### 移动数据后视频符号链接失效

合并脚本创建的是绝对路径符号链接。如果移动了数据集，重新运行合并，或将符号链接转换为相对路径：
```bash
find output/ego_sharpa_merged/videos -type l -exec sh -c \
    'target=$(readlink "$1"); ln -sf "$(realpath --relative-to="$(dirname "$1")" "$target")" "$1"' _ {} \;
```

### smplx / MANO 模型未找到

确保 `scripts/mano_models/` 包含 `MANO_LEFT.pkl` 和 `MANO_RIGHT.pkl`，或从仓库根目录运行 `assets/mano/download_mano.sh` 设置符号链接。这些模型来自 [MANO 项目](https://mano.is.tue.mpg.de/)（需要注册）。

### Sharpa URDF 找不到

确认 `scripts/urdf/sharpa-urdf-usd-xml/wave_01/` 目录存在，且包含 `left_sharpa_wave/` 和 `right_sharpa_wave/` 子目录（内含 `.urdf` 和 `meshes/`）。这些文件已预置在仓库中，无需额外下载。


## 目录结构

```
gr00t_n1d7_recipe/
├── README.md                                   # 英文版
├── README-zh.md                                # 本文件（中文版）
├── pyproject.toml                              # 依赖与包配置
├── configs/
│   ├── human_ego_config.py                     # 模式配置 (gripper/sharpa)
│   └── register_aoe_modality.py                # GR00T modality 配置注册
├── scripts/
│   ├── convert_mano_to_keypoints.py            # Step 0: MANO FK → 21 关键点
│   ├── retarget_to_sharpa.py                   # Step 1a: → Sharpa 22-DoF (CasADi+IPOPT)
│   ├── retarget_to_gripper.py                  # Step 1b: → 夹爪开合度
│   ├── mano_to_eef.py                          # EEF 9D 提取 + 数据组装
│   ├── convert_ego_to_lerobot.py               # Step 2: → LeRobot V2 (--mode sharpa|gripper)
│   ├── merge_lerobot_datasets.py               # 合并各 episode → 统一训练数据集
│   ├── validate_dataset.py                     # 数据集验证
│   ├── visualize_sharpa_overlay_3d.py          # Sharpa AR overlay (MuJoCo)
│   ├── visualize_gripper_overlay.py            # Gripper overlay
│   ├── launch_pretrain.sh                         # 训练启动器
│   ├── utils.py                                # 公共工具
│   ├── mano_models/                            # MANO .pkl 权重
│   └── urdf/sharpa-urdf-usd-xml/               # Sharpa Wave URDF/MJCF/meshes
├── gr00t/                                      # 自包含 GR00T N1.7 训练框架
│   ├── configs/                                # 模型、数据、训练配置
│   ├── data/                                   # 数据集加载与处理
│   ├── model/                                  # VLA 架构 (Qwen3-VL + DiT)
│   ├── experiment/                             # 训练编排
│   └── utils/
└── output/                                     # 所有中间结果与训练输出 (gitignored)
```

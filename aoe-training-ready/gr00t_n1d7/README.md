# AoE Training-Ready: GR00T N1.7 Recipe

A self-contained toolchain that converts raw egocentric hand data (from AoE smartphone collection) into training-ready format and pretrains NVIDIA GR00T N1.7 — no external Isaac-GR00T dependency required.

```
AoE Raw Data                          Trained Model
(MANO params + ego video)             (GR00T N1.7 checkpoint)
         │                                    ▲
         ▼                                    │
 ┌───────────────────────────────────────────────────────────┐
 │  Step 0        Step 1         Step 2       Step 3         │
 │  MANO FK  ──► Retarget  ──► LeRobot  ──► Pretrain        │
 │  (21 kp)     (Sharpa/Grip)   (V2.1)      (GR00T N1.7)    │
 └───────────────────────────────────────────────────────────┘
```


## Prerequisites

### Hardware

### Data & Models

| Item | Description | How to Obtain |
|------|-------------|---------------|
| AoE dataset | `poc_deliver/` directory with raw episodes | Internal: see PROJECT.md |
| GR00T-N1.7-3B | Base model weights (full VLA checkpoint) | Download to `/path/to/GR00T-N1.7-3B` |
| Cosmos-Reason2-2B | VLM backbone (Qwen3-VL); provides tokenizer & visual processor | `huggingface-cli download nvidia/Cosmos-Reason2-2B` |
| MANO models | `MANO_LEFT.pkl` + `MANO_RIGHT.pkl` | Run `assets/mano/download_mano.sh` from repo root |
| Sharpa URDF | Robot hand description files (URDF + MJCF + STL meshes) | Bundled in `scripts/urdf/sharpa-urdf-usd-xml/` |

> **Note on Cosmos-Reason2-2B**: GR00T N1.7 uses `nvidia/Cosmos-Reason2-2B` (a Qwen3-VL architecture model) as its vision-language backbone. The training code loads the tokenizer and image processor from this model at startup. It will be auto-downloaded from HuggingFace Hub on first run, or you can pre-download it and pass the local path via `--vlm-model-name`:
> ```bash
> huggingface-cli download nvidia/Cosmos-Reason2-2B --local-dir /path/to/Cosmos-Reason2-2B
> # Then use: --vlm-model-name /path/to/Cosmos-Reason2-2B
> ```


## Quick Start: End-to-End

This section walks through the complete pipeline — from raw AoE data to a running training job.

### Step 0: Install

```bash
cd aoe-training-ready/gr00t_n1d7

# Create the environment
uv sync --python 3.10
source .venv/bin/activate

# Verify key dependencies
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import smplx; print('smplx OK')"
python -c "import casadi; print('CasADi OK')"
```

### Step 1: Data Conversion

Choose ONE mode: **sharpa** (62D, for dexterous hands) or **gripper** (20D, for parallel grippers).

#### Sharpa mode (recommended):

```bash
DATA_ROOT=/path/to/poc_deliver
OUTPUT=output

# 1a. MANO FK → 21 keypoints (~2 min for 111 episodes)
python scripts/convert_mano_to_keypoints.py \
    --data-root $DATA_ROOT \
    --output-dir $OUTPUT

# 1b. Keypoints → Sharpa 22-DoF joint angles (~5 min, CasADi+IPOPT optimization)
python scripts/retarget_to_sharpa.py \
    --data-root $DATA_ROOT \
    --keypoints-dir $OUTPUT \
    --output-dir $OUTPUT

# 1c. Assemble into LeRobot V2.1 format (~1 min)
python scripts/convert_ego_to_lerobot.py --mode sharpa \
    --data-root $DATA_ROOT \
    --retarget-dir $OUTPUT \
    --output-dir $OUTPUT
```

#### Gripper mode (simpler alternative):

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

**Expected output** after Step 1 (per episode):
```
output/<episode>/
├── hands_keypoints.npz            # (2, T, 21, 3) world + cam keypoints
├── hands_retargeted_sharpa.npz    # (T, 22) per hand joint angles
└── ego_sharpa_lerobotv21/         # LeRobot V2.1 dataset
    ├── data/chunk-000/episode_000000.parquet
    ├── videos/chunk-000/observation.images.ego_view/episode_000000.mp4
    └── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
```

### Step 2: Merge Datasets

Combine per-episode datasets into a single training dataset:

```bash
python scripts/merge_lerobot_datasets.py \
    --input-dir $OUTPUT \
    --mode sharpa \
    --output-dir $OUTPUT/ego_sharpa_merged
```

**Expected output**:
```
output/ego_sharpa_merged/
├── data/chunk-000/episode_000000.parquet ... episode_000110.parquet
├── videos/chunk-000/observation.images.ego_view/episode_*.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
```

### Step 3: Validate (optional but recommended)

```bash
python scripts/validate_dataset.py $OUTPUT/ego_sharpa_merged
```

Expected: all checks pass, prints dataset summary (episodes, frames, dimensions).

### Step 4: Launch Training

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

**Expected output**:
```
output/sharpa_train_YYYYMMDD_HHMMSS/
├── train.log
├── config.json
├── checkpoint-*/
│   ├── model*.safetensors
│   ├── optimizer.pt
│   └── trainer_state.json
└── wandb_config.json (if WandB enabled)
```


## Training Configuration

This recipe implements **EgoScale Stage I: large-scale human pretraining** — all model parameters are unfrozen (VLM backbone + visual encoder + projector + DiT action head). The model learns general manipulation priors from egocentric human hand data via flow-matching action prediction.

### launch_pretrain.sh Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `sharpa` | `sharpa` (62D) or `gripper` (20D) |
| `--base-model` | (required) | Path to GR00T-N1.7-3B weights |
| `--dataset` | (required) | Path to merged LeRobot dataset |
| `--vlm-model-name` | `nvidia/Cosmos-Reason2-2B` | HuggingFace model ID or local path for the VLM backbone (tokenizer + visual processor). Use a local path to avoid downloading from HuggingFace Hub on every run. |
| `--output` | auto-generated | Output directory for checkpoints |
| `--deepspeed-stage` | 3 | DeepSpeed ZeRO stage. 2=optimizer sharding, 3=full parameter+optimizer+gradient sharding. Use 3 for consumer GPUs (e.g. RTX 4090). |
| `--no-gradient-checkpointing` | (on by default) | Disable gradient checkpointing. On by default to save VRAM (trades compute for memory). |
| `--max-steps` | 1000 | Total training steps |
| `--batch-size` | 32 | Global batch size (across all GPUs) |
| `--lr` | 1e-4 | Learning rate |
| `--num-gpus` | 1 | Number of GPUs (auto-uses torchrun if >1) |
| `--save-steps` | (none) | Save checkpoint every N steps |
| `--skip-weight-loading` | false | Train from scratch (ignore base model weights) |

### Mode Comparison

| | Sharpa (62D) | Gripper (20D) |
|---|---|---|
| State dim | 62 | 20 |
| Hand repr | 22-DoF Sharpa Wave joint angles | 1D open/close scalar |
| Retarget | CasADi + IPOPT nonlinear optimization | Thumb-index distance normalization |
| Use case | Deploy to Sharpa Wave dexterous hands | Deploy to parallel grippers |
| Pretrained projector | Yes (reuses GR00T index 26) | No (trains new projector) |
| Recommended steps | 500-1000 (finetuning) | 2000+ (training from scratch) |

**State layout**:
```
sharpa  (62D): [left_wrist_eef(9), right_wrist_eef(9), left_hand_joints(22), right_hand_joints(22)]
gripper (20D): [left_wrist_eef(9), right_wrist_eef(9), left_gripper(1), right_gripper(1)]
```

### Multi-GPU Training

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

Uses `torchrun` with DeepSpeed ZeRO-3 + gradient checkpointing by default. Batch size is automatically aligned to GPU count.

### WandB Integration

To enable Weights & Biases logging, remove the `--no-use-wandb` flag by passing extra args:

```bash
bash scripts/launch_pretrain.sh \
    --mode sharpa \
    --base-model /path/to/GR00T-N1.7-3B \
    --dataset output/ego_sharpa_merged \
    -- --use-wandb --wandb-project aoe-gr00t
```

### Key Hyperparameters (in training code)

| Parameter | Value | Notes |
|-----------|-------|-------|
| action_horizon | 16 | Action prediction window (steps) |
| state_dropout_prob | 0.8 | High dropout → model relies on vision+language |
| tune_llm | true | Pretrain: unfreeze VLM backbone (Qwen3-VL) |
| tune_visual | true | Pretrain: unfreeze visual encoder (ViT) |
| tune_projector | true | Pretrain: unfreeze multimodal projector |
| tune_diffusion_model | true | Pretrain: unfreeze DiT action head |
| DeepSpeed | ZeRO-3 | Full parameter+optimizer+gradient sharding |
| gradient_checkpointing | true | Gradient checkpointing (trades compute for VRAM) |

> **Note**: All four `tune_*` flags are enabled — this is full-parameter pretraining (EgoScale Stage I), not finetuning. For downstream finetuning (Stage III), you would freeze VLM+ViT and only tune projector+DiT.


## Pipeline Details

### MANO FK → 21 Keypoints (Step 0)

**Script**: `scripts/convert_mano_to_keypoints.py`

Converts MANO parametric representation → explicit 3D joint positions using `smplx.MANOLayer` forward kinematics:

```
MANO params (pred_rot, pred_trans, pred_hand_pose, pred_betas)
    → smplx.MANOLayer (Linear Blend Skinning)
    → 778 mesh vertices + 21 joint positions
    → mano_to_openpose index mapping
    → joints_world (2, T, 21, 3)
```

Fingertip keypoints (indices 4, 8, 12, 16, 20) come from specific mesh vertices, not FK chain endpoints — this is standard MANO convention with typical accuracy of 5-10mm.

**21 keypoint ordering (OpenPose Hand)**:
```
 [0]     Wrist
 [1-4]   Thumb:  CMC → MCP → IP → TIP
 [5-8]   Index:  MCP → PIP → DIP → TIP
 [9-12]  Middle: MCP → PIP → DIP → TIP
[13-16]  Ring:   MCP → PIP → DIP → TIP
[17-20]  Pinky:  MCP → PIP → DIP → TIP
```

Joint abbreviations: CMC = Carpometacarpal, MCP = Metacarpophalangeal, PIP = Proximal Interphalangeal, DIP = Distal Interphalangeal, IP = Interphalangeal (thumb only), TIP = fingertip (mesh vertex).

**Output**: `output/<episode>/hands_keypoints.npz`

| Field | Shape | Description |
|-------|-------|-------------|
| `joints_world` | (2, T, 21, 3) | 21 keypoints in world coordinates |
| `joints_cam` | (2, T, 21, 3) | 21 keypoints in camera coordinates |
| `pred_valid` | (2, T) | Validity mask (1=valid, 0=hand not detected) |

### Retarget → Sharpa Wave 22-DoF (Step 1a)

**Script**: `scripts/retarget_to_sharpa.py`

Solves for 22 joint angles of the Sharpa Wave dexterous hand such that its fingertips match the human hand's fingertip positions. Uses CasADi symbolic framework with IPOPT nonlinear solver.

**Per-frame algorithm**:
```
Input: keypoints[t] (21, 3) in world coordinates

A. Extract 5 fingertip positions relative to wrist
   tips = keypoints[[4,8,12,16,20]] - keypoints[0]    # (5, 3)

B. Build palm-local coordinate frame (aligns with Sharpa rest pose)
   Z = normalize(middle_MCP - wrist)       # finger direction
   Y = normalize(ring_MCP - index_MCP)     # lateral (L/R mirrored)
   X = Y × Z                               # palm normal
   R_hand = [X | Y | Z]

C. Transform tips to Sharpa-local frame
   tips_local = R_hand.T @ tips.T

D. Nonlinear optimization (CasADi + IPOPT)
   minimize  ||FK(q) - tips_local||²
   subject to  q_min ≤ q ≤ q_max
   warm start from q_prev
```

**Why Step B (coordinate alignment)?** Sharpa FK computes fingertip positions in the hand's local frame (rest pose: fingers along +Z). Without alignment, when the human hand rotates in world space, the target positions don't match Sharpa's FK output frame — the solver fails catastrophically.

**Left/right difference**: Left and right Sharpa URDFs have Y-axis mirroring. The lateral axis construction is flipped for the right hand.

**Sharpa Wave 22-DoF joint order**:
```
Thumb  (5): CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP
Index  (4): MCP_FE, MCP_AA, PIP, DIP
Middle (4): MCP_FE, MCP_AA, PIP, DIP
Ring   (4): MCP_FE, MCP_AA, PIP, DIP
Pinky  (5): CMC, MCP_FE, MCP_AA, PIP, DIP
```

FE = Flexion/Extension (bend/straighten), AA = Abduction/Adduction (spread/close).

**Retarget accuracy**:

| Metric | Left hand | Right hand |
|--------|-----------|------------|
| Mean error | 4.4 mm | 2.2 mm |
| P95 error | 25.5 mm | 6.7 mm |

**Output**: `output/<episode>/hands_retargeted_sharpa.npz`

| Field | Shape | Description |
|-------|-------|-------------|
| `left_sharpa_joints` | (T, 22) | Left hand joint angles (rad) |
| `right_sharpa_joints` | (T, 22) | Right hand joint angles (rad) |
| `pred_valid` | (2, T) | Validity mask |

### Retarget → Gripper (Step 1b)

**Script**: `scripts/retarget_to_gripper.py`

Computes normalized thumb-to-index fingertip distance as a [0, 1] gripper aperture:

```
gripper = ||thumb_tip - index_tip|| / sqrt(thumb_length² + index_length²)
    thumb_length = ||kp[4] - kp[1]||   (thumb CMC → tip)
    index_length = ||kp[8] - kp[5]||   (index MCP → tip)
```

- 0.0 = pinch (thumb touches index)
- 1.0 = fully open (~90° spread)

**Output**: `output/<episode>/hands_retargeted_gripper.npz`

| Field | Shape | Description |
|-------|-------|-------------|
| `left_gripper` | (T, 1) | Left hand aperture [0, 1] |
| `right_gripper` | (T, 1) | Right hand aperture [0, 1] |
| `pred_valid` | (2, T) | Validity mask |

### Wrist EEF 9D (shared by both modes)

Extracted directly from `hands.npz` global rotation and translation:

```
EEF 9D = [x, y, z, rot6d(6)]
    rot6d = rotation_matrix[:2, :].flatten()   # first two rows flattened
```

### LeRobot V2 Conversion (Step 2)

**Script**: `scripts/convert_ego_to_lerobot.py`

Converts processed data into LeRobot V2.1 format compatible with GR00T N1.7's data loader:
- Splits episodes into continuous valid segments (by `pred_valid` mask)
- Sub-samples video from 30fps → 15fps
- Computes action as delta between consecutive states
- Applies action chunking (horizon=16 steps)
- Writes per-episode dataset: parquet + video symlinks + metadata JSONs


## AoE Data Format Reference

Each AoE episode directory contains:

```
raw_{collector_id}_seg_{segment_id}/
├── ego_process/ego_hands_reconstruction/
│   └── hands.npz               ← MANO hand reconstruction (source data)
├── ego_process/ego_undistorted_video/
│   └── raw_video_undistorted.mp4
└── ego_annotation/
    └── ego_action_annotation.json
```

### hands.npz Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `pred_rot` | (2, T, 3) | Wrist global rotation (axis-angle, world frame), [0]=left [1]=right |
| `pred_trans` | (2, T, 3) | Wrist position (meters, world frame) |
| `pred_hand_pose` | (2, T, 45) | 15 finger joint rotations (axis-angle, 15x3=45D) |
| `pred_betas` | (2, T, 10) | MANO shape parameters |
| `pred_valid` | (2, T) | Valid frame flags (1=valid, 0=hand not detected) |
| `R_c2w` / `t_c2w` | (T,3,3) / (T,3) | Camera → world transform |
| `R_w2c` / `t_w2c` | (T,3,3) / (T,3) | World → camera transform |
| `focal` | scalar | Camera focal length (pixels) |

### Dataset Statistics (poc_deliver)

| Metric | Value |
|--------|-------|
| Episodes | 111 |
| Total frames | 317,553 |
| Duration | 2.94 hours (30fps) |
| Episode length | 18.5s ~ 255.6s (mean 95.4s) |
| Left hand validity | 98.05% |
| Right hand validity | 96.77% |
| Scenes | 18 |
| Action verbs | 97 |
| Annotated segments | 1,338 |


## Visualization

### Sharpa AR Overlay (MuJoCo mesh + keypoints on ego video)

```bash
python scripts/visualize_sharpa_overlay_3d.py \
    --data-root /path/to/poc_deliver \
    --keypoints-dir output --retarget-dir output \
    --output-dir output --max-episodes 2
```

### Gripper Overlay (thumb-index line + aperture value)

```bash
python scripts/visualize_gripper_overlay.py \
    --data-root /path/to/poc_deliver \
    --keypoints-dir output --retarget-dir output \
    --output-dir output --max-episodes 2
```

Output videos are saved as `output/<episode>/vis_retargeted_*_overlay.mp4`.


## Troubleshooting

### CasADi / IPOPT installation fails

CasADi bundles IPOPT for most platforms. If `pip install casadi` fails:
```bash
conda install -c conda-forge casadi   # alternative via conda
```

### CUDA out of memory during training

- Reduce `--batch-size` (try 8 or 4)
- Use multi-GPU: `--num-gpus 2` (or more)
- Use ZeRO-3 (on by default): `--deepspeed-stage 3`
- Keep gradient checkpointing on (default): do not pass `--no-gradient-checkpointing`

### "ERROR: Dataset meta/ not found"

You forgot the merge step. Run:
```bash
python scripts/merge_lerobot_datasets.py \
    --input-dir output --mode sharpa --output-dir output/ego_sharpa_merged
```

### Video symlinks broken after moving data

The merge script creates absolute symlinks. If you move the dataset, re-run the merge, or convert symlinks to relative:
```bash
find output/ego_sharpa_merged/videos -type l -exec sh -c \
    'target=$(readlink "$1"); ln -sf "$(realpath --relative-to="$(dirname "$1")" "$target")" "$1"' _ {} \;
```

### smplx / MANO model not found

If MANO models are missing, run `bash assets/mano/download_mano.sh` from the repo root with your downloaded `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` (see above). These models are from the [MANO project](https://mano.is.tue.mpg.de/) (requires registration).

### Sharpa URDF not found

Ensure `scripts/urdf/sharpa-urdf-usd-xml/wave_01/` exists with `left_sharpa_wave/` and `right_sharpa_wave/` subdirectories containing `.urdf` and `meshes/`. These files are bundled in the repo; no extra download needed.


## Directory Structure

```
gr00t_n1d7/
├── README.md                                   # This file
├── README_zh.md                                # Chinese version
├── pyproject.toml                              # Dependencies & package config
├── configs/
│   ├── human_ego_config.py                     # Mode registry (gripper/sharpa)
│   └── register_aoe_modality.py                # GR00T modality config registrar
├── scripts/
│   ├── convert_mano_to_keypoints.py            # Step 0: MANO FK → 21 keypoints
│   ├── retarget_to_sharpa.py                   # Step 1a: → Sharpa 22-DoF (CasADi+IPOPT)
│   ├── retarget_to_gripper.py                  # Step 1b: → Gripper aperture
│   ├── mano_to_eef.py                          # EEF 9D extraction + data assembly
│   ├── convert_ego_to_lerobot.py               # Step 2: → LeRobot V2 (--mode sharpa|gripper)
│   ├── merge_lerobot_datasets.py               # Merge per-episode → single training dataset
│   ├── validate_dataset.py                     # Dataset validation
│   ├── visualize_sharpa_overlay_3d.py          # Sharpa AR overlay (MuJoCo)
│   ├── visualize_gripper_overlay.py            # Gripper overlay
│   ├── launch_pretrain.sh                         # Training launcher
│   ├── utils.py                                # Shared utilities
│   ├── mano_models/                            # MANO .pkl weights
│   └── urdf/sharpa-urdf-usd-xml/               # Sharpa Wave URDF/MJCF/meshes
├── gr00t/                                      # Self-contained GR00T N1.7 training framework
│   ├── configs/                                # Model, data, training configs
│   ├── data/                                   # Dataset loading & processing
│   ├── model/                                  # VLA architecture (Qwen3-VL + DiT)
│   ├── experiment/                             # Training orchestration
│   └── utils/
└── output/                                     # All intermediate & training outputs (gitignored)
```

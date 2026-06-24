## AoE Training-Ready

Zero-setup recipes for training state-of-the-art vision-language-action (VLA) models on [AoE](https://github.com/AoE-Ego) ego-centric data.

### Architecture

AoE Training-Ready provides **per-model conversion recipes** that respect each model's native training data format, rather than forcing all data through a single intermediate representation. The goal is **train ready** — users can go from raw AoE data to a running training job with minimal effort.

```
AoE Data ──→ vitra/          ──→ VITRA episodic format ──→ VITRA pretraining
         ├──→ gr00t_n1d7/     ──→ LeRobot V2.1 ──→ GR00T N1.7 pretraining
         └──→ H-RDT/          ──→ 48D action HDF5 ──→ H-RDT pretraining
```

### Available Recipes

| Model | Recipe | Status | Description |
|-------|--------|--------|-------------|
| [VITRA](https://github.com/microsoft/VITRA) | [`vitra/`](vitra/) | ✅ Verified | Ego-centric hand manipulation pretraining via MANO FK conversion |
| [GR00T N1.7](https://developer.nvidia.com/gr00t) | [`gr00t_n1d7/`](gr00t_n1d7/) | ✅ Verified | Bimanual robot manipulation pretraining (sharpa + gripper modes) |
| [H-RDT](https://github.com/HongzheBi/H_RDT) | [`H-RDT/`](H-RDT/) | ✅ Verified | Bimanual hand-action pretraining via 48D action representation |

### Getting Started

Each recipe is self-contained under `<model_name>/` with its own README, conversion scripts, and configuration templates. Navigate to the recipe directory for model-specific instructions.

### Contributing a New Recipe

To add support for a new VLA model:

1. Create `<model_name>/` with at minimum:
   - A conversion script (`convert_aoe_to_<model>.py`)
   - A verification script
   - A training config template
   - A `README.md` with Quick Start instructions
2. Verify end-to-end: convert → train → confirm loss convergence
3. Submit a PR to this repository

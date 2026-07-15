# AoE Training-Ready

Open-AoE Training-Ready is a collection of **model-specific integration recipes**. Each recipe follows the target model's own data design, action semantics, dependencies, and training workflow.

> [!IMPORTANT]
> There is no universal Open-AoE training format, action specification, converter, or launcher across these recipes. Choose a target model below, then treat that submodule's README as the source of truth.

## Choose a Recipe

### VLA and robot policies

| Target model | Start here | Integration focus |
|---|---|---|
| ACT / Diffusion Policy / π0.5 | [LeRobot recipe](lerobot/README_OPEN_AOE.md) | Open-AoE conversion and training through a compatible LeRobot checkout |
| SmolVLA | [SmolVLA recipe](smolvla/README_OPEN_AOE.md) | Open-AoE conversion, fine-tuning, evaluation, and loss inspection |
| GR00T N1.7 | [GR00T N1.7 recipe](gr00t_n1d7/README.md) · [中文](gr00t_n1d7/README_zh.md) | End-to-end preparation, embodiment adaptation, validation, and pretraining |
| H-RDT | [H-RDT recipe](H-RDT/README.md) | H-RDT-specific action preprocessing, language encoding, statistics, and pretraining |
| VITRA | [VITRA recipe](vitra/README.md) | VITRA episodic conversion, verification, upstream patch, and training setup |

### Video and world-action models

| Target model | Start here | Integration focus |
|---|---|---|
| DreamZero | [DreamZero recipe](dreamzero/README_OPEN_AOE.md) | Open-AoE conversion, upstream adaptation, training, and loss curves |
| LingBot-VA | [LingBot-VA recipe](lingbot-va/README_OPEN_AOE.md) | Open-AoE conversion, latent preparation, upstream adaptation, and post-training |
| Ctrl-World | [Ctrl-World recipe](Ctrl-World/README_OPEN_AOE.md) | Ctrl-World-specific data preparation and training without an upstream patch |
| iVideoGPT | [iVideoGPT recipe](ivideogpt/README_OPEN_AOE.md) | Action-conditioned video modeling, conversion, training, and evaluation |

### Latent-action and world models

| Target model | Start here | Integration focus |
|---|---|---|
| GenieRedux | [GenieRedux recipe](genie-redux/README_OPEN_AOE.md) | Latent-action, tokenizer, dynamics, guided-action, and rollout workflows |
| LAOM | [LAOM recipe](laom/README_OPEN_AOE.md) | LAOM-specific HDF5 conversion and supervised latent-action training |
| AdaWorld | [AdaWorld recipe](adaworld/README_OPEN_AOE.md) | Open-AoE frame-pair preparation and latent-action model training |
| DreamDojo | [DreamDojo recipe](dreamdojo/README_OPEN_AOE.md) | Zero-shot preview and MANO-conditioned post-training workflow |

## How to Start

1. Get Open-AoE data from [Hugging Face](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h) or [ModelScope](https://www.modelscope.cn/datasets/inclusionAI/OpenAoE-2000h).
2. Choose the model family and target recipe from the tables above.
3. Open that recipe's README and follow its documented upstream version, environment, expected Open-AoE inputs, conversion steps, and training commands.
4. Keep outputs and intermediate datasets separate between recipes unless both recipe READMEs explicitly document compatibility.

Do not assume that a converter, action representation, environment variable, or launch command from one recipe applies to another.

## Repository Convention

Each model integration lives in its own directory and should document:

- the target upstream project and compatible version;
- required data, checkpoints, and external dependencies;
- model-specific conversion or preprocessing;
- training, evaluation, and expected outputs;
- any upstream patch or project-local source changes.

Most upstream projects and model weights are not vendored into this repository. Follow the selected recipe's setup and license notes before use.

## Contributing a Recipe

Add a self-contained directory for the target model. Include a README that explains the complete Open-AoE workflow, plus only the converters, launchers, configuration, evaluation tools, and patches required by that model.

Avoid presenting model-specific representations or commands as repository-wide standards.

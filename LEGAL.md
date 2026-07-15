# Legal Notice — Open-AoE

## Project License

Open-AoE is licensed under the Apache License, Version 2.0 (see [LICENSE](./LICENSE)).
All original code contributed by Open-AoE Contributors is available under this license.

## Third-Party Components

### MANO Hand Model

The MANO hand model (MANO_LEFT.pkl, MANO_RIGHT.pkl) is **not included** in this repository.
Users must register at https://mano.is.tue.mpg.de/ and agree to the MANO license terms
before downloading and using these files. See `assets/mano/download_mano.sh` for setup.

### GR00T N1.7 Training Framework

Code under `aoe-training-ready/gr00t_n1d7/gr00t/` is originally from
NVIDIA CORPORATION & AFFILIATES, licensed under Apache 2.0.
See [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).

### Sharpa Robot Description

URDF/USD/MJCF files under `aoe-training-ready/gr00t_n1d7/scripts/urdf/`
are from Sharpa Group, licensed under Apache 2.0.

### Phantom

The retargeting pipeline under `aoe-reconstruct-retarget/phantom/` is an adaptation of
[Phantom](https://github.com/MarionLepert/phantom) by MarionLepert, licensed under MIT.

### dex-retargeting

The Dex3 and Inspire hand URDF files under `aoe-reconstruct-retarget/phantom/assets/robots/hands/`
are adapted from the [dex-retargeting](https://github.com/dex-retargeting/dex-retargeting) project.

### VITRA

The VITRA training recipe under `aoe-training-ready/vitra/` is an
adaptation layer for the [VITRA](https://github.com/microsoft/VITRA) project by Microsoft.
Users must clone VITRA separately and apply the provided patch.

### H-RDT

The H-RDT training recipe under `aoe-training-ready/H-RDT/` is an
adaptation layer for the [H-RDT](https://github.com/HongzheBi/H_RDT) project.
Users must clone H-RDT separately and apply the provided patch. The recipe
also depends on [HaWoR](https://github.com/ThunderVVV/HaWoR) (CC-BY-NC-ND 4.0)
for MANO forward kinematics — users must clone HaWoR separately.

### iVideoGPT

The iVideoGPT training recipe under `aoe-training-ready/ivideogpt/` is an adaptation
layer for the [iVideoGPT](https://github.com/thuml/iVideoGPT) project (MIT). Users must
clone iVideoGPT separately and apply the provided patch.

### GenieRedux

The GenieRedux training recipe under `aoe-training-ready/genie-redux/` is an adaptation
layer for the [GenieRedux](https://github.com/insait-institute/GenieRedux) project (MIT).
Users must clone GenieRedux separately and apply the provided patch (a VQ codebook-collapse fix).

### SmolVLA / LeRobot

The SmolVLA training recipe under `aoe-training-ready/smolvla/` depends on the
[LeRobot](https://github.com/huggingface/lerobot) library (Apache 2.0) and the
`lerobot/smolvla_base` model from HuggingFace. LeRobot is a pip dependency (not vendored);
no patch is required. Base weights are downloaded from HuggingFace under their model-card license.

### LAOM

The laom training recipe under `aoe-training-ready/laom/` is an adaptation layer for the
[laom](https://github.com/dunnolab/laom) project (Apache 2.0). Users must clone laom
separately; no patch is required.

### AdaWorld

The AdaWorld training recipe under `aoe-training-ready/adaworld/` is an adaptation layer for
the [AdaWorld](https://github.com/Little-Podi/AdaWorld) project (Apache 2.0). Users must
clone AdaWorld separately; no patch is required.

### DreamDojo

The DreamDojo training recipe under `aoe-training-ready/dreamdojo/` is an adaptation layer for
the [DreamDojo](https://github.com/NVIDIA/DreamDojo) project (Cosmos-Predict2.5, Apache 2.0).
Users must clone DreamDojo separately and apply the provided patch, which modifies three upstream
files: a checkpoint-revision pin (a 404 fix); a data-loader change that injects AoE MANO hand
actions into the model action vector and swaps the video decoder to `decord`; and an action-embedder
weight-init fix (initializes an action-embedder layer the upstream leaves uninitialized). No upstream
code is vendored. Cosmos-Predict2.5 model weights are downloaded from NVIDIA/HuggingFace under their
respective licenses and are not included; no MANO model file is used or shipped (finger kinematics
are computed analytically from the standard MANO joint tree).

## Contributing

By contributing to this project, you agree that your contributions will be licensed
under the Apache License, Version 2.0.

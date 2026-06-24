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

The retargeting pipeline under `aoe-retarget-replay/phantom/` is an adaptation of
[Phantom](https://github.com/MarionLepert/phantom) by MarionLepert, licensed under MIT.

### dex-retargeting

The Dex3 and Inspire hand URDF files under `aoe-retarget-replay/phantom/assets/robots/hands/`
are adapted from the [dex-retargeting](https://github.com/dex-retargeting/dex-retargeting) project.

### VITRA

The VITRA training recipe under `aoe-training-ready/vitra/` is an
adaptation layer for the [VITRA](https://github.com/microsoft/VITRA) project by Microsoft.
Users must clone VITRA separately and apply the provided patch.

## Contributing

By contributing to this project, you agree that your contributions will be licensed
under the Apache License, Version 2.0.
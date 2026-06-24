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

Code under `release/aoe-training-ready/recipes/gr00t_n1d7/gr00t/` is originally from
NVIDIA CORPORATION & AFFILIATES, licensed under Apache 2.0.

### Sharpa Robot Description

URDF/USD/MJCF files under `release/aoe-training-ready/recipes/gr00t_n1d7/scripts/urdf/`
are from Sharpa Group, licensed under Apache 2.0.

### dex-retargeting

The Dex3 and Inspire hand URDF files under `release/aoe-retarget-replay/phantom/assets/robots/hands/`
are adapted from the [dex-retargeting](https://github.com/dex-retargeting/dex-retargeting) project.

### VITRA

The VITRA training recipe under `release/aoe-training-ready/recipes/vitra/` is an
adaptation layer for the [VITRA](https://github.com/microsoft/VITRA) project by Microsoft.
Users must clone VITRA separately and apply the provided patch.

## Contributing

By contributing to this project, you agree that your contributions will be licensed
under the Apache License, Version 2.0.
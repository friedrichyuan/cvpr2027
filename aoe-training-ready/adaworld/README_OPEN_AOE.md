# AdaWorld (LAM) on Open-AoE

Open-AoE integration recipe for [Little-Podi/AdaWorld](https://github.com/Little-Podi/AdaWorld)
(ICML 2025, *"Learning Adaptable World Models with Latent Actions"*) — specifically its
`lam.modules.lam.LatentActionModel`, a VAE-style latent-action autoencoder
(spatiotemporal encoder → z_mu/z_var → decode → reconstruct the next frame). This is an
integration recipe; the upstream is an external checkout, **not** vendored.

## Version

- Upstream: [Little-Podi/AdaWorld](https://github.com/Little-Podi/AdaWorld) — verified commit `fe3701e`
- Patch: none

## Environment

```bash
git clone https://github.com/Little-Podi/AdaWorld
cd AdaWorld && git checkout fe3701e
python -m venv .venv && . .venv/bin/activate     # py3.10 base (their pins: torch 2.0.1+cu117)
pip install -r requirements.txt
export ADAWORLD_UPSTREAM_ROOT=/PATH_TO/AdaWorld
```

## Data + train

Trains the official `LatentActionModel` at the shipped `lam/config/lam.yaml` config
(474.7M params, dim=1024, 16+16 blocks, latent_dim=32) on AoE 64×64 frame-pairs — fits a
single 24GB 4090 at batch 4 (at 64-res / patch 16 there are only 16 patches/frame). Objective
replicates their `LAM.shared_step`: `loss = mse(videos[:,1:] − recon) + β·KL(z_mu, z_var)`, β=2e-4.

```bash
export OPEN_AOE_NPZ_DIR=/PATH_TO/aoe_npz/aoe26
export EPOCHS=10
./scripts/aoe_train --dry-run train
./scripts/aoe_train train
```

The npz frames come from any Open-AoE converter that writes `image` arrays (e.g. the
[`ivideogpt/`](../ivideogpt/) or [`smolvla/`](../smolvla/) converters).

## Results (POC, full data, 10 epochs / 20k steps)

- Loss `0.128 → 0.0025` (mse `0.125 → 0.0025`), KL `21 → 0.011` — clean convergence at their full config (not gutted).

## Scope note

This recipe delivers the **LAM core**, which runs at full config on one 4090. AdaWorld's
headline *adaptable world model* (`worldmodel/train_adapt.py`, an 8-GPU SVD-based finetune) is
out of scope here. It complements [`genie-redux/`](../genie-redux/) — both are Genie-style
latent-action models.

## License

Apache-2.0 (Open-AoE Contributors). Upstream AdaWorld is referenced by checkout (not vendored);
see the root [`LEGAL.md`](../../LEGAL.md).

## File reference

| File | Purpose |
|------|---------|
| `scripts/aoe_train` | Launcher: `train` |
| `scripts/adaworld_lam.py` | Thin LAM trainer: AoE frame-pairs → `LatentActionModel` |

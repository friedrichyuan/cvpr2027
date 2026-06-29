# laom (LAOM) on Open-AoE

Open-AoE integration recipe for [dunnolab/laom](https://github.com/dunnolab/laom)
(ICML 2025, *"Latent Action Learning Requires Supervision in the Presence of
Distractors"*) — running its **LAOM-with-supervision** pretraining on Open-AoE
egocentric video, where the in-video **distractor is camera ego-motion** and the
**task action is the hand (20D)**. This is an integration recipe; the upstream is an
external checkout, **not** vendored.

## Version

- Upstream: [dunnolab/laom](https://github.com/dunnolab/laom) — verified commit `e133195`
- Patch: none

## Environment

```bash
git clone https://github.com/dunnolab/laom
cd laom && git checkout e133195
python -m venv .venv && . .venv/bin/activate     # build from a py3.9 base (needs ensurepip)
pip install -r requirements.txt "numpy<2"        # numpy<2 mandatory (h5py ABI)
export LAOM_UPSTREAM_ROOT=/PATH_TO/laom
```

## Data format

AoE npz (`image`, `action`) → laom HDF5: one group per clip with
`{obs (T,64,64,3) uint8, actions (T,A) f32, states (T,1) f32}` + root attr `img_hw`,
fixed 16-frame clips (their `DCSLAOMInMemoryDataset` requires uniform length). Hand 20D is
the task action; camera 6D is the distractor. See [`../ACTION_SPEC.md`](../ACTION_SPEC.md).

## Convert + train

```bash
export OPEN_AOE_NPZ_DIR=/PATH_TO/aoe_npz/aoe26
export LAOM_DATA_DIR=/PATH_TO/laom_data
./scripts/aoe_train convert            # writes aoe_laom_full.hdf5 + aoe_laom_labeled.hdf5 (20%)
export EPOCHS=30
./scripts/aoe_train --dry-run train
./scripts/aoe_train train
```

## Results (POC, 30 epochs / 5280 steps / ~9 min on one RTX 4090)

- Forward-dynamics (FDM) loss `1.514 → 0.056`.
- Supervised hand-action MSE (20% labels) `2.226 → 0.003` — small supervision grounds the latent on hand actions (LAOM's thesis, validated on real egocentric data).
- Latent-action R²: camera `0.27` vs hand `0.05` (~6× camera) unsupervised; with 20% labels hand R² rises to `0.52`.

## Notes

- Build the venv from a py3.9 base (system py3.12 lacks `ensurepip`).
- `numpy<2` is mandatory (else h5py ABI error "numpy.dtype size changed").
- `DCSLAOMInMemoryDataset` loads the HDF5 into RAM — for large data, convert a subset (cap `--max-clips`) or add a streaming loader.

## License

Apache-2.0 (Open-AoE Contributors). Upstream laom is referenced by checkout (not vendored);
see the root [`LEGAL.md`](../../LEGAL.md).

## File reference

| File | Purpose |
|------|---------|
| `scripts/aoe_train` | Launcher: `convert` / `train` |
| `scripts/aoe_to_laom_hdf5.py` | AoE npz → laom HDF5 (full + 20% labeled) |
| `scripts/run_laom_aoe.py` | Thin runner: upstream `train_laom()` + loss → CSV |

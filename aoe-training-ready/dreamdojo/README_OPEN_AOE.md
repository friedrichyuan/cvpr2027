# DreamDojo on Open-AoE (zero-shot preview + MANO post-train harness)

Open-AoE integration recipe for [NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo)
(Cosmos-Predict2.5) — a SOTA action-conditioned video world model. Two routes, both against an
external upstream checkout (**not** vendored):

1. **Zero-shot preview** — run the pretrained world model on Open-AoE to verify the data pipeline
   + inference on accessible hardware.
2. **MANO-conditioned post-train** — feed **real AoE hand actions** into the model's action vector
   and post-train, to close the robot→human domain gap.

> **Status.**
> - Zero-shot preview: ✅ works (numbers below).
> - MANO post-train: a reproducible harness — AoE MANO action export/adaptation, the
>   action-embedder `fc1` init fix, and the training launcher. Run it on your own hardware
>   (see *Data conversion* and *Post-train*).

## Version

- Upstream: [NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo) (Cosmos-Predict2.5) — verified commit `02f119b`
- Patch: `patches/open_aoe_support.patch` — **3 files**:
  - `checkpoint_db.py`: pin a checkpoint `revision` to `main` (a 404 infra fix).
  - `groot_dreams/data/dataset_mano.py`: **inject the real MANO action into the action vector
    `[220:352]`** (the upstream egodex path zeroed that slot — see *Post-train*), read the egodex
    video root from `$AOE_EGODEX_VIDEO_ROOT`, and decode video with `decord` (self-contained;
    avoids torchcodec's FFmpeg-lib requirement).
  - `cosmos_predict2/…/networks/minimal_v4_dit.py`: **action-embedder `fc1` init** — upstream leaves
    the action embedder's `fc1` uninitialized (all-zero on meta-device builds); the patch
    `trunc_normal_`-inits it (+ a startup assert) so injected MANO actions propagate through
    DreamDojo's conditioning path.

## Environment

```bash
git clone https://github.com/NVIDIA/DreamDojo
cd DreamDojo && git checkout 02f119b
git apply /PATH_TO/Open-AoE-dev/aoe-training-ready/dreamdojo/patches/open_aoe_support.patch
# follow the upstream install (uv sync). Known extra infra:
#   - pure-torch pytorch3d shim on PYTHONPATH (groot_dreams only imports transforms)
#   - the patched loader imports `decord` (pip install decord) — replaces torchcodec
#   - torchcodec/ffmpeg only needed for the zero-shot MP4 route; export HF_TOKEN=<your_token>
export DREAMDOJO_UPSTREAM_ROOT=/PATH_TO/DreamDojo
```

Key env vars (all default to `/PATH_TO/...` placeholders):

| Var | Meaning |
|---|---|
| `DREAMDOJO_UPSTREAM_ROOT` | the upstream checkout (on `PYTHONPATH`, cwd for training) |
| `OPEN_AOE_RAW_ROOT` | AoE raw delivery (`poc_deliver`) — converter input |
| `AOE_EGODEX_HDF5_ROOT` | egodex HDF5 output = training `dataset_path` |
| `AOE_EGODEX_VIDEO_ROOT` | paired mp4 symlinks; the patched `_VIDEO_ROOT` reads this |
| `IMAGINAIRE_OUTPUT_ROOT` | training output + checkpoints (fixed → DCP auto-resume) |

## Data conversion

**MANO route (post-train).** AoE `hands.npz` → per-node global-rotation **rot6d HDF5** (wrist +
20 ARKit nodes, camera frame, via analytic MANO-tree FK — no MANO `.pkl` needed) + a paired mp4
symlink, in the egodex layout DreamDojo's `MANODataset` consumes.

```bash
export OPEN_AOE_RAW_ROOT=/PATH_TO/Open-AoE/poc_deliver
export AOE_EGODEX_HDF5_ROOT=/PATH_TO/aoe_egodex_hdf5
export AOE_EGODEX_VIDEO_ROOT=/PATH_TO/aoe_egodex_videos
export NUM_CLIPS=500
./scripts/aoe_train --dry-run build-egodex   # prints the converter command
./scripts/aoe_train build-egodex             # writes $HDF5_ROOT/part0/*.hdf5 + $VIDEO_ROOT/part0/*.mp4
```

**Zero-shot route.** AoE undistorted video → 480×640 MP4. The output folder name must **not**
contain `gr1`/`g1`/`yam`/`agibot`, so DreamDojo routes it to the generic `VideoDataset` and the LAM
infers latent actions from the video.

```bash
export DD_VIDEO_DIR=/PATH_TO/aoe_dd_videos
export FRAMES=150        # must exceed the inference --num-frames
./scripts/aoe_train convert
```

## Post-train (MANO-conditioned)

**What the patch changes and why it matters.** DreamDojo's action vector is 384-dim with a built-in
**MANO slot at `[220:352]`**. The upstream egodex path shipped that slot **zeroed** (`gt=zeros(352)`,
`latent=ones(32)`) — i.e. it trained *without* real hand conditioning. The patch injects the real
AoE MANO action: `gt=zeros(220)` + `mano=action(132)` + `latent=zeros(32)` → concat 384. This turns
a nominal egodex run into a genuine **MANO-action-conditioned post-train**. The patch *also* initializes
the action embedder's `fc1` (`minimal_v4_dit.py`, see *Version*) so the injected action propagates
through DreamDojo's conditioning path.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4        # nproc_per_node & fsdp_shard follow this
export IMAGINAIRE_OUTPUT_ROOT=/PATH_TO/aoe_runs/dreamdojo_posttrain
export MAX_ITER=10000 SAVE_ITER=500 BATCH=1
./scripts/aoe_train --dry-run posttrain      # prints the torchrun command
./scripts/aoe_train posttrain
```

- **Hardware (corrected).** The **2B** model at 480×640 post-trains on **≥45GB GPUs (L40-class),
  ~24GB/card at batch 1** — a smoke test confirmed it fits free L40s, so it does **not** require
  8×H100 (only the 14B does). Multi-GPU is FSDP-sharded (`fsdp_shard_size = #GPUs`), single-node.
- **Auto-resume.** A fixed `IMAGINAIRE_OUTPUT_ROOT` + job name lets the DCP checkpointer resume from
  `latest_checkpoint.txt`; `checkpoint.load_training_state=False` inits from the cached pretrained 2B.
- The launcher intentionally omits node-specific free-GPU picking / watchdog restarts; drive those
  from your own scheduler if needed.

## Zero-shot inference

Needs the DreamDojo environment and a **≥45GB GPU** (2B teacher peaks ~31GB; a 24GB 4090 is
insufficient for the WAN-VAE decode). `cp_size=1`, so multi-GPU does not help.

```bash
export DD_CKPT_DIR=/PATH_TO/hf_ckpt/2B_GR1_post-train
export SAVE_DIR=/PATH_TO/aoe_runs/dreamdojo_aoe
export CUDA_VISIBLE_DEVICES=0
./scripts/aoe_train --dry-run infer   # prints the upstream inference command
./scripts/aoe_train infer
```

## Results

- **Zero-shot (GR1 robot post-train weights):** PSNR `10.01` / SSIM `0.214` / LPIPS `0.702` — scene +
  hands qualitatively preserved, pixel fidelity low (a **robot→human domain gap**, not a data-format
  problem). This is the motivation for the MANO post-train.
- **MANO post-train:** provided as a reproducible harness (`build-egodex` → `posttrain`); run it on
  your own hardware and evaluate against your target.

## Notes

- 480×640 is mandatory (position encodings are hard-coded; lower resolution NaNs/black-screens).
- Each clip must have **> `--num-frames`** frames (else the sampler raises `empty range for randrange`).
- The patched loader uses `decord`; keep it installed in the upstream env.
- The MANO route ships **no** MANO model file: finger kinematics use the standard MANO joint-parent
  tree applied analytically (`aoe_to_dd_mano.py`), so nothing license-gated is included.

## License

Apache-2.0 (Open-AoE Contributors) for the Open-AoE converter/launcher. The upstream
DreamDojo / Cosmos-Predict2.5 is NVIDIA's (Apache-2.0) and is referenced by checkout (not vendored);
the patch modifies three upstream files (a checkpoint-revision pin; a data-loader change that injects
AoE hand actions + swaps the video decoder; and an action-embedder weight-init fix). Model weights
are downloaded from
NVIDIA/HuggingFace under their own licenses and are not included. See the root
[`LEGAL.md`](../../LEGAL.md) and the upstream license for terms.

## File reference

| File | Purpose |
|------|---------|
| `scripts/aoe_train` | Launcher: `convert` / `infer` (zero-shot) · `build-egodex` / `posttrain` (MANO) |
| `scripts/aoe_to_dreamdojo.py` | AoE undistorted video → 480×640 MP4 (zero-shot generic `VideoDataset`) |
| `scripts/aoe_to_dd_mano.py` | AoE `hands.npz` → egodex rot6d HDF5 + paired mp4 (MANO post-train), analytic FK |
| `patches/open_aoe_support.patch` | 3-file upstream patch (checkpoint revision + MANO `[220:352]` injection/decord + action-embedder init fix) |

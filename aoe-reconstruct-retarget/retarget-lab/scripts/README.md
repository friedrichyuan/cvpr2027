# Script layout

The `scripts/` root is reserved for production orchestration, input adapters,
route selection, and backend entrypoints.

- `diagnostics/`: optional dataset screening and trajectory inspection.
- `review/`: RGB overlays, mesh visualizations, contact sheets, triptychs, and
  review-pool summaries.
- `maintenance/`: pinned third-party source setup. It changes the local
  execution environment and is not an ordinary retarget entrypoint.

Run tools from the repository root, for example:

```bash
python scripts/diagnostics/screen_aoe_retarget_candidates.py --help
python scripts/review/compose_triptych.py --help
```

On a newly configured third-party machine, run this before any GPU job:

```bash
cp local_env.example.sh local_env.sh
source local_env.sh
python3 scripts/check_third_party_config.py --json third_party_config_report.json
```

Production runners use the configured executables, dataset root, and local
`third_party/` paths.

Production entrypoints remain directly under `scripts/`, including
`run_fresh_aoe_auto_window.py`,
`run_fresh_aoe_scene_12_demos.sh`, `run_full_12_demos.sh`,
`run_v4_two_full_pipelines.sh`, `run_do_as_i_do_official_retarget.sh`, and
`run_spider_retarget.sh`.

`reuse_v4_for_12_demos.py` reports all 12 matrix cells by default. Pass one of
its 12 exact `--cell` keys, or provide `--trajectory-6dof`, `--hand-source`,
and `--retargeting` together, to select one combination through the same reuse
path. DAI and SPIDER expose four native bindings each; EgoInfinity/G1 exposes
only its native estimated-hand/Ego-trajectory binding. Unsupported cells stay
explicitly unavailable rather than borrowing another cell's video.

## AoE input modes

Both modes enter the same reconstruction and DAI/SPIDER retarget pipeline.

- Automatic window selection: `run_fresh_aoe_auto_window.py` tests the
  configured durations longest-first with the unchanged SAM3 prompt/mask
  gates, then launches one formal full run with the first passing window.
- Explicit input window: use `--window-selection-mode direct` together with
  `--direct-duration-sec` and pass a manually reviewed `--ref-source-frame`
  after `--`. This bypasses automatic window selection. It does not bypass
  reconstruction or the native SAM3/SAM3D/DAI/SPIDER execution.

Example automatic invocation:

```bash
python scripts/run_fresh_aoe_auto_window.py \
  --runner scripts/run_fresh_aoe_scene_12_demos.sh \
  --experiments-root experiments \
  --source-run cooking_wine_auto_source \
  --matrix-run cooking_wine_auto_matrix \
  --window-selection-mode auto \
  --candidate-durations-sec 3.0,2.6,2.4,2.2,2.0 \
  --manifest experiments/cooking_wine_auto_window.json -- \
  --scene cooking_wine_bottle --segment SEGMENT --annotation-id 6 \
  --object-name "cooking wine bottle" --task cooking_wine_bottle_left \
  --hand-type left --anchor-hand left --ref-source-frame 411
```

Example explicit-window invocation:

```bash
python scripts/run_fresh_aoe_auto_window.py \
  --runner scripts/run_fresh_aoe_scene_12_demos.sh \
  --experiments-root experiments \
  --source-run cooking_wine_direct_source \
  --matrix-run cooking_wine_direct_matrix \
  --window-selection-mode direct --direct-duration-sec 2.4 \
  --manifest experiments/cooking_wine_direct_window.json -- \
  --scene cooking_wine_bottle --segment SEGMENT --annotation-id 6 \
  --object-name "cooking wine bottle" --task cooking_wine_bottle_left \
  --hand-type left --anchor-hand left --ref-source-frame 411
```

Candidate screening is an optional diagnostic step. It is kept under
`scripts/diagnostics/` because the automatic and explicit production entries
do not depend on a precomputed candidate list.

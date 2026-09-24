#!/usr/bin/env bash
# Run many episodes on Ray. One GPU pool each for segment, inpaint, depth, and action.
# Composite and curation run on CPU. The VLM audit is not called.
# Pool sizes must add up to the number of GPUs. Finished stages are skipped.
# Usage: scripts/batch.sh <episode.hdf5 | dataset-dir> [more paths...]
# Optional flags are passed through, for example --inflight 8 --limit 40
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale.batch "$@"

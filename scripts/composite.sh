#!/usr/bin/env bash
# Render the robot and composite it onto the inpainted frames.
# Needs the inpaint, masks, depth, gripper, base, IK, and approach outputs.
# Writes visual/composite.mp4.
# Usage: scripts/composite.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.step import ROOT, run
from egowhale.visual.composite import Composite
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (Composite,))
' "$@"

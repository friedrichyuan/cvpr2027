"""Shared paths and timing constants, kept independent of visualization."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"
DEFAULT_EPISODE = Path("/home/ymq/code/EGODEX_DATASET/test/stack/0.hdf5")
EGODEX_FPS = 30.0

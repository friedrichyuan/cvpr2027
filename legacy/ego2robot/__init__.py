"""EgoDex Path A → robot-format data. Manager orchestrates registered processors."""

from .manager import PipelineManager
from .processor import Artifacts, EpisodeContext, Processor, ResourceSpec
from .registry import PATH_A_EGODEX, build_registry

__all__ = [
    "Artifacts",
    "EpisodeContext",
    "PATH_A_EGODEX",
    "PipelineManager",
    "Processor",
    "ResourceSpec",
    "build_registry",
]

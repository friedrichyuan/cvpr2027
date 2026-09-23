"""Register processors, enable a profile, run them in dependency order."""

from __future__ import annotations

from collections import defaultdict, deque

from .processor import EpisodeContext, Processor
from .store import ArtifactStore


class Registry:
    def __init__(self) -> None:
        self._processors: dict[str, Processor] = {}

    def register(self, processor: Processor) -> None:
        self._processors[processor.name] = processor

    def get(self, name: str) -> Processor:
        if name not in self._processors:
            raise KeyError(f"Unknown processor '{name}'. Known: {sorted(self._processors)}")
        return self._processors[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._processors)


class PipelineManager:
    def __init__(self, registry: Registry, store: ArtifactStore, runtime) -> None:
        self.registry = registry
        self.store = store
        self.runtime = runtime
        self._enabled: list[str] = []

    def enable(self, names: list[str]) -> None:
        self._enabled = list(names)
        self.schedule()  # fail fast on bad graphs

    def schedule(self) -> list[Processor]:
        processors = [self.registry.get(name) for name in self._enabled]
        return _topo_sort(processors)

    def run_episode(self, ctx: EpisodeContext) -> None:
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        for processor in self.schedule():
            if self.store.is_done(ctx, processor.name):
                continue
            missing = [
                artifact
                for artifact in processor.requires
                if not self.store.exists(ctx, artifact)
            ]
            if missing:
                raise FileNotFoundError(
                    f"{processor.name} missing required artifacts {missing} for {ctx.key}"
                )
            self.runtime.run(processor, ctx)
            for artifact in processor.produces:
                if not self.store.exists(ctx, artifact):
                    raise FileNotFoundError(
                        f"{processor.name} did not write {artifact} for {ctx.key}"
                    )
            self.store.mark_done(ctx, processor.name)


def _topo_sort(processors: list[Processor]) -> list[Processor]:
    produced_by = {}
    for processor in processors:
        for artifact in processor.produces:
            if artifact in produced_by:
                raise ValueError(
                    f"Artifact '{artifact}' produced by both "
                    f"'{produced_by[artifact]}' and '{processor.name}'"
                )
            produced_by[artifact] = processor.name

    names = {processor.name for processor in processors}
    incoming: dict[str, int] = {processor.name: 0 for processor in processors}
    edges: dict[str, list[str]] = defaultdict(list)
    for processor in processors:
        deps = set()
        for artifact in processor.requires:
            producer = produced_by.get(artifact)
            if producer is None:
                raise ValueError(
                    f"'{processor.name}' requires '{artifact}', which no enabled processor produces"
                )
            if producer == processor.name:
                continue
            if producer not in names:
                raise ValueError(f"'{processor.name}' depends on disabled '{producer}'")
            deps.add(producer)
        incoming[processor.name] = len(deps)
        for dep in deps:
            edges[dep].append(processor.name)

    ready = deque(name for name, count in incoming.items() if count == 0)
    ordered: list[str] = []
    while ready:
        name = ready.popleft()
        ordered.append(name)
        for child in edges[name]:
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    if len(ordered) != len(processors):
        raise ValueError("Processor graph has a cycle")
    by_name = {processor.name: processor for processor in processors}
    return [by_name[name] for name in ordered]

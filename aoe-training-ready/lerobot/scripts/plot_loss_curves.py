# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


LOSS_RE = re.compile(r"([A-Za-z0-9_./-]*loss[A-Za-z0-9_./-]*)\s*[:=]\s*([-+0-9.eE]+)")
STEP_RE = re.compile(r"(?:step|global_step|iter|iteration)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?[KMBkmb]?)")
TQDM_STEP_RE = re.compile(r"\|\s*([0-9]+)\s*/\s*[0-9]+")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract AoE loss curves from jsonl or text logs.")
    parser.add_argument("--input", required=True, help="Log file, jsonl file, or directory of logs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="train")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = collect_metrics(_iter_files(input_path))
    if not metrics:
        print(f"[loss_curves][warn] no loss metrics found in {input_path}")
        return 1

    summary_path = output_dir / f"{args.name}_loss_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "step", "value"])
        for metric, points in sorted(metrics.items()):
            for step, value in points:
                writer.writerow([metric, step, value])

    try:
        plot_metrics(metrics, output_dir / f"{args.name}_loss_curves.png")
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[loss_curves][warn] plotting failed: {exc}")

    print(f"[loss_curves] wrote {summary_path}")
    return 0


def _iter_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix in {".log", ".txt", ".jsonl", ".out"}
        )
    return [path]


def collect_metrics(paths: Iterable[Path]) -> dict[str, list[tuple[int, float]]]:
    explicit_metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)
    fallback_metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)
    fallback_step = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                fallback_step += 1
                step, explicit_step = _line_step(line, fallback_step)
                for metric, value in _line_metrics(line):
                    if math.isfinite(value):
                        normalized = _normalize_metric(metric)
                        if explicit_step:
                            explicit_metrics[normalized].append((step, value))
                        else:
                            fallback_metrics[normalized].append((step, value))
    metrics = {metric: _dedupe_points(points) for metric, points in explicit_metrics.items()}
    for metric, points in fallback_metrics.items():
        if metric not in metrics:
            metrics[metric] = _dedupe_points(points)
    return dict(metrics)


def _line_step(line: str, fallback_step: int) -> tuple[int, bool]:
    tqdm_match = TQDM_STEP_RE.search(line)
    if tqdm_match and "loss" in line.lower():
        return int(tqdm_match.group(1)), True
    match = STEP_RE.search(line)
    if match:
        return _parse_step_token(match.group(1)), True
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return fallback_step, False
    if isinstance(payload, dict):
        for key in ("step", "global_step", "iter", "iteration"):
            if key in payload:
                return _parse_step_token(payload[key]), True
    return fallback_step, False


def _parse_step_token(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    token = str(value).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMBkmb]?)", token)
    if not match:
        return int(float(token))
    number = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(number * multiplier)


def _line_metrics(line: str) -> list[tuple[str, float]]:
    metrics: list[tuple[str, float]] = []
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key, value in _flatten(payload).items():
            if "loss" in key.lower() and isinstance(value, int | float):
                metrics.append((key, float(value)))
    for name, raw_value in LOSS_RE.findall(line):
        try:
            metrics.append((name, float(raw_value)))
        except ValueError:
            continue
    return metrics


def _flatten(payload: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def _normalize_metric(name: str) -> str:
    return name.strip().replace("/", "_").replace(".", "_")


def _dedupe_points(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    by_step: dict[int, float] = {}
    for step, value in points:
        by_step[step] = value
    return sorted(by_step.items())


def plot_metrics(metrics: dict[str, list[tuple[int, float]]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    for metric, points in sorted(metrics.items()):
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, linewidth=1.2, label=metric)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

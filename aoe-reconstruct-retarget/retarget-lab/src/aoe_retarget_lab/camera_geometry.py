from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


CAMERA_INTRINSICS_BINDING_SCHEMA_VERSION = 1
CAMERA_INTRINSICS_BINDING_POLICY = "normalized_pinhole_intrinsics_match"
CAMERA_INTRINSICS_NORMALIZATION_POLICY = "explicit_ray_depth_intrinsics_normalization"
CAMERA_FOCAL_RELATIVE_TOLERANCE = 0.05
CAMERA_PRINCIPAL_POINT_NORMALIZED_TOLERANCE = 0.01
CAMERA_ASPECT_RELATIVE_TOLERANCE = 0.01


def _canonical_camera_intrinsics(
    value: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return None, ["camera_entry_missing_or_invalid"]
    raw = value.get("intrinsics") if isinstance(value.get("intrinsics"), Mapping) else value
    model = str(raw.get("model") or "pinhole")
    if model != "pinhole":
        errors.append(f"camera_model_unsupported:{model}")

    numeric: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        try:
            number = float(raw.get(key))
        except (TypeError, ValueError):
            errors.append(f"camera_{key}_missing_or_invalid")
            continue
        if not np.isfinite(number):
            errors.append(f"camera_{key}_nonfinite")
        elif key in {"fx", "fy"} and number <= 0.0:
            errors.append(f"camera_{key}_nonpositive")
        else:
            numeric[key] = number

    dimensions: dict[str, int] = {}
    for key in ("width", "height"):
        try:
            number = int(raw.get(key))
        except (TypeError, ValueError):
            errors.append(f"camera_{key}_missing_or_invalid")
            continue
        if number <= 0:
            errors.append(f"camera_{key}_nonpositive")
        else:
            dimensions[key] = number

    if errors:
        return None, errors
    return {
        "model": "pinhole",
        "width": dimensions["width"],
        "height": dimensions["height"],
        "fx": numeric["fx"],
        "fy": numeric["fy"],
        "cx": numeric["cx"],
        "cy": numeric["cy"],
    }, []


def camera_intrinsics_fingerprint(value: Mapping[str, Any]) -> str:
    canonical, errors = _canonical_camera_intrinsics(value)
    if canonical is None:
        raise ValueError("invalid camera intrinsics: " + ",".join(errors))
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _camera_binding_entry(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    canonical, errors = _canonical_camera_intrinsics(value)
    source = (
        dict(value.get("source"))
        if isinstance(value, Mapping) and isinstance(value.get("source"), Mapping)
        else {}
    )
    if canonical is None:
        prefixed = [f"{label}_{item}" for item in errors]
        return {
            "status": "invalid",
            "source": source,
            "intrinsics": None,
            "fingerprint": None,
            "errors": prefixed,
        }, prefixed
    return {
        "status": "ok",
        "source": source,
        "intrinsics": canonical,
        "fingerprint": camera_intrinsics_fingerprint(canonical),
        "errors": [],
    }, []


def build_camera_intrinsics_binding(
    object_camera: Any,
    hand_camera: Any,
) -> dict[str, Any]:
    """Describe whether two pinhole cameras match after image-size normalization."""

    object_entry, object_errors = _camera_binding_entry(
        object_camera, label="object_camera"
    )
    hand_entry, hand_errors = _camera_binding_entry(hand_camera, label="hand_camera")
    errors = [*object_errors, *hand_errors]
    comparison: dict[str, Any] = {
        "comparable": False,
        "matches": False,
        "thresholds": {
            "max_relative_focal_error": CAMERA_FOCAL_RELATIVE_TOLERANCE,
            "max_normalized_principal_point_error": CAMERA_PRINCIPAL_POINT_NORMALIZED_TOLERANCE,
            "max_relative_aspect_error": CAMERA_ASPECT_RELATIVE_TOLERANCE,
        },
    }
    object_intrinsics = object_entry.get("intrinsics")
    hand_intrinsics = hand_entry.get("intrinsics")
    if isinstance(object_intrinsics, Mapping) and isinstance(hand_intrinsics, Mapping):
        object_fx = float(object_intrinsics["fx"]) / float(object_intrinsics["width"])
        object_fy = float(object_intrinsics["fy"]) / float(object_intrinsics["height"])
        hand_fx = float(hand_intrinsics["fx"]) / float(hand_intrinsics["width"])
        hand_fy = float(hand_intrinsics["fy"]) / float(hand_intrinsics["height"])
        focal_errors = {
            "x": abs(object_fx - hand_fx) / max(abs(hand_fx), 1.0e-12),
            "y": abs(object_fy - hand_fy) / max(abs(hand_fy), 1.0e-12),
        }
        principal_errors = {
            "x": abs(
                float(object_intrinsics["cx"]) / float(object_intrinsics["width"])
                - float(hand_intrinsics["cx"]) / float(hand_intrinsics["width"])
            ),
            "y": abs(
                float(object_intrinsics["cy"]) / float(object_intrinsics["height"])
                - float(hand_intrinsics["cy"]) / float(hand_intrinsics["height"])
            ),
        }
        object_aspect = float(object_intrinsics["width"]) / float(
            object_intrinsics["height"]
        )
        hand_aspect = float(hand_intrinsics["width"]) / float(
            hand_intrinsics["height"]
        )
        aspect_error = abs(object_aspect - hand_aspect) / max(
            abs(hand_aspect), 1.0e-12
        )
        matches = bool(
            max(focal_errors.values()) <= CAMERA_FOCAL_RELATIVE_TOLERANCE
            and max(principal_errors.values())
            <= CAMERA_PRINCIPAL_POINT_NORMALIZED_TOLERANCE
            and aspect_error <= CAMERA_ASPECT_RELATIVE_TOLERANCE
        )
        comparison.update(
            {
                "comparable": True,
                "matches": matches,
                "relative_focal_error": focal_errors,
                "normalized_principal_point_error": principal_errors,
                "relative_aspect_error": aspect_error,
            }
        )
        if not matches:
            errors.append("camera_intrinsics_mismatch")

    return {
        "schema_version": CAMERA_INTRINSICS_BINDING_SCHEMA_VERSION,
        "policy": CAMERA_INTRINSICS_BINDING_POLICY,
        "status": "invalid" if errors else "ok",
        "geometry_modified": False,
        "object_camera": object_entry,
        "hand_camera": hand_entry,
        "comparison": comparison,
        "errors": errors,
    }


def camera_ray_depth_transform(
    source_camera: Mapping[str, Any],
    target_camera: Mapping[str, Any],
) -> np.ndarray:
    """Preserve source pixels and metric depth under a pinhole-camera change."""

    source, source_errors = _canonical_camera_intrinsics(source_camera)
    target, target_errors = _canonical_camera_intrinsics(target_camera)
    if source is None or target is None:
        raise ValueError(
            "invalid camera intrinsics: "
            + ",".join([*source_errors, *target_errors])
        )
    return np.array(
        [
            [
                float(source["fx"]) / float(target["fx"]),
                0.0,
                (float(source["cx"]) - float(target["cx"])) / float(target["fx"]),
            ],
            [
                0.0,
                float(source["fy"]) / float(target["fy"]),
                (float(source["cy"]) - float(target["cy"])) / float(target["fy"]),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

"""Stable schema contract for off-policy runtime manifests.

TensorBoard and W&B metrics have their own schema version in
:mod:`uni_rl.logging.metric_schema`.  Runtime manifests describe process,
device, IPC, and lifecycle state; the two version namespaces must not be
merged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

RUNTIME_MANIFEST_SCHEMA_VERSION = 1

_STABLE_NONNEGATIVE_INT_FIELDS = (
    ("inference_flight", "queue_depth"),
    ("inference_flight", "publication_lag"),
    ("inference_flight", "max_in_flight"),
    ("inference_flight", "max_publication_lag"),
    ("replay_ingress", "ingress_depth"),
    ("replay_ingress", "ingress_slot_rows"),
    ("replay_ingress", "published_sequence"),
    ("replay_ingress", "release_sequence"),
    ("replay_ingress", "occupancy"),
    ("replay_ingress", "high_water_occupancy"),
    ("replay_ingress", "backpressure_waits"),
    ("replay_ingress", "early_returns"),
    ("replay_ingress", "dropped_batches"),
    ("replay_ingress", "closed_returns"),
    ("replay_ingress", "stop_returns"),
    ("inference_memory_budget", "ipc_event_count"),
)
_COMPLETED_REQUIRED_SECTIONS = (
    "inference_flight",
    "replay_ingress",
)


def validate_runtime_manifest(
    manifest: Mapping[str, Any],
    *,
    completed: bool = False,
) -> None:
    """Validate the stable v1 contract without constraining diagnostics.

    Fields not declared here remain producer diagnostics and carry no cross-run
    compatibility promise.  ``shutdown`` is intentionally opaque in v1.  A
    missing stable lifecycle section is allowed only for early failure
    summaries; pass ``completed=True`` for normal final summaries.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("runtime_manifest must be a mapping")
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("runtime_manifest.schema_version must be an integer")
    if schema_version != RUNTIME_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported runtime_manifest.schema_version "
            f"{schema_version}; expected {RUNTIME_MANIFEST_SCHEMA_VERSION}"
        )

    _validate_positive_int(manifest, "inference_ring_capacity", required=completed)
    _validate_positive_int(manifest, "collector_metrics_interval", required=completed)
    _validate_optional_device(manifest, "collector_backend_device", required=completed)

    for section_name in _COMPLETED_REQUIRED_SECTIONS:
        section = manifest.get(section_name)
        if section is None and not completed:
            continue
        if not isinstance(section, Mapping):
            error = "must be a mapping" if section is not None else "is required"
            raise ValueError(f"completed runtime_manifest.{section_name} {error}")

    for section_name, field_name in _STABLE_NONNEGATIVE_INT_FIELDS:
        section = manifest.get(section_name)
        if not isinstance(section, Mapping):
            continue
        value = section.get(field_name)
        if value is None:
            raise ValueError(f"runtime_manifest.{section_name}.{field_name} is required")
        _require_nonnegative_int(
            value,
            f"runtime_manifest.{section_name}.{field_name}",
        )

    if isinstance(manifest.get("replay_ingress"), Mapping):
        value = manifest["replay_ingress"].get("backpressure_wait_s")
        if value is None:
            raise ValueError("runtime_manifest.replay_ingress.backpressure_wait_s is required")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                "runtime_manifest.replay_ingress.backpressure_wait_s "
                "must be a finite non-negative number"
            )
        if value < 0:
            raise ValueError(
                "runtime_manifest.replay_ingress.backpressure_wait_s must be non-negative"
            )


def _validate_positive_int(
    manifest: Mapping[str, Any],
    field_name: str,
    *,
    required: bool,
) -> None:
    value = manifest.get(field_name)
    if value is None and not required:
        return
    if value is None:
        raise ValueError(f"runtime_manifest.{field_name} is required")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"runtime_manifest.{field_name} must be a positive integer")


def _validate_optional_device(
    manifest: Mapping[str, Any],
    field_name: str,
    *,
    required: bool,
) -> None:
    if field_name not in manifest:
        if required:
            raise ValueError(f"runtime_manifest.{field_name} is required")
        return
    value = manifest.get(field_name)
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime_manifest.{field_name} must be a non-empty string or null")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")

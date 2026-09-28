"""Shared resolution for task-owner tensor-native collection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omegaconf import DictConfig, OmegaConf

DEFAULT_INFERENCE_SLOT_CAPACITY = 1
MAX_INFERENCE_SLOT_CAPACITY = 16
DEFAULT_COLLECTOR_METRICS_INTERVAL = 1
MAX_COLLECTOR_METRICS_INTERVAL = 10_000
DEFAULT_REPLAY_INGRESS_DEPTH = 2
MAX_REPLAY_INGRESS_DEPTH = 16


def _validated_positive_int(
    value: object,
    *,
    label: str,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a positive integer, got {value!r}")
    if value <= 0 or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f" and no greater than {maximum}"
        raise ValueError(f"{label} must be a positive integer{suffix}, got {value!r}")
    return value


@dataclass(frozen=True)
class TensorRuntimeSettings:
    """Effective, bounded tensor-runtime values resolved before spawn."""

    inference_slot_capacity: int
    collector_metrics_interval: int
    replay_ingress_depth: int
    replay_ingress_slot_rows: int
    batch_size: int
    updates_per_step: int
    num_envs: int
    configured_batch_size: int | None = None
    configured_updates_per_step: int | None = None
    configured_inference_slot_capacity: int | None = None
    configured_collector_metrics_interval: int | None = None
    configured_replay_ingress_depth: int | None = None
    configured_replay_ingress_slot_rows: int | None = None

    def __post_init__(self) -> None:
        num_envs = _validated_positive_int(self.num_envs, label="algo.num_envs")
        inference_capacity = _validated_positive_int(
            self.inference_slot_capacity,
            label="training.inference_slot_capacity",
            maximum=MAX_INFERENCE_SLOT_CAPACITY,
        )
        metrics_interval = _validated_positive_int(
            self.collector_metrics_interval,
            label="training.collector_metrics_interval",
            maximum=MAX_COLLECTOR_METRICS_INTERVAL,
        )
        ingress_depth = _validated_positive_int(
            self.replay_ingress_depth,
            label="training.replay_ingress_depth",
            maximum=MAX_REPLAY_INGRESS_DEPTH,
        )
        ingress_slot_rows = _validated_positive_int(
            self.replay_ingress_slot_rows,
            label="training.replay_ingress_slot_rows",
            maximum=num_envs,
        )
        batch_size = _validated_positive_int(self.batch_size, label="algo.batch_size")
        updates_per_step = _validated_positive_int(
            self.updates_per_step,
            label="algo.updates_per_step",
        )
        object.__setattr__(self, "inference_slot_capacity", inference_capacity)
        object.__setattr__(self, "collector_metrics_interval", metrics_interval)
        object.__setattr__(self, "replay_ingress_depth", ingress_depth)
        object.__setattr__(self, "replay_ingress_slot_rows", ingress_slot_rows)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "updates_per_step", updates_per_step)
        configured_values = (
            (
                "inference_slot_capacity",
                self.configured_inference_slot_capacity,
                inference_capacity,
            ),
            (
                "collector_metrics_interval",
                self.configured_collector_metrics_interval,
                metrics_interval,
            ),
            ("replay_ingress_depth", self.configured_replay_ingress_depth, ingress_depth),
            (
                "replay_ingress_slot_rows",
                self.configured_replay_ingress_slot_rows,
                ingress_slot_rows,
            ),
            ("batch_size", self.configured_batch_size, batch_size),
            ("updates_per_step", self.configured_updates_per_step, updates_per_step),
        )
        for name, configured, effective in configured_values:
            if configured is None:
                continue
            configured_int = _validated_positive_int(
                configured,
                label=f"configured {name}",
            )
            if configured_int != effective:
                raise ValueError(
                    f"configured {name} must match its effective value "
                    f"({configured_int!r} != {effective!r})"
                )

    @property
    def learner_sample_count(self) -> int:
        return self.batch_size * self.updates_per_step

    def manifest(self) -> dict[str, dict[str, object]]:
        return {
            "inference_ring_capacity": {
                "configured": self.configured_inference_slot_capacity,
                "default": DEFAULT_INFERENCE_SLOT_CAPACITY,
                "effective": self.inference_slot_capacity,
                "maximum": MAX_INFERENCE_SLOT_CAPACITY,
            },
            "collector_metrics_interval": {
                "configured": self.configured_collector_metrics_interval,
                "default": DEFAULT_COLLECTOR_METRICS_INTERVAL,
                "effective": self.collector_metrics_interval,
                "maximum": MAX_COLLECTOR_METRICS_INTERVAL,
            },
            "replay_ingress_depth": {
                "configured": self.configured_replay_ingress_depth,
                "default": DEFAULT_REPLAY_INGRESS_DEPTH,
                "effective": self.replay_ingress_depth,
                "maximum": MAX_REPLAY_INGRESS_DEPTH,
            },
            "replay_ingress_slot_rows": {
                "configured": self.configured_replay_ingress_slot_rows,
                "default": "algo.num_envs",
                "effective": self.replay_ingress_slot_rows,
                "maximum": self.num_envs,
            },
            "learner_sampling": {
                "configured_batch_size": self.configured_batch_size,
                "configured_updates_per_step": self.configured_updates_per_step,
                "configured_rows_per_sync": (
                    self.configured_batch_size * self.configured_updates_per_step
                    if (
                        self.configured_batch_size is not None
                        and self.configured_updates_per_step is not None
                    )
                    else None
                ),
                "default": "required owner configuration",
                "effective_batch_size": self.batch_size,
                "effective_updates_per_step": self.updates_per_step,
                "effective_rows_per_sync": self.learner_sample_count,
                "maximum": "device-memory budget",
            },
        }


def resolve_collector_tensor_native(
    cfg: DictConfig,
    *,
    device: str,
    algo_name: str,
) -> bool:
    """Resolve a task owner's explicit tensor capability before spawning."""
    value = OmegaConf.select(cfg, "env.tensor_runtime", default=False)
    if value is None:
        value = False
    if type(value) is not bool:
        raise TypeError(
            f"{algo_name} env.tensor_runtime must be a boolean or omitted, got {value!r}"
        )
    if value and torch.device(device).type != "cuda":
        raise ValueError(
            f"{algo_name} env.tensor_runtime=true requires CUDA, "
            f"but the replay device is {device!r}"
        )
    return value


def _resolve_positive_training_int(
    cfg: DictConfig,
    *,
    key: str,
    default: int,
    algo_name: str,
    maximum: int | None = None,
) -> int:
    value = OmegaConf.select(cfg, f"training.{key}", default=default)
    if value is None:
        return default
    if type(value) is not int:
        raise TypeError(
            f"{algo_name} training.{key} must be a positive integer or omitted, got {value!r}"
        )
    suffix = "" if maximum is None else f" and no greater than {maximum}"
    if value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(
            f"{algo_name} training.{key} must be a positive integer{suffix}, got {value!r}"
        )
    return value


def resolve_inference_slot_capacity(cfg: DictConfig, *, algo_name: str) -> int:
    """Resolve the bounded inference-ring capacity before IPC construction."""
    return _resolve_positive_training_int(
        cfg,
        key="inference_slot_capacity",
        default=DEFAULT_INFERENCE_SLOT_CAPACITY,
        algo_name=algo_name,
        maximum=MAX_INFERENCE_SLOT_CAPACITY,
    )


def resolve_collector_metrics_interval(cfg: DictConfig, *, algo_name: str) -> int:
    """Resolve the device-metric compaction/reporting interval."""
    return _resolve_positive_training_int(
        cfg,
        key="collector_metrics_interval",
        default=DEFAULT_COLLECTOR_METRICS_INTERVAL,
        algo_name=algo_name,
        maximum=MAX_COLLECTOR_METRICS_INTERVAL,
    )


def _resolved_training_int(
    cfg: DictConfig,
    *,
    key: str,
    default: int,
    algo_name: str,
    maximum: int | None = None,
) -> tuple[int, int | None]:
    value = OmegaConf.select(cfg, f"training.{key}", default=default)
    if value is None:
        return default, None
    effective = _validated_positive_int(
        value,
        label=f"{algo_name} training.{key}",
        maximum=maximum,
    )
    return effective, effective


def _required_algo_int(cfg: DictConfig, *, key: str, algo_name: str) -> int:
    value = OmegaConf.select(cfg, f"algo.{key}")
    if value is None:
        raise ValueError(
            f"{algo_name} algo.{key} must be configured as a positive integer, got None"
        )
    return _validated_positive_int(value, label=f"{algo_name} algo.{key}")


def resolve_tensor_runtime_settings(
    cfg: DictConfig,
    *,
    algo_name: str,
    num_envs: int,
) -> TensorRuntimeSettings:
    """Resolve all spawn-facing tensor-runtime knobs and learner sampling."""
    inference_capacity, configured_inference_capacity = _resolved_training_int(
        cfg,
        key="inference_slot_capacity",
        default=DEFAULT_INFERENCE_SLOT_CAPACITY,
        algo_name=algo_name,
        maximum=MAX_INFERENCE_SLOT_CAPACITY,
    )
    metrics_interval, configured_metrics_interval = _resolved_training_int(
        cfg,
        key="collector_metrics_interval",
        default=DEFAULT_COLLECTOR_METRICS_INTERVAL,
        algo_name=algo_name,
        maximum=MAX_COLLECTOR_METRICS_INTERVAL,
    )
    ingress_depth, configured_ingress_depth = _resolved_training_int(
        cfg,
        key="replay_ingress_depth",
        default=DEFAULT_REPLAY_INGRESS_DEPTH,
        algo_name=algo_name,
        maximum=MAX_REPLAY_INGRESS_DEPTH,
    )
    ingress_slot_rows, configured_ingress_slot_rows = _resolved_training_int(
        cfg,
        key="replay_ingress_slot_rows",
        default=num_envs,
        algo_name=algo_name,
    )
    batch_size = _required_algo_int(cfg, key="batch_size", algo_name=algo_name)
    updates_per_step = _required_algo_int(cfg, key="updates_per_step", algo_name=algo_name)
    return TensorRuntimeSettings(
        inference_slot_capacity=inference_capacity,
        collector_metrics_interval=metrics_interval,
        replay_ingress_depth=ingress_depth,
        replay_ingress_slot_rows=ingress_slot_rows,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
        num_envs=num_envs,
        configured_batch_size=batch_size,
        configured_updates_per_step=updates_per_step,
        configured_inference_slot_capacity=configured_inference_capacity,
        configured_collector_metrics_interval=configured_metrics_interval,
        configured_replay_ingress_depth=configured_ingress_depth,
        configured_replay_ingress_slot_rows=configured_ingress_slot_rows,
    )

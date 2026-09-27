"""Shared resolution for task-owner tensor-native collection."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf


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
) -> int:
    value = OmegaConf.select(cfg, f"training.{key}", default=default)
    if value is None:
        value = default
    if type(value) is not int:
        raise TypeError(
            f"{algo_name} training.{key} must be a positive integer or omitted, got {value!r}"
        )
    if value <= 0:
        raise ValueError(f"{algo_name} training.{key} must be a positive integer, got {value!r}")
    return int(value)


def resolve_inference_slot_capacity(cfg: DictConfig, *, algo_name: str) -> int:
    """Resolve the bounded inference-ring capacity before IPC construction."""
    return _resolve_positive_training_int(
        cfg,
        key="inference_slot_capacity",
        default=1,
        algo_name=algo_name,
    )


def resolve_collector_metrics_interval(cfg: DictConfig, *, algo_name: str) -> int:
    """Resolve the device-metric compaction/reporting interval."""
    return _resolve_positive_training_int(
        cfg,
        key="collector_metrics_interval",
        default=1,
        algo_name=algo_name,
    )

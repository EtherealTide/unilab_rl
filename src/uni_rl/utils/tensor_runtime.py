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

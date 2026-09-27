"""Resolution contracts for tensor-native off-policy runtime knobs."""

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

from uni_rl.utils.tensor_runtime import (
    resolve_collector_metrics_interval,
    resolve_collector_tensor_native,
    resolve_inference_slot_capacity,
)


def _cfg(training: dict | None = None, env: dict | None = None) -> Any:
    return OmegaConf.create({"training": training or {}, "env": env or {}})


@pytest.mark.parametrize(
    "resolver",
    [resolve_inference_slot_capacity, resolve_collector_metrics_interval],
)
def test_tensor_runtime_intervals_default_to_one(resolver) -> None:
    cfg = _cfg()

    assert resolver(cfg, algo_name="SAC") == 1
    assert resolver(_cfg({"inference_slot_capacity": None}), algo_name="SAC") == 1


@pytest.mark.parametrize(
    "resolver",
    [resolve_inference_slot_capacity, resolve_collector_metrics_interval],
)
@pytest.mark.parametrize("value", [2, 100])
def test_tensor_runtime_intervals_accept_positive_integers(resolver, value: int) -> None:
    cfg = _cfg(
        {
            "inference_slot_capacity": value,
            "collector_metrics_interval": value,
        }
    )

    assert resolver(cfg, algo_name="FlashSAC") == value


@pytest.mark.parametrize(
    "resolver",
    [resolve_inference_slot_capacity, resolve_collector_metrics_interval],
)
@pytest.mark.parametrize("value", [0, -1])
def test_tensor_runtime_intervals_reject_non_positive_integers(resolver, value: int) -> None:
    cfg = _cfg(
        {
            "inference_slot_capacity": value,
            "collector_metrics_interval": value,
        }
    )

    with pytest.raises(ValueError, match="must be a positive integer"):
        resolver(cfg, algo_name="SAC")


@pytest.mark.parametrize(
    "resolver",
    [resolve_inference_slot_capacity, resolve_collector_metrics_interval],
)
@pytest.mark.parametrize("value", [True, False, "2", 1.0])
def test_tensor_runtime_intervals_reject_non_integer_values(resolver, value) -> None:
    cfg = _cfg(
        {
            "inference_slot_capacity": value,
            "collector_metrics_interval": value,
        }
    )

    with pytest.raises(TypeError, match="must be a positive integer or omitted"):
        resolver(cfg, algo_name="SAC")


def test_collector_tensor_runtime_requires_explicit_boolean() -> None:
    with pytest.raises(TypeError, match="env.tensor_runtime must be a boolean"):
        resolve_collector_tensor_native(
            _cfg(env={"tensor_runtime": 1}), device="cuda", algo_name="SAC"
        )

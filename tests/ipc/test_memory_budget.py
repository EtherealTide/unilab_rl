from __future__ import annotations

import shutil

import pytest

from uni_rl.ipc.inference_ring import estimate_inference_ring_bytes
from uni_rl.ipc.memory_budget import (
    estimate_offpolicy_bytes,
    raise_if_shared_memory_over_budget,
)


def test_offpolicy_memory_budget_notes_native_exclusions() -> None:
    estimate = estimate_offpolicy_bytes(
        num_envs=5120,
        replay_buffer_n=1024,
        obs_dim=98,
        action_dim=29,
        critic_dim=101,
    )

    breakdown = str(estimate["breakdown"])
    assert "MuJoCo BatchEnvPool" in breakdown
    assert "CUDA pinned/shared" in breakdown
    assert "driver memory" in breakdown


def test_device_replay_host_budget_is_independent_of_replay_capacity() -> None:
    estimates = [
        estimate_offpolicy_bytes(
            num_envs=10,
            replay_buffer_n=replay_buffer_n,
            obs_dim=2,
            action_dim=1,
            critic_dim=3,
            ingress_depth=2,
        )
        for replay_buffer_n in (4, 4000)
    ]

    assert estimates[0]["replay_buffer"] == 0
    assert estimates[0]["bounded_ingress_slots"] > 0
    assert estimates[0]["total"] == estimates[1]["total"]
    assert "authoritative on the learner device" in str(estimates[0]["breakdown"])


def test_host_budget_includes_cpu_inference_ring_storage() -> None:
    estimate = estimate_offpolicy_bytes(
        num_envs=2,
        replay_buffer_n=4,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        inference_ring_bytes=168,
    )

    assert estimate["inference_ring"] == 168
    assert estimate["total"] == estimate["bounded_ingress_slots"] + 168
    assert "Inference ring" in str(estimate["breakdown"])
    assert estimate_inference_ring_bytes(2, 4, 2, capacity=3) == 168


def test_shared_memory_budget_unknown_available_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: (_ for _ in ()).throw(OSError()))
    estimate = {"total": 1024, "breakdown": "test"}

    raise_if_shared_memory_over_budget(estimate, label="test", path="/missing-shm")


def test_shared_memory_budget_allows_within_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Usage:
        free = 100 * 1024

    monkeypatch.setattr(shutil, "disk_usage", lambda path: _Usage())
    estimate = {"total": 80 * 1024, "breakdown": "test"}

    raise_if_shared_memory_over_budget(estimate, label="test", threshold=0.8)


def test_shared_memory_budget_raises_before_over_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Usage:
        free = 100 * 1024

    monkeypatch.setattr(shutil, "disk_usage", lambda path: _Usage())
    estimate = {"total": 81 * 1024, "breakdown": "test"}

    with pytest.raises(MemoryError) as excinfo:
        raise_if_shared_memory_over_budget(estimate, label="Off-policy (sac)", threshold=0.8)

    message = str(excinfo.value)
    assert "Off-policy (sac)" in message
    assert "/dev/shm" in message
    assert "estimated" in message
    assert "available" in message
    assert "algo.num_envs" in message
    assert "algo.replay_buffer_n" in message

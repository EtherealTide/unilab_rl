from __future__ import annotations

import shutil

import pytest

from uni_rl.ipc.inference_ring import estimate_inference_ring_bytes
from uni_rl.ipc.memory_budget import (
    CUDA_INFERENCE_EVENT_RESERVE_BYTES,
    CUDA_INFERENCE_MIN_OVERHEAD_BYTES,
    CUDA_INFERENCE_TIMING_EVENT_COUNT,
    estimate_cuda_inference_ipc_bytes,
    estimate_offpolicy_bytes,
    raise_if_cuda_memory_over_budget,
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


def test_device_replay_host_budget_excludes_gpu_collector_ingress() -> None:
    estimate = estimate_offpolicy_bytes(
        num_envs=2,
        replay_buffer_n=4,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        ingress_on_device=True,
    )

    assert estimate["bounded_ingress_slots"] == 0
    assert estimate["total"] == 0
    assert "device-resident" in str(estimate["breakdown"])


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


def test_cuda_inference_budget_accounts_for_all_startup_categories() -> None:
    persistent_scratch = 2 * (2 * 4 + 2 * 4)
    estimate = estimate_cuda_inference_ipc_bytes(
        2,
        4,
        2,
        capacity=3,
        ring_on_device=True,
        persistent_exploration_scratch=persistent_scratch,
    )
    exact = 168 + 2 * 4 * 4 + 2 * 4 + 2 * 2 * 4 + persistent_scratch

    assert estimate["ring_storage"] == 168
    assert estimate["observation_scratch"] == 32
    assert estimate["done_scratch"] == 8
    assert estimate["action_output"] == 16
    assert estimate["persistent_exploration_scratch"] == persistent_scratch
    assert estimate["ipc_event_count"] == 9
    assert estimate["timing_event_count"] == CUDA_INFERENCE_TIMING_EVENT_COUNT
    assert estimate["cuda_event_count"] == 11
    assert estimate["cuda_events"] == 11 * CUDA_INFERENCE_EVENT_RESERVE_BYTES
    assert estimate["conservative_overhead"] == CUDA_INFERENCE_MIN_OVERHEAD_BYTES
    assert estimate["total"] == exact + estimate["cuda_events"] + estimate["conservative_overhead"]
    assert "CUDA inference IPC budget" in str(estimate["breakdown"])

    cpu_ring = estimate_cuda_inference_ipc_bytes(
        2,
        4,
        2,
        capacity=3,
        ring_on_device=False,
    )
    assert cpu_ring["ring_storage"] == 0
    assert cpu_ring["ipc_event_count"] == 0
    assert cpu_ring["cuda_events"] == (
        CUDA_INFERENCE_TIMING_EVENT_COUNT * CUDA_INFERENCE_EVENT_RESERVE_BYTES
    )


def test_cuda_budget_threshold_and_actionable_error() -> None:
    estimate = estimate_cuda_inference_ipc_bytes(2, 4, 2, capacity=3, ring_on_device=True)

    raise_if_cuda_memory_over_budget(
        estimate,
        label="Off-policy (sac)",
        available_bytes=int(estimate["total"]) * 10,
        threshold=0.8,
    )

    with pytest.raises(MemoryError) as excinfo:
        raise_if_cuda_memory_over_budget(
            estimate,
            label="Off-policy (sac)",
            available_bytes=int(estimate["total"]),
            threshold=0.8,
        )

    message = str(excinfo.value)
    assert "Off-policy (sac)" in message
    assert "conservative CUDA inference IPC budget" in message
    assert "80% limit" in message
    assert "device free" in message
    assert "training.inference_slot_capacity" in message
    assert "Inference ring storage" in message


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

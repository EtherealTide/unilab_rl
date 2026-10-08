"""Tests for the learner-owned inference single-slot contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from uni_rl.ipc.inference_slot import SharedInferenceSlot


def test_inference_slot_roundtrip_preserves_tick_and_policy_version() -> None:
    slot = SharedInferenceSlot(num_envs=2, obs_dim=3, action_dim=2)
    observations = np.arange(6, dtype=np.float32).reshape(2, 3)
    dones = np.array([0.0, 1.0], dtype=np.float32)

    slot.publish_observation(tick_id=7, observations=observations, dones=dones)
    obs_destination = torch.empty(2, 3)
    dones_destination = torch.empty(2)
    slot.copy_observation_to(
        tick_id=7,
        observations=obs_destination,
        dones=dones_destination,
    )
    slot.publish_action(
        tick_id=7,
        policy_version=11,
        actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    actions, policy_version = slot.consume_action(tick_id=7)

    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.float32
    np.testing.assert_array_equal(obs_destination.numpy(), observations)
    np.testing.assert_array_equal(dones_destination.numpy(), dones)
    np.testing.assert_array_equal(actions, [[1.0, 2.0], [3.0, 4.0]])
    assert policy_version == 11


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_inference_slot_accepts_learner_cuda_actions() -> None:
    slot = SharedInferenceSlot(num_envs=2, obs_dim=3, action_dim=2)
    observations = np.zeros((2, 3), dtype=np.float32)
    dones = np.zeros(2, dtype=np.float32)
    actions = torch.arange(4, dtype=torch.float32, device="cuda").reshape(2, 2)

    slot.publish_observation(tick_id=1, observations=observations, dones=dones)
    slot.publish_action(
        tick_id=1,
        policy_version=2,
        actions=actions,
    )
    received, policy_version = slot.consume_action(tick_id=1)

    assert isinstance(received, np.ndarray)
    assert received.dtype == np.float32
    np.testing.assert_array_equal(received, actions.cpu().numpy())
    assert policy_version == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inference_slot_publication_uses_current_stream_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_current_stream = torch.cuda.current_stream
    current_stream_devices: list[torch.device] = []

    def fail_device_sync(*args, **kwargs) -> None:
        raise AssertionError("inference publication must not synchronize the whole device")

    def tracking_current_stream(device=None, *args, **kwargs):
        current_stream_devices.append(
            torch.device(device) if device is not None else torch.device("cuda")
        )
        return original_current_stream(device, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", fail_device_sync)
    monkeypatch.setattr(torch.cuda, "current_stream", tracking_current_stream)

    slot = SharedInferenceSlot(num_envs=2, obs_dim=3, action_dim=2, device="cuda")
    observations = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
    dones = torch.tensor([0.0, 1.0], device="cuda")
    actions = torch.arange(4, dtype=torch.float32, device="cuda").reshape(2, 2)

    slot.publish_observation(tick_id=1, observations=observations, dones=dones)
    slot.copy_observation_to(
        tick_id=1,
        observations=torch.empty_like(observations),
        dones=torch.empty_like(dones),
    )
    slot.publish_action(tick_id=1, policy_version=2, actions=actions)
    received, policy_version = slot.consume_action(tick_id=1)

    torch.testing.assert_close(received, actions)
    assert policy_version == 2
    assert len(current_stream_devices) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_inference_slot_cuda_action_uses_producer_stream_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_current_stream = torch.cuda.current_stream
    current_stream_devices: list[torch.device] = []

    def fail_device_sync(*args, **kwargs) -> None:
        raise AssertionError("action publication must not synchronize the whole device")

    def tracking_current_stream(device=None, *args, **kwargs):
        current_stream_devices.append(
            torch.device(device) if device is not None else torch.device("cuda")
        )
        return original_current_stream(device, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", fail_device_sync)
    monkeypatch.setattr(torch.cuda, "current_stream", tracking_current_stream)

    slot = SharedInferenceSlot(num_envs=2, obs_dim=3, action_dim=2)
    actions = torch.arange(4, dtype=torch.float32, device="cuda").reshape(2, 2)
    slot.publish_observation(
        tick_id=1,
        observations=np.zeros((2, 3), dtype=np.float32),
        dones=np.zeros(2, dtype=np.float32),
    )
    slot.publish_action(tick_id=1, policy_version=2, actions=actions)

    assert len(current_stream_devices) == 1


def test_inference_slot_rejects_early_reuse_and_tick_mismatch() -> None:
    slot = SharedInferenceSlot(num_envs=1, obs_dim=2, action_dim=1)
    observations = np.zeros((1, 2), dtype=np.float32)
    dones = np.zeros(1, dtype=np.float32)
    slot.publish_observation(tick_id=3, observations=observations, dones=dones)

    with pytest.raises(RuntimeError, match="cannot be reused"):
        slot.publish_observation(tick_id=4, observations=observations, dones=dones)
    with pytest.raises(RuntimeError, match="tick mismatch"):
        slot.copy_observation_to(
            tick_id=4,
            observations=torch.empty(1, 2),
            dones=torch.empty(1),
        )

    slot.publish_action(tick_id=3, policy_version=5, actions=torch.ones(1, 1))
    with pytest.raises(RuntimeError, match="tick mismatch"):
        slot.consume_action(tick_id=4)

    actions, policy_version = slot.consume_action(tick_id=3)
    np.testing.assert_array_equal(actions, [[1.0]])
    assert policy_version == 5
    slot.publish_observation(tick_id=4, observations=observations, dones=dones)

"""Contracts for the bounded learner-owned inference ring."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
import pytest
import torch

from uni_rl.ipc.inference_ring import SharedInferenceRing


def _observations(tick: int) -> np.ndarray:
    return np.arange(6, dtype=np.float32).reshape(2, 3) + tick


def _dones(tick: int) -> np.ndarray:
    return np.asarray([tick % 2, (tick + 1) % 2], dtype=np.float32)


def _actions(tick: int) -> torch.Tensor:
    return torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10 * tick


def _roundtrip_one(ring: SharedInferenceRing, tick: int, policy_version: int) -> None:
    observations = _observations(tick)
    dones = _dones(tick)
    actions = _actions(tick)
    observation_destination = torch.empty((2, 3))
    done_destination = torch.empty(2)

    ring.publish_observation(
        tick_id=tick,
        observations=observations,
        dones=dones,
        epoch=0,
    )
    ring.copy_observation_to(
        tick_id=tick,
        observations=observation_destination,
        dones=done_destination,
        epoch=0,
    )
    ring.publish_action(
        tick_id=tick,
        policy_version=policy_version,
        actions=actions,
        epoch=0,
    )
    received_actions, received_version = ring.consume_action(
        tick_id=tick,
        epoch=0,
    )

    np.testing.assert_array_equal(observation_destination.numpy(), observations)
    np.testing.assert_array_equal(done_destination.numpy(), dones)
    if ring.device.type == "cuda":
        torch.testing.assert_close(received_actions, actions)
    else:
        np.testing.assert_array_equal(received_actions, actions.numpy())
    assert received_version == policy_version


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_inference_ring_sequences_contiguous_ticks_and_wraps(capacity: int) -> None:
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        capacity=capacity,
        epoch=0,
    )

    for tick in range(12):
        _roundtrip_one(ring, tick, policy_version=tick + 7)

    assert ring.diagnostics == {
        "capacity": capacity,
        "device": "cpu",
        "epoch": 0,
        "published_tick": 11,
        "observation_tick": 11,
        "consumed_tick": 11,
        "closed": False,
    }
    assert ring.nbytes == (2 * 3 + 2 + 2 * 2) * capacity * 4


def test_inference_ring_starts_at_tick_zero() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1)

    with pytest.raises(RuntimeError, match="publication tick mismatch: expected 0, got 1"):
        ring.publish_observation(
            tick_id=1,
            observations=np.zeros((1, 1), dtype=np.float32),
            dones=np.zeros(1, dtype=np.float32),
        )


def test_inference_ring_rejects_duplicate_and_skipped_publication() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=2)
    observations = np.zeros((1, 1), dtype=np.float32)
    dones = np.zeros(1, dtype=np.float32)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)

    with pytest.raises(RuntimeError, match="publication tick mismatch: expected 1, got 0"):
        ring.publish_observation(tick_id=0, observations=observations, dones=dones)
    with pytest.raises(RuntimeError, match="publication tick mismatch: expected 1, got 2"):
        ring.publish_observation(tick_id=2, observations=observations, dones=dones)


def test_inference_ring_rejects_response_before_observation_and_stale_actions() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=2)
    observations = np.zeros((1, 1), dtype=np.float32)
    dones = np.zeros(1, dtype=np.float32)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)

    with pytest.raises(RuntimeError, match="action publication tick mismatch"):
        ring.publish_action(
            tick_id=1,
            policy_version=1,
            actions=torch.ones((1, 1)),
        )

    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(tick_id=0, policy_version=3, actions=torch.ones((1, 1)))
    ring.publish_observation(tick_id=1, observations=observations, dones=dones)
    ring.copy_observation_to(
        tick_id=1,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    actions, policy_version = ring.consume_action(tick_id=0)
    np.testing.assert_array_equal(actions, [[1.0]])
    assert policy_version == 3

    with pytest.raises(RuntimeError, match="action consumption tick mismatch: expected 1, got 0"):
        ring.consume_action(tick_id=0)
    assert ring.try_consume_action(tick_id=1, timeout_sec=0.0) is None


def test_inference_ring_rejects_duplicate_action_publication() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1)
    observations = np.zeros((1, 1), dtype=np.float32)
    dones = np.zeros(1, dtype=np.float32)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)
    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(
        tick_id=0,
        policy_version=4,
        actions=torch.full((1, 1), 7.0),
    )

    with pytest.raises(RuntimeError, match="action response already published"):
        ring.publish_action(
            tick_id=0,
            policy_version=5,
            actions=torch.full((1, 1), 9.0),
        )

    actions, policy_version = ring.consume_action(tick_id=0)
    np.testing.assert_array_equal(actions, [[7.0]])
    assert policy_version == 4
    ring.publish_observation(tick_id=1, observations=observations, dones=dones)
    ring.copy_observation_to(
        tick_id=1,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(
        tick_id=1,
        policy_version=6,
        actions=torch.full((1, 1), 11.0),
    )
    actions, policy_version = ring.consume_action(tick_id=1)
    np.testing.assert_array_equal(actions, [[11.0]])
    assert policy_version == 6


def test_inference_ring_rejects_out_of_order_observation_and_action_headers() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=2)
    observations = np.zeros((1, 1), dtype=np.float32)
    dones = np.zeros(1, dtype=np.float32)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)

    with pytest.raises(
        RuntimeError, match="observation consumption tick mismatch: expected 0, got 1"
    ):
        ring.copy_observation_to(
            tick_id=1,
            observations=torch.empty((1, 1)),
            dones=torch.empty(1),
        )

    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(tick_id=0, policy_version=1, actions=torch.ones((1, 1)))
    ring.publish_observation(tick_id=1, observations=observations, dones=dones)
    ring.copy_observation_to(
        tick_id=1,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )


def test_inference_ring_epoch_is_fail_closed() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, epoch=7)

    with pytest.raises(RuntimeError, match="Inference epoch mismatch: expected 7, got 6"):
        ring.publish_observation(
            tick_id=0,
            observations=np.zeros((1, 1), dtype=np.float32),
            dones=np.zeros(1, dtype=np.float32),
            epoch=6,
        )
    ring.publish_observation(
        tick_id=0,
        observations=np.zeros((1, 1), dtype=np.float32),
        dones=np.zeros(1, dtype=np.float32),
        epoch=7,
    )
    with pytest.raises(RuntimeError, match="Inference epoch mismatch: expected 7, got 8"):
        ring.copy_observation_to(
            tick_id=0,
            observations=torch.empty((1, 1)),
            dones=torch.empty(1),
            epoch=8,
        )


def test_inference_ring_backpressure_releases_after_action_consumption() -> None:
    ring = SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=1)
    observations = np.zeros((1, 1), dtype=np.float32)
    dones = np.zeros(1, dtype=np.float32)
    ring.publish_observation(tick_id=0, observations=observations, dones=dones)

    assert (
        ring.try_publish_observation(
            tick_id=1,
            observations=observations,
            dones=dones,
            timeout_sec=0.0,
        )
        is False
    )

    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty((1, 1)),
        dones=torch.empty(1),
    )
    ring.publish_action(tick_id=0, policy_version=2, actions=torch.ones((1, 1)))
    ring.consume_action(tick_id=0)

    assert (
        ring.try_publish_observation(
            tick_id=1,
            observations=observations,
            dones=dones,
            timeout_sec=0.0,
        )
        is True
    )


def test_inference_ring_close_is_idempotent_and_keeps_diagnostics() -> None:
    ring = SharedInferenceRing(num_envs=2, obs_dim=3, action_dim=2, capacity=3)
    expected_bytes = ring.nbytes

    ring.close()
    ring.cleanup()

    assert ring.diagnostics["closed"] is True
    assert ring.nbytes == expected_bytes
    with pytest.raises(RuntimeError, match="inference ring is closed"):
        ring.publish_observation(
            tick_id=0,
            observations=np.zeros((2, 3), dtype=np.float32),
            dones=np.zeros(2, dtype=np.float32),
        )


def test_inference_ring_validates_constructor_and_payload_contract() -> None:
    with pytest.raises(ValueError, match="dimensions and capacity must be positive"):
        SharedInferenceRing(num_envs=0, obs_dim=1, action_dim=1)
    with pytest.raises(TypeError, match="capacity must be an integer"):
        SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, capacity=True)
    with pytest.raises(ValueError, match="epoch must be non-negative"):
        SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, epoch=-1)
    with pytest.raises(ValueError, match="CPU and CUDA only"):
        SharedInferenceRing(num_envs=1, obs_dim=1, action_dim=1, device="mps")

    ring = SharedInferenceRing(num_envs=1, obs_dim=2, action_dim=1)
    with pytest.raises(ValueError, match="observation shape must be"):
        ring.publish_observation(
            tick_id=0,
            observations=np.zeros((1, 3), dtype=np.float32),
            dones=np.zeros(1, dtype=np.float32),
        )
    with pytest.raises(TypeError, match="observations must be a NumPy array or Torch tensor"):
        ring.publish_observation(tick_id=0, observations=None, dones=np.zeros(1))
    with pytest.raises(TypeError, match="observation destinations must be Torch tensors"):
        ring.copy_observation_to(
            tick_id=0,
            observations=np.zeros((1, 2), dtype=np.float32),
            dones=torch.empty(1),
        )
    with pytest.raises(TypeError, match="destinations must have float32 dtype"):
        ring.copy_observation_to(
            tick_id=0,
            observations=torch.empty((1, 2), dtype=torch.float64),
            dones=torch.empty(1),
        )
    with pytest.raises(ValueError, match="policy version must be non-negative"):
        ring.publish_action(
            tick_id=0,
            policy_version=-1,
            actions=torch.empty((1, 1)),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inference_ring_waits_for_delayed_consumer_clone() -> None:
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        device="cuda",
        capacity=1,
    )
    observations = torch.zeros((2, 3), device="cuda")
    dones = torch.zeros(2, device="cuda")

    ring.publish_observation(tick_id=0, observations=observations, dones=dones, epoch=0)
    ring.copy_observation_to(
        tick_id=0,
        observations=torch.empty_like(observations),
        dones=torch.empty_like(dones),
        epoch=0,
    )
    ring.publish_action(
        tick_id=0,
        policy_version=1,
        actions=torch.zeros((2, 2), device="cuda"),
        epoch=0,
    )

    with torch.cuda.Stream() as delayed_stream:
        with torch.cuda.stream(delayed_stream):
            torch.cuda._sleep(50_000_000)
            received, _ = ring.consume_action(tick_id=0, epoch=0)

    ring.publish_observation(tick_id=1, observations=observations, dones=dones, epoch=0)
    ring.copy_observation_to(
        tick_id=1,
        observations=torch.empty_like(observations),
        dones=torch.empty_like(dones),
        epoch=0,
    )
    ring.publish_action(
        tick_id=1,
        policy_version=2,
        actions=torch.ones((2, 2), device="cuda"),
        epoch=0,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(received, torch.zeros((2, 2), device="cuda"))


def _cuda_ring_child(ring: SharedInferenceRing, result_queue: Any) -> None:
    try:
        for tick in range(3):
            ring.publish_observation(
                tick_id=tick,
                observations=(
                    torch.arange(6, dtype=torch.float32, device=ring.device).reshape(2, 3) + tick
                ),
                dones=torch.tensor([0.0, 1.0], device=ring.device),
                epoch=0,
            )
            actions, policy_version = ring.consume_action(tick_id=tick, epoch=0)
            expected_action = float(10 * tick)
            if not bool(torch.all(actions == expected_action).item()):
                raise AssertionError(f"action {tick} was overwritten: {actions.cpu()}")
        ring.close()
        result_queue.put(ring.diagnostics)
    except BaseException as error:  # pragma: no cover - propagated to parent
        result_queue.put(error)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inference_ring_ipc_events_cross_process() -> None:
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.Queue()
    ring = SharedInferenceRing(
        num_envs=2,
        obs_dim=3,
        action_dim=2,
        device="cuda",
        capacity=2,
        epoch=0,
    )
    process = spawn_context.Process(target=_cuda_ring_child, args=(ring, result_queue))

    process.start()
    try:
        observations = torch.empty((2, 3), device="cuda")
        dones = torch.empty(2, device="cuda")
        for tick in range(3):
            ring.copy_observation_to(
                tick_id=tick,
                observations=observations,
                dones=dones,
                epoch=0,
            )
            torch.testing.assert_close(
                observations,
                torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3) + tick,
            )
            ring.publish_action(
                tick_id=tick,
                policy_version=tick + 10,
                actions=torch.full((2, 2), float(10 * tick), dtype=torch.float32, device="cuda"),
                epoch=0,
            )
        result: Any | BaseException = result_queue.get(timeout=30)
    finally:
        process.join(timeout=30)
        if process.is_alive():  # pragma: no cover - failure path
            process.terminate()
            process.join()
        ring.close()

    assert not process.exitcode, f"child exited with {process.exitcode}"
    assert not isinstance(result, BaseException), result
    diagnostics = result
    assert diagnostics["published_tick"] == 2
    assert diagnostics["consumed_tick"] == 2
    assert diagnostics["closed"] is True
    assert not hasattr(ring, "observations")

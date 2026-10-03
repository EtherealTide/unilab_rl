"""Off-policy logger timing-window contracts."""

from __future__ import annotations

import pytest

from uni_rl.logging.offpolicy import OffPolicyLogger


def test_tail_env_steps_per_sec_uses_bounded_window() -> None:
    logger = OffPolicyLogger(algo_name="SAC", num_envs=4, max_iterations=30)
    for iteration in range(1, 31):
        logger.log_step(
            iteration=iteration,
            iteration_time=0.5,
            extra_info={"throughput_steps": 8},
        )

    assert logger._tail_iteration_times.maxlen == 20
    assert len(logger._tail_iteration_times) == 20
    assert logger._tail_throughput_env_steps == 20 * 8
    assert logger._get_tail_env_steps_per_sec() == pytest.approx(16.0)

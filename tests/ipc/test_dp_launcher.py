from __future__ import annotations

from typing import Any

import pytest

from uni_rl.ipc import dp_launcher
from uni_rl.ipc.dp_launcher import (
    DpRankSupervisor,
    rank_local_cuda_device,
    reject_removed_device_config,
    resolve_dp_rank_device,
    selected_visible_entries,
    visible_cuda_entries,
)


class _FakePopen:
    calls: list["_FakePopen"] = []

    def __init__(self, command: list[str], *, env: dict[str, str], start_new_session: bool) -> None:
        self.command = command
        self.env = env
        self.start_new_session = start_new_session
        self.pid = 123456
        self.returncode = 0
        type(self).calls.append(self)

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode


def test_dp_rank_supervisor_reuses_downstream_entry_script(monkeypatch: Any) -> None:
    """Spawned ranks must re-run the owner application's script."""
    _FakePopen.calls.clear()
    monkeypatch.setattr(dp_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_install_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_restore_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher, "_process_group_exists", lambda child: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0,GPU-1")
    monkeypatch.setattr(
        dp_launcher.sys,
        "argv",
        ["/workspace/UniLab/src/unilab/scripts/train_sac.py", "task=g1_walk_flat", "--debug"],
    )

    with DpRankSupervisor(world_size=2, log_dir="/tmp/run"):
        pass

    assert len(_FakePopen.calls) == 1
    assert _FakePopen.calls[0].command == [
        dp_launcher.sys.executable,
        "/workspace/UniLab/src/unilab/scripts/train_sac.py",
        "task=g1_walk_flat",
        "--debug",
    ]


def test_visible_cuda_entries_accepts_opaque_rank_local_tokens() -> None:
    assert visible_cuda_entries(None) == ()
    assert visible_cuda_entries("") == ()
    assert visible_cuda_entries("-1") == ()
    assert visible_cuda_entries("0") == ("0",)
    assert visible_cuda_entries("GPU-AAAA, MIG-BBBB") == ("GPU-AAAA", "MIG-BBBB")


def test_single_visible_gpu_is_rank_local_cuda_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid-7")

    assert rank_local_cuda_device() == "cuda:0"


def test_rank_device_uses_single_visibility_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv(dp_launcher.UNILAB_DP_WORLD_SIZE, raising=False)

    assert resolve_dp_rank_device() == "cuda:0"


def test_removed_training_devices_fails_closed() -> None:
    reject_removed_device_config(None)
    reject_removed_device_config([])

    with pytest.raises(ValueError, match="training.devices was removed"):
        reject_removed_device_config((0, 1))


def test_multi_visible_rank_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    with pytest.raises(ValueError, match="exactly one CUDA_VISIBLE_DEVICES"):
        resolve_dp_rank_device()


def test_selected_visibility_requires_exact_rank_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    assert selected_visible_entries(world_size=3) == ("GPU-a", "GPU-b", "GPU-c")

    with pytest.raises(ValueError, match="one entry per data-parallel rank"):
        selected_visible_entries(world_size=2)


def test_supervisor_children_each_receive_one_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakePopen.calls.clear()
    monkeypatch.setattr(dp_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_install_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_restore_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher, "_process_group_exists", lambda child: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    with DpRankSupervisor(world_size=3, log_dir="/tmp/run"):
        pass

    assert len(_FakePopen.calls) == 2
    assert [child.env["CUDA_VISIBLE_DEVICES"] for child in _FakePopen.calls] == ["GPU-b", "GPU-c"]


def test_unset_mask_with_one_visible_cuda_device_is_rank_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(dp_launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dp_launcher.torch.cuda, "device_count", lambda: 1)

    assert rank_local_cuda_device() == "cuda:0"


def test_unset_mask_with_multiple_visible_cuda_devices_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(dp_launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dp_launcher.torch.cuda, "device_count", lambda: 2)

    assert rank_local_cuda_device() is None

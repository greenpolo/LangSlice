"""The machine's cores divided among sections (langslice.core.hardware)."""

from __future__ import annotations

import contextvars
import threading
import time

import pytest

from langslice.core import hardware


def test_section_workers_divides_the_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hardware.CORES_VARIABLE, "24")
    assert hardware.section_workers(43, 8) == 3
    assert hardware.section_workers(43, 1) == 24
    assert hardware.section_workers(2, 1) == 2
    assert hardware.section_workers(1, 1) == 1
    monkeypatch.setenv(hardware.CORES_VARIABLE, "4")
    assert hardware.section_workers(43, 8) == 1


def test_map_sections_keeps_order_and_runs_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hardware.CORES_VARIABLE, "8")
    running, most = [0], [0]
    lock = threading.Lock()

    def step(item: int) -> int:
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return item * item

    assert hardware.map_sections(step, list(range(8)), 2) == [
        i * i for i in range(8)
    ]
    assert most[0] == 4


def test_map_sections_one_core_runs_in_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hardware.CORES_VARIABLE, "1")
    seen: list[str] = []
    hardware.map_sections(
        lambda item: seen.append(threading.current_thread().name), [1, 2], 1
    )
    assert seen == [threading.current_thread().name] * 2


def test_map_sections_carries_context_and_raises_after_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hardware.CORES_VARIABLE, "4")
    collected: contextvars.ContextVar[list[int] | None] = (
        contextvars.ContextVar("collected", default=None)
    )
    notes: list[int] = []
    collected.set(notes)
    finished: list[int] = []

    def step(item: int) -> int:
        held = collected.get()
        assert held is not None
        held.append(item)
        if item == 1:
            raise ValueError("section 1")
        time.sleep(0.02)
        finished.append(item)
        return item

    with pytest.raises(ValueError, match="section 1"):
        hardware.map_sections(step, [0, 1, 2, 3], 1)
    assert sorted(notes) == [0, 1, 2, 3]
    assert sorted(finished) == [0, 2, 3]


def test_detect_names_the_machine() -> None:
    found = hardware.detect(gpu=False)
    assert found.cores >= 1
    assert found.memory_gib >= 0
    assert "cores" in found.describe()

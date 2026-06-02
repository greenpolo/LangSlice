"""Tests for langslice_training.utils.fast_io.

Detects when a training run's hot I/O paths land on the slow Windows/9p bind
instead of the fast WSL2 ext4 volume, so the trainer can warn (or, in strict
mode, refuse). Mount-table data is injected so the logic is testable off-box.
"""
from __future__ import annotations

import logging

import pytest
from langslice_training.utils import fast_io

# Mirrors the real dev-container layout, verified via /proc/mounts on
# 2026-05-27: the repo bind and out/ are 9p (Windows host), while the named
# volumes (out/cache_fast, data/datasets, /models, brainglobe) are ext4.
MOUNTS = [
    ("/", "overlay"),
    ("/workspace/LangSlice", "9p"),
    ("/workspace/LangSlice/out", "9p"),
    ("/workspace/LangSlice/out/cache_fast", "ext4"),
    ("/workspace/LangSlice/data/datasets", "ext4"),
    ("/models", "ext4"),
    ("/home/unsloth/.brainglobe", "ext4"),
]


def test_filesystem_for_picks_longest_prefix():
    # cache_fast is nested under out/ (9p) but is its own ext4 mount — the
    # longest-prefix match must win, not the parent 9p mount.
    assert (
        fast_io.filesystem_for("/workspace/LangSlice/out/cache_fast/atlas", MOUNTS)
        == "ext4"
    )
    assert fast_io.filesystem_for("/workspace/LangSlice/out/runs/v8", MOUNTS) == "9p"
    assert fast_io.filesystem_for("/workspace/LangSlice/src/x.py", MOUNTS) == "9p"
    assert fast_io.filesystem_for("/models/sft-base", MOUNTS) == "ext4"


def test_filesystem_for_unknown_when_no_mounts():
    assert fast_io.filesystem_for("/anything", []) is None


def test_is_slow_path():
    assert fast_io.is_slow_path("/workspace/LangSlice/out/runs/v8", MOUNTS) is True
    assert (
        fast_io.is_slow_path("/workspace/LangSlice/out/cache_fast/atlas", MOUNTS)
        is False
    )
    assert fast_io.is_slow_path("/models/sft-base", MOUNTS) is False


def test_is_slow_path_unknown_is_not_slow():
    # Can't determine the filesystem (no /proc/mounts, e.g. non-Linux host) ->
    # don't cry wolf.
    assert fast_io.is_slow_path("/whatever", []) is False


def test_warn_if_slow_io_flags_slow_skips_none_and_fast(caplog):
    paths = {
        "output-dir": "/workspace/LangSlice/out/runs/v8",  # 9p -> slow
        "atlas-embedding-cache": "/workspace/LangSlice/out/cache_fast/atlas",  # ext4
        "query-embedding-cache": None,  # unset -> skipped
    }
    with caplog.at_level(logging.WARNING):
        flagged = fast_io.warn_if_slow_io(paths, mounts=MOUNTS)
    assert {f[0] for f in flagged} == {"output-dir"}
    assert any(
        "output-dir" in r.message and "9p" in r.message for r in caplog.records
    )


def test_warn_if_slow_io_no_warning_when_all_fast(caplog):
    paths = {
        "output-dir": "/workspace/LangSlice/out/cache_fast/runs/v8",
        "models": "/models/sft-base",
    }
    with caplog.at_level(logging.WARNING):
        flagged = fast_io.warn_if_slow_io(paths, mounts=MOUNTS)
    assert flagged == []
    assert not caplog.records


def test_warn_if_slow_io_strict_raises():
    paths = {"output-dir": "/workspace/LangSlice/out/runs/v8"}
    with pytest.raises(RuntimeError, match="Slow filesystem"):
        fast_io.warn_if_slow_io(paths, mounts=MOUNTS, strict=True)


def test_strict_env_enables_hard_error(monkeypatch):
    monkeypatch.setenv(fast_io.STRICT_ENV, "1")
    with pytest.raises(RuntimeError):
        fast_io.warn_if_slow_io(
            {"output-dir": "/workspace/LangSlice/out/runs/x"}, mounts=MOUNTS
        )


def test_read_mounts_parses_real_9p_line(tmp_path):
    # The real /proc/mounts 9p line has a messy options field full of ; and =.
    content = (
        "overlay / overlay rw,relatime 0 0\n"
        "C:\\134 /workspace/LangSlice 9p rw,noatime,aname=drvfs;path=C:\\ 0 0\n"
        "/dev/sde /workspace/LangSlice/out/cache_fast ext4 rw,relatime 0 0\n"
    )
    f = tmp_path / "mounts"
    f.write_text(content)
    mounts = fast_io.read_mounts(f)
    assert ("/workspace/LangSlice", "9p") in mounts
    assert ("/workspace/LangSlice/out/cache_fast", "ext4") in mounts


def test_read_mounts_missing_file_returns_empty(tmp_path):
    assert fast_io.read_mounts(tmp_path / "does-not-exist") == []


def test_filesystem_for_resolves_relative_against_cwd():
    # The trainer's CWD in the container is on the 9p bind, so a RELATIVE
    # output-dir must resolve there and be caught, not silently ignored.
    cwd = "/workspace/LangSlice/models/langslice-gemma-4/training"
    assert fast_io.filesystem_for("out/rl/run0", MOUNTS, cwd=cwd) == "9p"
    assert (
        fast_io.filesystem_for(
            "out/cache_fast/atlas", MOUNTS, cwd="/workspace/LangSlice"
        )
        == "ext4"
    )


def test_warn_flags_relative_path_on_slow_cwd(caplog):
    with caplog.at_level(logging.WARNING):
        flagged = fast_io.warn_if_slow_io(
            {"--output-dir": "out/rl/run0"},
            mounts=MOUNTS,
            cwd="/workspace/LangSlice/models/langslice-gemma-4/training",
        )
    assert {f[0] for f in flagged} == {"--output-dir"}

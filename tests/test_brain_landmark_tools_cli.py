"""``estimate-brain --landmark-tools/--no-landmark-tools``: parsing + wiring."""

from __future__ import annotations

import argparse
from pathlib import Path

from langslice import cli


def _parse(args: list[str]) -> argparse.Namespace:
    return cli._build_parser().parse_args(args)


def test_landmark_tools_defaults_on(tmp_path: Path):
    args = _parse(["linear", "estimate-brain", str(tmp_path)])
    assert args.landmark_tools is True


def test_landmark_tools_flag_parses_both_ways(tmp_path: Path):
    on = _parse(["linear", "estimate-brain", str(tmp_path), "--landmark-tools"])
    off = _parse(["linear", "estimate-brain", str(tmp_path), "--no-landmark-tools"])
    assert on.landmark_tools is True
    assert off.landmark_tools is False


def test_landmark_tools_flag_reaches_brain_config(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_run_brain(config, **kwargs):
        del kwargs
        captured["landmark_tools"] = config.landmark_tools
        from langslice.linear.whole_brain.state import StackState

        return StackState()

    monkeypatch.setattr("langslice.linear.whole_brain.run_brain", fake_run_brain)

    args = _parse(["linear", "estimate-brain", str(tmp_path), "--no-landmark-tools"])
    cli._run_estimate_brain(args)

    assert captured["landmark_tools"] is False

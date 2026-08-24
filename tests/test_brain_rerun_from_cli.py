"""``estimate-brain --rerun-from``: parsing, checkpoint requirement, wiring."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from langslice import cli
from langslice.linear.whole_brain.checkpoint import (
    default_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from langslice.linear.whole_brain.state import SliceState, StackState


def _parse(args: list[str]) -> argparse.Namespace:
    return cli._build_parser().parse_args(args)


def test_rerun_from_parses_node_choice(tmp_path: Path):
    args = _parse(
        ["linear", "estimate-brain", str(tmp_path), "--rerun-from", "transforms"]
    )
    assert args.rerun_from == "transforms"
    assert args.resume is True  # default, --fresh not given


def test_rerun_from_rejects_unknown_node(tmp_path: Path, capsys):
    with pytest.raises(SystemExit):
        _parse(["linear", "estimate-brain", str(tmp_path), "--rerun-from", "survey"])
    assert "invalid choice" in capsys.readouterr().err


def test_rerun_from_and_fresh_are_mutually_exclusive(tmp_path: Path, capsys):
    with pytest.raises(SystemExit):
        _parse(
            ["linear", "estimate-brain", str(tmp_path), "--fresh", "--rerun-from", "position"]
        )
    assert "not allowed with argument" in capsys.readouterr().err


def test_rerun_from_without_checkpoint_errors_clearly(tmp_path: Path):
    args = _parse(["linear", "estimate-brain", str(tmp_path), "--rerun-from", "position"])
    with pytest.raises(SystemExit, match="checkpoint"):
        cli._run_estimate_brain(args)


def test_rerun_from_rewinds_checkpoint_and_resumes(tmp_path: Path, monkeypatch):
    checkpoint_path = default_checkpoint_path(str(tmp_path))
    state = StackState(
        image_folder=str(tmp_path),
        slices=[
            SliceState(
                id="s1.png",
                index_original=0,
                index_corrected=0,
                position_mm=1.0,
                position_source="refined",
                affine=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
        ],
        completed_nodes=[
            "ingest", "survey", "fix", "seed",
            "position", "transforms", "review", "emit",
        ],
        node_cycles={
            name: 1
            for name in (
                "ingest", "survey", "fix", "seed",
                "position", "transforms", "review", "emit",
            )
        },
    )
    save_checkpoint(state, checkpoint_path)

    captured: dict[str, object] = {}

    async def fake_run_brain(config, **kwargs):
        captured["resume"] = config.resume
        captured["stop_after"] = kwargs.get("stop_after")
        return StackState()

    monkeypatch.setattr("langslice.linear.whole_brain.run_brain", fake_run_brain)

    args = _parse(["linear", "estimate-brain", str(tmp_path), "--rerun-from", "position"])
    cli._run_estimate_brain(args)

    assert captured["resume"] is True

    rewound = load_checkpoint(checkpoint_path)
    assert rewound is not None
    assert rewound.slices[0].position_mm is None
    assert rewound.slices[0].affine is None
    assert rewound.completed_nodes == ["ingest", "survey", "fix", "seed"]

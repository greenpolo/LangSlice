"""Tests for the ``--plane`` flag on ``estimate`` and ``register`` CLI parsers."""

from __future__ import annotations

import pytest
from PIL import Image

from langslice_harness.cli import _build_parser


def _parse(args: list[str]):
    return _build_parser().parse_args(args)


def test_estimate_default_plane_is_coronal():
    args = _parse(["estimate", "tests/fixture.png"])
    assert args.command == "estimate"
    assert args.plane == "coronal"


def test_estimate_accepts_sagittal_plane():
    args = _parse(["estimate", "tests/fixture.png", "--plane", "sagittal"])
    assert args.command == "estimate"
    assert args.plane == "sagittal"


def test_estimate_accepts_horizontal_plane():
    args = _parse(["estimate", "tests/fixture.png", "--plane", "horizontal"])
    assert args.command == "estimate"
    assert args.plane == "horizontal"


def test_estimate_rejects_unknown_plane(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["estimate", "tests/fixture.png", "--plane", "axial"])
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "axial" in err


def test_register_default_plane_is_coronal():
    args = _parse(["register", "tests/fixture.png", "--position", "5.0"])
    assert args.command == "register"
    assert args.plane == "coronal"


def test_register_accepts_sagittal_plane():
    args = _parse(
        ["register", "tests/fixture.png", "--position", "5.0", "--plane", "sagittal"]
    )
    assert args.command == "register"
    assert args.plane == "sagittal"


def test_register_accepts_horizontal_plane():
    args = _parse(
        ["register", "tests/fixture.png", "--position", "5.0", "--plane", "horizontal"]
    )
    assert args.command == "register"
    assert args.plane == "horizontal"


def test_register_rejects_unknown_plane(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["register", "tests/fixture.png", "--position", "5.0", "--plane", "axial"]
        )
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_estimate_runs_tool_use_for_any_plane(monkeypatch):
    """``estimate`` always routes through the single tool-use estimator."""
    from langslice_harness.harness.estimation._types import PositionResult

    captured: dict[str, object] = {}

    def fake_open(_path):
        return Image.new("RGB", (32, 24), color=(120, 120, 120))

    monkeypatch.setattr("PIL.Image.open", fake_open)

    def fake_adaptive_preprocess(img):
        return img

    monkeypatch.setattr(
        "langslice_harness.image_prep.adaptive_preprocess",
        fake_adaptive_preprocess,
    )

    def fake_estimate_position(**kwargs):
        captured["tool_use_called"] = True
        captured["tool_use_kwargs"] = kwargs
        return PositionResult(position_mm=4.56, reasoning="stubbed tool-use result")

    monkeypatch.setattr(
        "langslice_harness.estimation.estimate_position", fake_estimate_position
    )

    from langslice_harness import cli

    monkeypatch.setattr(
        "sys.argv",
        ["langslice", "estimate", "tests/fixture.png", "--plane", "sagittal"],
    )

    cli.main()

    assert captured.get("tool_use_called") is True

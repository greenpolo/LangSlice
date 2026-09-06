"""Tests for the ``--plane`` flag on the ``linear run``/``register`` CLI parsers."""

from __future__ import annotations

import pytest

from langslice.cli import _build_parser


def _parse(args: list[str]):
    return _build_parser().parse_args(args)


def test_linear_run_default_plane_is_coronal():
    args = _parse(["linear", "run", "sections/"])
    assert args.subcommand == "run"
    assert args.plane == "coronal"


def test_linear_run_accepts_sagittal_plane():
    args = _parse(["linear", "run", "sections/", "--plane", "sagittal"])
    assert args.plane == "sagittal"


def test_linear_run_accepts_horizontal_plane():
    args = _parse(["linear", "run", "sections/", "--plane", "horizontal"])
    assert args.plane == "horizontal"


def test_linear_run_rejects_unknown_plane(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["linear", "run", "sections/", "--plane", "axial"])
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "axial" in err


def test_register_default_plane_is_coronal():
    args = _parse(["nonlinear", "register", "tests/fixture.png", "--position", "5.0"])
    assert args.subcommand == "register"
    assert args.plane == "coronal"


def test_register_accepts_sagittal_plane():
    args = _parse(
        ["nonlinear", "register", "tests/fixture.png", "--position", "5.0", "--plane", "sagittal"]
    )
    assert args.subcommand == "register"
    assert args.plane == "sagittal"


def test_register_accepts_horizontal_plane():
    args = _parse(
        ["nonlinear", "register", "tests/fixture.png", "--position", "5.0", "--plane", "horizontal"]
    )
    assert args.subcommand == "register"
    assert args.plane == "horizontal"


def test_register_rejects_unknown_plane(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["nonlinear", "register", "tests/fixture.png", "--position", "5.0", "--plane", "axial"]
        )
    err = capsys.readouterr().err
    assert "invalid choice" in err

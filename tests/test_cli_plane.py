"""Tests for the ``--plane`` flag on the ``linear run`` CLI parser."""

from __future__ import annotations

import pytest

from langslice.doors.cli import build_parser


def _parse(args: list[str]):
    return build_parser().parse_args(args)


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
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["linear", "run", "sections/", "--plane", "axial"])
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "axial" in err


def test_the_nonlinear_register_command_is_gone(capsys):
    """``nonlinear register`` (a one-shot pipeline outside the job) was
    removed 2026-10-04; its group went with it."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nonlinear", "register", "x.png", "--position", "5.0"])
    assert "invalid choice" in capsys.readouterr().err

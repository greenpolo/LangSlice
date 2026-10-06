"""The stack-opening commands' job flags: ``--plane`` and bad JSON values."""

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


@pytest.mark.parametrize("command", [["linear", "run"], ["mcp"]])
def test_a_bad_job_flag_ends_with_a_message_not_a_traceback(tmp_path, command):
    from langslice.doors.cli import main

    with pytest.raises(SystemExit) as stopped:
        main([*command, str(tmp_path), "--positions", "{not json"])
    assert "--positions" in str(stopped.value.code)

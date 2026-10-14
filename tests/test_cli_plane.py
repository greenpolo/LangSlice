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


@pytest.mark.parametrize("value, expected", [
    ("2.5", 2.5),
    ('{"a.png": 3.0, "b.png": 4.5}', {"a.png": 3.0, "b.png": 4.5}),
])
def test_pixel_size_is_one_number_or_one_per_file(tmp_path, value, expected):
    from langslice.doors.cli.linear import spec_from_args

    args = _parse(["linear", "run", str(tmp_path), "--pixel-size-um", value])
    assert spec_from_args(args, str(tmp_path)).inputs["pixel_size_um"] == expected

"""Argument parsing of the stack-opening commands: ``--plane``, ``--force-view``,
``--require-deformation``, bad JSON flag values and ``--pixel-size-um``."""

from __future__ import annotations

import pytest

from langslice.doors.cli import build_parser
from langslice.doors.cli.linear import spec_from_args


def _parse(args: list[str]):
    return build_parser().parse_args(args)


@pytest.mark.parametrize("plane", ["coronal", "sagittal", "horizontal"])
def test_linear_run_takes_the_three_planes(plane):
    assert _parse(["linear", "run", "sections/", "--plane", plane]).plane == plane


def test_linear_run_rejects_unknown_plane(capsys):
    with pytest.raises(SystemExit):
        _parse(["linear", "run", "sections/", "--plane", "axial"])
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "axial" in err


def test_force_view_and_require_deformation_flags_reach_the_spec(tmp_path):
    args = _parse(["linear", "run", str(tmp_path), "--force-view", "--require-deformation",
                   "--disable", "elastix_affine"])
    spec = spec_from_args(args, str(tmp_path))
    assert spec.force_view is True
    assert spec.nonlinear.require_deformation is True


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
    args = _parse(["linear", "run", str(tmp_path), "--pixel-size-um", value])
    assert spec_from_args(args, str(tmp_path)).inputs["pixel_size_um"] == expected

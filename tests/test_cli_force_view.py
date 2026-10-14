"""``langslice linear run --force-view`` turns on JobSpec.force_view (change tools always show)."""

import argparse

from langslice.doors.cli import linear


def _spec(tmp_path, *extra):
    parser = argparse.ArgumentParser()
    linear.add_run_parser(parser.add_subparsers(dest="cmd"))
    args = parser.parse_args(["run", str(tmp_path), "--atlas", "allen_mouse_25um",
                              "--plane", "coronal", *extra])
    return linear.spec_from_args(args, str(tmp_path))


def test_force_view_is_off_by_default_and_on_with_the_flag(tmp_path):
    assert _spec(tmp_path).force_view is False
    assert _spec(tmp_path, "--force-view", "--disable", "elastix_affine").force_view is True

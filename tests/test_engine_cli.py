from __future__ import annotations

import pytest

from langslice.doors.cli import build_parser, main
from langslice.doors.cli.linear import build_linear_spec
from langslice.hosts.cli import run_abba


def test_parser_supports_serve_command() -> None:
    args = build_parser().parse_args(["serve", "--stdio"])
    assert args.command == "serve"
    assert args.stdio is True


def test_nonlinear_task_notes_tool_configuration_and_host_transforms() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "linear", "run", "/stack", "--tasks", "nonlinear",
        "--image-provider", "openai-oauth", "--image-model", "test-image",
        "--transforms", '{"s.png": {"params": [1, 0, 0, 0, 1, 0]}}',
    ])
    spec = build_linear_spec(args, args.image_folder)
    assert spec.tasks == ["nonlinear"]
    assert spec.nonlinear.image_model == "test-image"
    assert spec.inputs["transforms"]["s.png"]["params"] == [1, 0, 0, 0, 1, 0]
    defaults = build_linear_spec(parser.parse_args(["linear", "run", "/stack"]), "/stack")
    assert defaults.tasks == ["reorder", "position", "transform"]
    # No image model: the nonlinear task fits deformations to the stain alone.
    none = build_linear_spec(parser.parse_args([
        "linear", "run", "/stack", "--tasks", "transform,nonlinear", "--image-provider", "none",
    ]), "/stack")
    assert none.nonlinear.provider == "none" and none.nonlinear.uses_image_model is False
    with pytest.raises(ValueError, match="nonlinear.provider"):
        build_linear_spec(parser.parse_args([
            "linear", "run", "/stack", "--image-provider", "mystery"]), "/stack")


def test_serve_requires_stdio() -> None:
    with pytest.raises(SystemExit):
        main(["serve"])


def test_abba_takes_no_linear_job_options_any_more() -> None:
    """`langslice abba` only starts ABBA with the connector: the fresh-import
    `--linear` run and `--save-state` were removed (2026-10-04)."""
    parser = build_parser()
    args = parser.parse_args(["abba", "--no-log"])
    assert args.no_log and not args.no_viewer and args.connector_jar is None
    for removed in (["--linear", "/stack"], ["--save-state", "/out.abba"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["abba", *removed])


def test_run_abba_starts_abba_with_the_connector_jar(monkeypatch, tmp_path) -> None:
    from typing import Any

    from langslice.hosts.integrations import abba_launch

    jar = tmp_path / "langslice-fiji-0.1.jar"
    jar.write_bytes(b"PK")
    calls: dict[str, Any] = {}
    monkeypatch.setattr(abba_launch, "run_abba_session", lambda **kw: calls.update(kw))
    run_abba(build_parser().parse_args(["abba", "--connector-jar", str(jar), "--no-viewer"]))
    assert calls == {"abba_atlas": "Adult Mouse Brain - Allen Brain Atlas V3p1",
                     "jar": str(jar.resolve()), "viewer": False, "log": True}


def test_run_abba_without_a_connector_jar_says_how_to_get_one(monkeypatch, tmp_path) -> None:
    from langslice.hosts.integrations import abba_launch

    monkeypatch.delenv(abba_launch.CONNECTOR_JAR_ENV, raising=False)
    monkeypatch.setattr(abba_launch, "repository_root", lambda: tmp_path)
    with pytest.raises(SystemExit, match="mvn package"):
        run_abba(build_parser().parse_args(["abba"]))

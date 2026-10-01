from __future__ import annotations

import pytest

import langslice.cli as cli


def test_parser_supports_serve_command() -> None:
    args = cli._build_parser().parse_args(["serve", "--stdio"])
    assert args.command == "serve"
    assert args.stdio is True


def test_nonlinear_task_notes_tool_configuration_and_host_transforms() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([
        "linear", "run", "/stack", "--tasks", "nonlinear",
        "--image-provider", "openai-oauth", "--image-model", "test-image",
        "--transforms", '{"s.png": {"params": [1, 0, 0, 0, 1, 0]}}',
    ])
    spec = cli._build_linear_spec(args, args.image_folder)
    assert spec.tasks == ["nonlinear"]
    assert spec.nonlinear.image_model == "test-image"
    assert spec.inputs["transforms"]["s.png"]["params"] == [1, 0, 0, 0, 1, 0]
    defaults = cli._build_linear_spec(parser.parse_args(["linear", "run", "/stack"]), "/stack")
    assert defaults.tasks == ["reorder", "position", "transform"]
    # No image model: the nonlinear task fits deformations to the stain alone.
    none = cli._build_linear_spec(parser.parse_args([
        "linear", "run", "/stack", "--tasks", "transform,nonlinear", "--image-provider", "none",
    ]), "/stack")
    assert none.nonlinear.provider == "none" and none.nonlinear.uses_image_model is False
    with pytest.raises(ValueError, match="nonlinear.provider"):
        cli._build_linear_spec(parser.parse_args([
            "linear", "run", "/stack", "--image-provider", "mystery"]), "/stack")


def test_serve_requires_stdio() -> None:
    with pytest.raises(SystemExit):
        cli.main(["serve"])


def test_abba_and_linear_run_parse_the_same_flags_to_the_same_jobspec() -> None:
    parser = cli._build_parser()
    linear_args = parser.parse_args(
        ["linear", "run", "/stack", "--thickness", "40", "--interval", "150", "--angles"]
    )
    abba_args = parser.parse_args(
        ["abba", "--linear", "/stack", "--thickness", "40", "--interval", "150", "--angles"]
    )

    linear_spec = cli._build_linear_spec(linear_args, linear_args.image_folder)
    abba_spec = cli._build_linear_spec(abba_args, abba_args.linear)

    assert linear_spec == abba_spec
    assert linear_spec.image_folder == "/stack"
    assert linear_spec.position.thickness_um == 40
    assert linear_spec.position.interval_um == 150
    assert linear_spec.transform.angles is True


def test_abba_parser_accepts_save_state_and_defaults_linear_off() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["abba", "--save-state", "/out.abba"])
    assert args.linear is None
    assert args.save_state == "/out.abba"


def test_run_abba_dispatches_to_run_linear_in_abba_when_linear_is_given(monkeypatch) -> None:
    import sys
    import types
    from typing import Any

    from langslice.linear.spec import JobSpec

    calls: dict[str, Any] = {}

    fake_abba_python = types.ModuleType("abba_python")
    monkeypatch.setitem(sys.modules, "abba_python", fake_abba_python)

    fake_module = types.ModuleType("langslice.integrations.abba_linear")

    def fake_run_linear_in_abba(spec: JobSpec, **kwargs: Any) -> None:
        calls["spec"] = spec
        calls["kwargs"] = kwargs

    fake_module.run_linear_in_abba = fake_run_linear_in_abba  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "langslice.integrations.abba_linear", fake_module)

    parser = cli._build_parser()
    args = parser.parse_args(["abba", "--linear", "/stack", "--save-state", "/out.abba"])
    cli._run_abba(args)

    spec = calls["spec"]
    assert isinstance(spec, JobSpec)
    assert spec.image_folder == "/stack"
    assert calls["kwargs"]["save_state"] == "/out.abba"

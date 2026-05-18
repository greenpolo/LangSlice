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


def _install_estimate_runtime_stubs(monkeypatch):
    """Stub the heavy CLI dependencies so ``_run_estimate`` can run end-to-end.

    Returns a dict the test can inspect (e.g. whether the image-gen pathway
    was actually invoked when no SystemExit is expected).
    """
    from langslice_harness.harness.estimation._types import PositionResult

    captured: dict[str, object] = {}

    # Stub the model-loading machinery so the CLI sees an image-gen model.
    import langslice_harness.vlm_config as vlm_config

    monkeypatch.setattr(vlm_config, "MODEL_NAME", "fake-image-gen-model")
    monkeypatch.setattr(vlm_config, "THINKING_LEVEL", "MEDIUM")
    monkeypatch.setattr(vlm_config, "TEMPERATURE", 0.0)
    monkeypatch.setattr(
        vlm_config, "is_image_generation_model", lambda model_name: True
    )

    # _run_estimate uses ``from PIL import Image`` locally; patching the
    # canonical PIL.Image.open is enough — the local ``Image`` binding still
    # resolves attribute lookups against the module.
    def fake_open(_path):
        return Image.new("RGB", (32, 24), color=(120, 120, 120))

    monkeypatch.setattr("PIL.Image.open", fake_open)

    # Skip preprocessing side-effects (CLAHE etc.).
    def fake_adaptive_preprocess(img):
        return img

    monkeypatch.setattr(
        "langslice_harness.image_prep.adaptive_preprocess",
        fake_adaptive_preprocess,
    )

    # Capture but don't actually run the image-gen estimator. The estimator
    # only gets called on the coronal pass-through path.
    def fake_estimate_position_image_gen(**kwargs):
        captured["image_gen_called"] = True
        captured["image_gen_kwargs"] = kwargs
        return PositionResult(
            position_mm=1.23, reasoning="stubbed image-gen result"
        )

    monkeypatch.setattr(
        "langslice_harness.harness.estimation.image_gen.estimate_position_image_gen",
        fake_estimate_position_image_gen,
    )

    # Same guard for the tool-use path so a misrouted test still won't make
    # a real network call.
    def fake_estimate_position(**kwargs):
        captured["tool_use_called"] = True
        captured["tool_use_kwargs"] = kwargs
        return PositionResult(
            position_mm=4.56, reasoning="stubbed tool-use result"
        )

    monkeypatch.setattr(
        "langslice_harness.estimation.estimate_position", fake_estimate_position
    )

    return captured


def test_estimate_image_gen_rejects_non_coronal_plane(monkeypatch, capsys):
    """``--workflow image_gen --plane sagittal`` must raise SystemExit."""
    _install_estimate_runtime_stubs(monkeypatch)
    from langslice_harness import cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "langslice",
            "estimate",
            "tests/fixture.png",
            "--workflow",
            "image_gen",
            "--plane",
            "sagittal",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    # SystemExit.code carries the explanatory string we raised.
    message = str(excinfo.value)
    assert "coronal" in message
    assert "sagittal" in message
    assert "tool_use" in message


def test_estimate_image_gen_rejects_horizontal_plane(monkeypatch):
    """The guard must reject ``horizontal`` too, not just ``sagittal``."""
    _install_estimate_runtime_stubs(monkeypatch)
    from langslice_harness import cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "langslice",
            "estimate",
            "tests/fixture.png",
            "--workflow",
            "image_gen",
            "--plane",
            "horizontal",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    message = str(excinfo.value)
    assert "coronal" in message
    assert "horizontal" in message


def test_estimate_image_gen_accepts_coronal_plane(monkeypatch):
    """``--workflow image_gen --plane coronal`` must reach the estimator."""
    captured = _install_estimate_runtime_stubs(monkeypatch)
    from langslice_harness import cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "langslice",
            "estimate",
            "tests/fixture.png",
            "--workflow",
            "image_gen",
            "--plane",
            "coronal",
        ],
    )

    # No SystemExit means the coronal path proceeded to the estimator stub.
    cli.main()

    assert captured.get("image_gen_called") is True
    assert captured.get("tool_use_called") is None

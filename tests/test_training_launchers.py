from __future__ import annotations

import pathlib

import tomllib

import langslice_harness.training_launchers as training_launchers


def test_pyproject_training_scripts_are_model_scoped() -> None:
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts["langslice-gemma-sft"] == "langslice_harness.training_launchers:gemma_sft"
    assert scripts["langslice-gemma-rl"] == "langslice_harness.training_launchers:gemma_rl"
    assert "langslice-sft-train" not in scripts
    assert "langslice-single-turn-rl" not in scripts
    assert "langslice-isft" not in scripts


def test_public_training_launchers_expose_model_scoped_wrappers() -> None:
    assert hasattr(training_launchers, "gemma_sft")
    assert hasattr(training_launchers, "gemma_rl")
    assert not hasattr(training_launchers, "sft_train")
    assert not hasattr(training_launchers, "single_turn_rl")
    assert not hasattr(training_launchers, "isft")

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_add_transition_roots_adds_legacy_training_path() -> None:
    compat = importlib.import_module("_compat")
    compat.add_transition_roots()
    expected = Path("models/langslice-gemma-4/training")
    assert any(Path(p).as_posix().endswith(expected.as_posix()) for p in sys.path)


def test_augmentation_shim_module_imports() -> None:
    mod = importlib.import_module("augmentation")
    assert hasattr(mod, "__path__")
    assert Path(next(iter(mod.__path__))).name == "augmentation"

"""Installed console-script wrappers for model-scoped training commands."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if not (candidate / "models" / "langslice-gemma-4" / "training" / "launchers.py").exists():
        msg = "LangSlice training launchers require a source checkout with models/"
        raise RuntimeError(msg)
    return candidate


def _load_model_launchers() -> ModuleType:
    repo_root = _repo_root()
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    path = repo_root / "models" / "langslice-gemma-4" / "training" / "launchers.py"
    spec = importlib.util.spec_from_file_location("_langslice_gemma_training_launchers", path)
    if spec is None or spec.loader is None:
        msg = f"Could not load training launchers from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gemma_rl() -> None:
    _load_model_launchers().gemma_rl(sys.argv[1:])


def gemma_sft() -> None:
    _load_model_launchers().gemma_sft(sys.argv[1:])

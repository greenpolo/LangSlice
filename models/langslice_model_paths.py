"""Shared path helpers for the LangSlice models workspace."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root_from(path: Path) -> Path:
    """Return the repository root for a path inside this checkout."""
    current = path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / ".git").exists():
            return candidate
    msg = f"Could not find LangSlice repo root from {path}"
    raise RuntimeError(msg)


def model_python_roots(repo_root: Path, *, existing_only: bool = True) -> list[Path]:
    """Return import roots for shared model packages and compatibility shims."""
    roots = [
        repo_root / "models" / "langslice-traces",
        repo_root / "models" / "training-core",
        repo_root / "models" / "data",
        repo_root / "models" / "langslice-gemma-4" / "data",
        repo_root / "models" / "langslice-gemma-4" / "training",
    ]
    if not existing_only:
        return roots
    return [root for root in roots if root.exists()]


def add_model_python_paths(repo_root: Path | None = None) -> list[Path]:
    """Add model workspace import roots to ``sys.path`` in priority order."""
    root = repo_root or repo_root_from(Path(__file__))
    added: list[Path] = []
    insert_at = 0
    for path in model_python_roots(root):
        path_str = str(path)
        if path_str in sys.path:
            continue
        sys.path.insert(insert_at, path_str)
        insert_at += 1
        added.append(path)
    return added

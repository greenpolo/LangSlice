from __future__ import annotations

from pathlib import Path


def default_manifest_root(repo_root: Path) -> Path:
    return repo_root / "models" / "data" / "manifest"


def fallback_manifest_root(repo_root: Path) -> Path:
    return repo_root / "data" / "manifest"


def resolve_manifest_root(repo_root: Path) -> Path:
    preferred = default_manifest_root(repo_root)
    if preferred.exists():
        return preferred
    fallback = fallback_manifest_root(repo_root)
    if fallback.exists():
        return fallback
    return preferred

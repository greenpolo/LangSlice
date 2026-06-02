from __future__ import annotations

import sys
from pathlib import Path


def test_model_python_roots_prefers_shared_hub_paths() -> None:
    from models.langslice_model_paths import model_python_roots

    repo_root = Path("C:/repo")

    assert model_python_roots(repo_root, existing_only=False) == [
        repo_root / "models" / "langslice-traces",
        repo_root / "models" / "training-core",
        repo_root / "models" / "data",
        repo_root / "models" / "langslice-gemma-4" / "data",
        repo_root / "models" / "langslice-gemma-4" / "training",
    ]


def test_add_model_python_paths_preserves_existing_order(tmp_path: Path) -> None:
    from models.langslice_model_paths import add_model_python_paths

    repo_root = tmp_path
    shared = repo_root / "models" / "langslice-traces"
    gemma_training = repo_root / "models" / "langslice-gemma-4" / "training"
    shared.mkdir(parents=True)
    gemma_training.mkdir(parents=True)

    original = sys.path[:]
    sys.path[:] = ["already-there"]
    try:
        added = add_model_python_paths(repo_root)
        assert added == [shared, gemma_training]
        assert sys.path[:3] == [str(shared), str(gemma_training), "already-there"]

        added_again = add_model_python_paths(repo_root)
        assert added_again == []
        assert sys.path[:3] == [str(shared), str(gemma_training), "already-there"]
    finally:
        sys.path[:] = original

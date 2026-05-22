"""Console launchers for LangSlice Gemma model-scoped training commands."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bootstrap() -> None:
    repo_root = _repo_root()
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from models.langslice_model_paths import add_model_python_paths  # noqa: PLC0415

    add_model_python_paths(repo_root)


def gemma_rl(argv: list[str] | None = None) -> None:
    """Run the Gemma single-turn RL training CLI."""
    _bootstrap()
    from langslice_training.rl.single_turn.train_grpo import main  # noqa: PLC0415

    main(argv)


def gemma_sft(argv: list[str] | None = None) -> None:
    """Run the Gemma SFT training CLI."""
    _bootstrap()
    from sft.train_sft import main  # noqa: PLC0415

    main(argv)

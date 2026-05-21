"""CLI-level tests for ``--data-source index_lane_a`` on ``train_grpo``.

The new ``index_lane_a`` data source pairs the ManifestIndex (full ~25k+
RLVR allocation) with the procedural Lane A prefix generator so RL can
sample multi-step prefixes without consuming the 1716-row SFT corpus
(which is reserved exclusively for SFT per ``feedback_rl_data_pool_policy``).

These tests stay at the argparse + validation layer — no TRL imports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from single_turn_rl import train_grpo as tg


def _make_cfg(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[grpo]\n", encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# --help advertises the new choice
# ---------------------------------------------------------------------------


def test_train_grpo_help_advertises_index_lane_a(tmp_path: Path) -> None:
    """``index_lane_a`` must show up in the ``--data-source`` choice list +
    the human-readable help so operators can discover it without grep."""
    # Reach into the argparse help text via the parser, not via SystemExit.
    # ``_parse_args`` builds the parser fresh on each call; mirror that.
    # Re-construct the parser by invoking the same logic _parse_args uses,
    # except we capture format_help() instead of parsing argv. Easiest: drive
    # parser.parse_args(["--help"]) inside a SystemExit context and read the
    # captured stdout.
    import io
    import sys as _sys

    buf = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = buf
    try:
        with pytest.raises(SystemExit) as exc:
            tg._parse_args(["--help"])
        # argparse exits with 0 on --help.
        assert exc.value.code == 0
    finally:
        _sys.stdout = old_stdout

    out = buf.getvalue()
    assert "index_lane_a" in out, (
        "help text must mention the new --data-source choice"
    )
    # Sanity: existing choices still listed.
    assert "terminal_states" in out
    assert "--atlas-embedding-cache" in out
    # The new help blurb references the cache dependency.
    assert "atlas-embedding-cache" in out

    # Round-trip via _parse_args doesn't expose the underlying parser object,
    # so parse-time acceptance is confirmed below.


# ---------------------------------------------------------------------------
# Validation: index_lane_a requires --atlas-embedding-cache
# ---------------------------------------------------------------------------


def test_train_grpo_index_lane_a_requires_atlas_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``--data-source index_lane_a`` without ``--atlas-embedding-cache``
    must SystemExit at parse-time with a message that names BOTH flags.
    ``argparse.ArgumentParser.error`` writes the message to stderr and
    raises ``SystemExit(2)``, so we assert against the captured stderr."""
    cfg_path = _make_cfg(tmp_path)
    with pytest.raises(SystemExit):
        tg._parse_args(
            [
                "--config", str(cfg_path),
                "--sft-model", str(tmp_path),
                "--output-dir", str(tmp_path),
                "--data-source", "index_lane_a",
            ]
        )
    captured = capsys.readouterr()
    err = captured.err
    assert "index_lane_a" in err, (
        f"expected 'index_lane_a' in stderr error message; got: {err!r}"
    )
    assert "atlas-embedding-cache" in err, (
        f"expected 'atlas-embedding-cache' in stderr error message; got: {err!r}"
    )


# ---------------------------------------------------------------------------
# Acceptance: full Lane A invocation parses cleanly
# ---------------------------------------------------------------------------


def test_train_grpo_index_lane_a_parses_with_cache(tmp_path: Path) -> None:
    """``--data-source index_lane_a`` + ``--atlas-embedding-cache`` parses
    successfully, exposing the expected namespace."""
    cfg_path = _make_cfg(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    args = tg._parse_args(
        [
            "--config", str(cfg_path),
            "--sft-model", str(tmp_path),
            "--output-dir", str(tmp_path),
            "--data-source", "index_lane_a",
            "--atlas-embedding-cache", str(cache_dir),
        ]
    )
    assert args.data_source == "index_lane_a"
    assert args.atlas_embedding_cache == cache_dir


# ---------------------------------------------------------------------------
# Existing data-source choices still parse byte-for-byte (regression guard)
# ---------------------------------------------------------------------------


def test_train_grpo_default_data_source_unchanged(tmp_path: Path) -> None:
    cfg_path = _make_cfg(tmp_path)
    args = tg._parse_args(
        [
            "--config", str(cfg_path),
            "--sft-model", str(tmp_path),
            "--output-dir", str(tmp_path),
        ]
    )
    # Default stays "index" — index_lane_a must be opt-in.
    assert args.data_source == "index"


def test_train_grpo_index_data_source_does_not_require_cache(tmp_path: Path) -> None:
    """``--data-source index`` (existing, deterministic-slate path) must
    stay valid without an atlas-embedding cache."""
    cfg_path = _make_cfg(tmp_path)
    args = tg._parse_args(
        [
            "--config", str(cfg_path),
            "--sft-model", str(tmp_path),
            "--output-dir", str(tmp_path),
            "--data-source", "index",
        ]
    )
    assert args.data_source == "index"
    assert args.atlas_embedding_cache is None

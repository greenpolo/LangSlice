"""The style checker (``STYLE.md``, ``tools/stylecheck``) finds nothing new.

Runs ``python -m tools.stylecheck --against HEAD`` in a fresh interpreter:
it fails on any error the baseline (``tools/stylecheck/baseline.json``)
does not cover, on either baseline (the checker's or basedpyright's) grown
since the last commit, and on a setting loosened since then. The baselines
only shrink: fix a new error rather than listing it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "tools" / "stylecheck" / "baseline.json"


def test_style_checker_finds_nothing_new() -> None:
    if not BASELINE.exists():
        pytest.skip(
            "tools/stylecheck/baseline.json is not written yet "
            "(python -m tools.stylecheck --write-baseline creates it)"
        )
    done = subprocess.run(
        [sys.executable, "-m", "tools.stylecheck", "--against", "HEAD"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        cwd=REPO,
    )
    assert done.returncode == 0, done.stdout[-12000:] + done.stderr[-3000:]

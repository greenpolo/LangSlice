"""Golden snapshots: every linear tool's pictures, data and text, unchanged.

The safety net for the layered-core refactor. ``tests/golden/record.py``
scripts a fixed sequence of calls through every tool ``build_tools`` returns
(plus the engine's first model request and the MCP door) on a small synthetic
stack, and this test compares a fresh recording with the checked-in one in
``tests/golden/linear_tools/``: pictures pixel for pixel, data and text
exactly. No tolerance anywhere; the recording is byte-stable across runs,
core counts and ``OMP_NUM_THREADS`` because every fit pins its threads and
seed (``langslice.core.deformable.engines``).

The recorder runs in a fresh interpreter: ANTs fixes its thread count at
its first import in a process, and another test importing it first would
leave the count unknown (the fit would say so in its notes).

An intended change: regenerate with ``LANGSLICE_UPDATE_GOLDEN=1`` (or
``python -m tests.golden.record``), look at the changed pictures, and commit
them with the change that explains them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.golden.record import GOLDEN_DIR, canonical

REPO = Path(__file__).resolve().parents[1]
UPDATE_ENV = "LANGSLICE_UPDATE_GOLDEN"
#: Problems listed in a failure before the rest are only counted.
MAX_REPORTED = 40

_NEEDS = ("ants", "itk", "mcp", "google.adk")
_MISSING = [name for name in _NEEDS if importlib.util.find_spec(name.split(".")[0]) is None]


def _record(out: Path) -> None:
    env = {key: value for key, value in os.environ.items()
           if key not in ("LANGSLICE_TRACE_DIR", "LANGSLICE_VLM_DEBUG_DIR")}
    completed = subprocess.run(
        [sys.executable, "-m", "tests.golden.record", "--out", str(out)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600, check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"the golden recorder failed:\n{completed.stderr[-6000:]}")


def _first_difference(want: Any, got: Any, path: str = "") -> str:
    """Where two JSON values first differ, as a readable path."""
    if isinstance(want, dict) and isinstance(got, dict):
        for key in sorted(set(want) | set(got)):
            if key not in got:
                return f"{path}.{key}: missing (golden {json.dumps(want[key])[:200]})"
            if key not in want:
                return f"{path}.{key}: new (actual {json.dumps(got[key])[:200]})"
            if canonical(want[key]) != canonical(got[key]):
                return _first_difference(want[key], got[key], f"{path}.{key}")
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return f"{path}: length {len(want)} -> {len(got)}"
        for index, (a, b) in enumerate(zip(want, got, strict=True)):
            if canonical(a) != canonical(b):
                return _first_difference(a, b, f"{path}[{index}]")
    return f"{path}: {json.dumps(want)[:300]} -> {json.dumps(got)[:300]}"


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def _compare(golden: Path, actual: Path) -> list[str]:
    problems: list[str] = []
    want_files = {path.name for path in golden.iterdir()}
    got_files = {path.name for path in actual.iterdir()}
    for name in sorted(want_files - got_files):
        problems.append(f"{name}: in the goldens, not recorded now")
    for name in sorted(got_files - want_files):
        problems.append(f"{name}: recorded now, not in the goldens")
    for name in sorted(want_files & got_files):
        want_path, got_path = golden / name, actual / name
        if name.endswith(".json"):
            want_text = want_path.read_text(encoding="utf-8")
            got_text = got_path.read_text(encoding="utf-8")
            if want_text != got_text:
                where = _first_difference(json.loads(want_text), json.loads(got_text))
                problems.append(f"{name} (call {name.split('.')[0]}): data/text differ at {where}")
        elif name.endswith(".png"):
            want, got = _pixels(want_path), _pixels(got_path)
            call, index = name.split(".")[0], name.split(".")[1]
            if want.shape != got.shape:
                problems.append(f"{name} (call {call}, picture {index}): shape "
                                f"{want.shape} -> {got.shape}")
            elif not np.array_equal(want, got):
                differ = want != got
                count = int(differ.any(axis=-1).sum()) if differ.ndim == 3 else int(differ.sum())
                delta = int(np.abs(want.astype(int) - got.astype(int)).max())
                problems.append(f"{name} (call {call}, picture {index}): {count} of "
                                f"{want.shape[0] * want.shape[1]} pixels differ (max {delta})")
    return problems


@pytest.mark.skipif(bool(_MISSING), reason=f"needs {', '.join(_MISSING)}")
def test_every_linear_tool_matches_its_golden(tmp_path: Path) -> None:
    if os.environ.get(UPDATE_ENV) == "1":
        _record(GOLDEN_DIR)
        pytest.skip(f"{UPDATE_ENV}=1: goldens regenerated in {GOLDEN_DIR}")
    assert GOLDEN_DIR.is_dir(), f"no goldens; record them with {UPDATE_ENV}=1"
    actual = tmp_path / "actual"
    _record(actual)
    problems = _compare(GOLDEN_DIR, actual)
    if problems:
        shown = "\n".join(f"  {problem}" for problem in problems[:MAX_REPORTED])
        more = len(problems) - MAX_REPORTED
        pytest.fail(
            f"{len(problems)} golden difference(s):\n{shown}"
            + (f"\n  ... and {more} more" if more > 0 else "")
            + f"\nActual outputs: {actual}\nGoldens: {GOLDEN_DIR}\n"
            f"If the change is intended, regenerate with {UPDATE_ENV}=1 and look at the pictures.",
            pytrace=False,
        )

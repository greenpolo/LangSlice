"""The layered core's dependency rule, enforced by import-linter.

The contracts are in ``pyproject.toml`` (``[tool.importlinter]``): hosts >
doors : agent > ops > job > core; core, job and ops import no provider,
agent framework or model client; and providers import nothing above them.
No contract lists ``ignore_imports``: a violation is fixed by moving
code, never listed. ``tests/test_core_imports.py`` checks
the same rule at run time (what a fresh interpreter actually loads).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_import_layer_contracts_hold():
    done = subprocess.run(
        [sys.executable, "-c",
         "import sys; from importlinter.cli import lint_imports; "
         "sys.exit(lint_imports(config_filename='pyproject.toml', no_cache=True, no_logo=True))"],
        capture_output=True, text=True, timeout=600, check=False, cwd=REPO,
    )
    assert done.returncode == 0, done.stdout[-6000:] + done.stderr[-3000:]

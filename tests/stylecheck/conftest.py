"""Make the repository root importable, so the tests can import ``tools``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

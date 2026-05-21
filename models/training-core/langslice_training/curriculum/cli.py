"""Backward-compatible shim — curriculum.cli is now adaptive.curriculum.cli.

To run the CLI use: python -m adaptive.curriculum.cli
This shim exists so legacy invocations via python -m curriculum.cli still work.
"""

import sys

from langslice_training.adaptive.curriculum.cli import main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

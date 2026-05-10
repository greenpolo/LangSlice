"""Module entry point for ``python -m langslice_single_turn_rl`` from the repo root."""

from __future__ import annotations

import sys

from . import main


if __name__ == "__main__":
    main(sys.argv[1:])

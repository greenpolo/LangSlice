"""Module entry point for ``python -m langslice_rlvr`` from the repo root."""

from __future__ import annotations

import sys

from . import main


def _print_help() -> None:
    print(
        "usage: python -m langslice_rlvr --config CONFIG --sft-model SFT_MODEL "
        "--output-dir OUTPUT_DIR [--resume-from-adapter ADAPTER] "
        "[--test-images-root TEST_IMAGES_ROOT] [--seed SEED]\n\n"
        "Run multi-turn GRPO on the LangSlice estimation env.\n\n"
        "options:\n"
        "  --config CONFIG\n"
        "  --sft-model SFT_MODEL\n"
        "  --resume-from-adapter ADAPTER\n"
        "  --output-dir OUTPUT_DIR\n"
        "  --test-images-root TEST_IMAGES_ROOT\n"
        "  --seed SEED"
    )


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        _print_help()
    else:
        main(sys.argv[1:])

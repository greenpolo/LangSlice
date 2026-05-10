"""Module entry point for ``python -m langslice_rlvr`` from the repo root."""

from __future__ import annotations

import sys

from . import main


def _print_help() -> None:
    print(
        "usage: python -m langslice_rlvr --config CONFIG --sft-model SFT_MODEL "
        "--output-dir OUTPUT_DIR [--resume-from-adapter ADAPTER] "
        "[--manifest-root MANIFEST_ROOT] [--repo-root REPO_ROOT] "
        "[--report-to REPORT_TO] [--seed SEED] "
        "[--curriculum-weights PATH]\n\n"
        "Run multi-turn GRPO on the LangSlice estimation env.\n\n"
        "options:\n"
        "  --config CONFIG\n"
        "  --sft-model SFT_MODEL\n"
        "  --resume-from-adapter ADAPTER\n"
        "  --output-dir OUTPUT_DIR\n"
        "  --manifest-root MANIFEST_ROOT\n"
        "  --repo-root REPO_ROOT\n"
        "  --report-to REPORT_TO\n"
        "  --seed SEED\n"
        "  --curriculum-weights PATH"
    )


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        _print_help()
    else:
        main(sys.argv[1:])

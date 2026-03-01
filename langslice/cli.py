"""LangSlice CLI entry point."""
import argparse
import sys

import langslice


def main():
    parser = argparse.ArgumentParser(
        prog="langslice",
        description="VLM-based brain slice registration using Gemini and BrainGlobe atlases",
    )
    subparsers = parser.add_subparsers(dest="command")

    # langslice gui
    subparsers.add_parser("gui", help="Launch the PySide6 desktop application")

    # langslice version
    subparsers.add_parser("version", help="Print version info")

    args = parser.parse_args()

    if args.command == "gui":
        from langslice.gui import launch
        launch()
    elif args.command == "version":
        print(f"langslice {langslice.__version__}")
    else:
        parser.print_help()
        sys.exit(1)

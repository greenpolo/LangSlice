"""The commands that drive a host: ``gui`` (the LangSlice window: the agent
viewer and the prompt generator on job folders) and ``serve`` (the engine
service the Fiji connector starts).

The ``langslice`` command (:mod:`langslice.doors.commands.parser`) registers
them by module path and loads this module only when it builds its parser
(``HOST_COMMANDS``), so the doors never import a host.
"""

from __future__ import annotations

import argparse
from typing import cast


def add_gui_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "gui",
        help="Open the LangSlice window: the agent viewer on job folders as "
        "their runs write them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "folders", nargs="*", metavar="FOLDER",
        help="Job folders (or image folders with a langslice/ job folder) to "
        "show, besides the recently opened jobs",
    )
    p.add_argument("--port", type=int, default=0,
                   help="Serve on this local port (0: any free port)")
    p.add_argument("--no-window", action="store_true",
                   help="Do not open a window; print the address and keep "
                   "serving")
    p.add_argument(
        "--tab",
        choices=("viewer", "prompt"),
        default="viewer",
        help="The tab the window opens on; prompt opens the prompt "
        "generator, filled in from the first FOLDER's job",
    )


def run_gui(args: argparse.Namespace) -> None:
    from pathlib import Path

    from langslice.hosts.gui import serve
    from langslice.job.layout import JOB_DIRNAME, JOB_FILE

    folders = []
    for given in args.folders:
        folder = Path(given).expanduser().resolve()
        # An image folder stands for the job folder beside its images; a folder
        # that does not exist yet is a job folder a run is about to make.
        if not (folder / JOB_FILE).exists() and (folder / JOB_DIRNAME / JOB_FILE).exists():
            folder = folder / JOB_DIRNAME
        folders.append(folder)
    tab = cast("str", args.tab)
    serve(folders, port=args.port, window=not args.no_window, tab=tab)


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "serve",
        help="Run LangSlice engine service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stdio",
        action="store_true",
        help="Run newline-delimited JSON service over stdin/stdout",
    )


def run_serve(args: argparse.Namespace) -> None:
    if not args.stdio:
        raise SystemExit("serve currently requires --stdio")
    from langslice.hosts.service import run_stdio

    raise SystemExit(run_stdio())

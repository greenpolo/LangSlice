"""The commands that drive a host: ``abba`` (ABBA started from Python with the
LangSlice connector) and ``serve`` (the engine service the Fiji connector
starts).

The ``langslice`` command (:mod:`langslice.doors.cli`) registers them by
module path and loads this module only when it builds its parser
(``HOST_COMMANDS``), so the doors never import a host.
"""

from __future__ import annotations

import argparse


def add_abba_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "abba",
        help="Start ABBA (0.24.x, from Python) with the LangSlice connector; its runs "
        "can be watched in the Python agent viewer and log",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--abba-atlas",
        default="Adult Mouse Brain - Allen Brain Atlas V3p1",
        help="Atlas name passed to ABBA",
    )
    p.add_argument(
        "--connector-jar",
        default=None,
        metavar="JAR",
        help="The LangSlice Fiji connector jar (default: $LANGSLICE_CONNECTOR_JAR, else "
        "connectors/fiji/target/*.jar of this checkout)",
    )
    p.add_argument(
        "--no-viewer",
        action="store_true",
        help="Do not open the agent viewer that follows the connector's runs",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not open the agent activity log in a browser window",
    )


def run_abba(args: argparse.Namespace) -> None:
    from langslice.hosts.integrations.abba_launch import connector_jar, run_abba_session

    try:
        jar = connector_jar(args.connector_jar)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    run_abba_session(abba_atlas=args.abba_atlas, jar=str(jar), viewer=not args.no_viewer,
                     log=not args.no_log)


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
    from langslice.hosts.api.service import run_stdio

    raise SystemExit(run_stdio())

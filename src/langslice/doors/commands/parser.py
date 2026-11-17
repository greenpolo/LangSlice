"""The ``langslice`` command, one module per command group.

``langslice.cli:main`` (the installed entry point) is :func:`main` here.
Groups: :mod:`~langslice.doors.commands.linear` (``linear run`` and the shared
job flags), :mod:`~langslice.doors.commands.mcp` (``mcp``),
:mod:`~langslice.doors.commands.transcripts` (``session-end``, the harness
plugins' hook) and the host command (``gui``:
:mod:`langslice.hosts.cli`, loaded by module path through ``HOST_COMMANDS``, so
the doors never import a host). The agent CLI is a command of its own,
``langslice-job`` (:mod:`~langslice.doors.agent_cli.main`), so the verbs can be
allowed without these.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

import langslice

HOST_COMMANDS = {
    "gui": ("langslice.hosts.cli", "add_gui_parser", "run_gui"),
}


def _host_command(command: str, part: int) -> Any:
    module, *functions = HOST_COMMANDS[command]
    return getattr(importlib.import_module(module), functions[part])


def build_parser() -> argparse.ArgumentParser:
    from langslice.doors.commands.linear import add_run_parser
    from langslice.doors.commands.mcp import add_mcp_parser
    from langslice.doors.commands.transcripts import (
        add_parser as add_session_end_parser,
    )

    parser = argparse.ArgumentParser(
        prog="langslice",
        description="Register brain sections to BrainGlobe atlases with VLM agents and image-gen",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print version info")

    subparsers.add_parser(
        "login",
        help="Sign in with ChatGPT (OAuth) so LangSlice can use your subscription",
    )

    linear = subparsers.add_parser(
        "linear",
        help="Linear methods: section order, position and in-plane alignment",
    )
    linear_sub = linear.add_subparsers(dest="subcommand", required=True)
    add_run_parser(linear_sub)

    for command in HOST_COMMANDS:
        _host_command(command, 0)(subparsers)

    add_mcp_parser(subparsers)

    add_session_end_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one ``langslice`` command."""
    from langslice.doors.credentials import load_credentials

    parser = build_parser()
    args = parser.parse_args(argv)
    load_credentials(saved=args.command not in {"gui", "login", "version", "session-end"})

    command = getattr(args, "subcommand", None) or args.command

    if command == "version":
        print(f"langslice {langslice.__version__}")
    elif command == "login":
        from langslice.providers.oauth.login import login

        print(f"Signed in. Credentials saved to {login()}")
    elif command in HOST_COMMANDS:
        _host_command(command, 1)(args)
    elif command == "run":
        from langslice.doors.commands.linear import run_linear

        run_linear(args)
    elif command == "mcp":
        from langslice.doors.commands.mcp import run_mcp

        run_mcp(args)
    elif command == "session-end":
        from langslice.doors.commands.transcripts import run as run_session_end

        run_session_end(args)
    else:
        parser.print_help()
        sys.exit(1)

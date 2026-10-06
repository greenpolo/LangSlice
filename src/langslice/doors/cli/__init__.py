"""The ``langslice`` command, one module per command group.

``langslice.cli:main`` (the installed entry point) is :func:`main` here.
Groups: :mod:`~langslice.doors.cli.linear` (``linear run`` and the shared
job flags), :mod:`~langslice.doors.cli.claude` (``mcp``,
``claude prepare``) and the host commands (``abba``, ``serve``:
:mod:`langslice.hosts.cli`, loaded by module path through
``HOST_COMMANDS``, so the doors never import a host). The agent CLI is a
command of its own, ``langslice-job`` (:mod:`~langslice.doors.cli.jobcli`),
so the verbs can be allowed without these.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

import langslice
from langslice.doors.cli.claude import (
    add_claude_prepare_parser,
    add_mcp_parser,
    run_claude_prepare,
    run_mcp,
)
from langslice.doors.cli.linear import add_run_parser, run_linear

#: The commands that drive a host (the hosts layer), by module path: command
#: -> (module, its add-parser function, its run function). The module is
#: imported when the parser is built, never statically, so the doors never
#: import a host (import-linter's layers contract).
HOST_COMMANDS = {
    "abba": ("langslice.hosts.cli", "add_abba_parser", "run_abba"),
    "serve": ("langslice.hosts.cli", "add_serve_parser", "run_serve"),
}


def _host_command(command: str, part: int) -> Any:
    module, *functions = HOST_COMMANDS[command]
    return getattr(importlib.import_module(module), functions[part])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langslice",
        description="Register brain sections to BrainGlobe atlases with VLM agents and image-gen",
    )
    subparsers = parser.add_subparsers(dest="command")

    # langslice version
    subparsers.add_parser("version", help="Print version info")

    # langslice login
    subparsers.add_parser(
        "login",
        help="Sign in with ChatGPT (OAuth) so LangSlice can use your subscription",
    )

    # langslice linear <cmd> — position / affine estimation
    linear = subparsers.add_parser(
        "linear",
        help="Linear methods: section order, position and in-plane alignment",
    )
    linear_sub = linear.add_subparsers(dest="subcommand", required=True)
    add_run_parser(linear_sub)

    # langslice abba, langslice serve (the host commands, loaded on demand)
    for command in HOST_COMMANDS:
        _host_command(command, 0)(subparsers)

    # langslice mcp
    add_mcp_parser(subparsers)

    # langslice claude <cmd> — jobs for the Claude connector
    claude = subparsers.add_parser(
        "claude",
        help="Claude connector: prepare a job to paste into Claude Desktop or Claude Code",
    )
    claude_sub = claude.add_subparsers(dest="subcommand", required=True)
    add_claude_prepare_parser(claude_sub)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one ``langslice`` command."""
    # `.env` holds the API keys (GEMINI_API_KEY, OPENAI_API_KEY); every lane
    # reads it, not only the one whose module happens to be imported. The
    # keys saved by setup too, except for the commands that need none.
    from langslice.doors.api.setup import load_credentials

    parser = build_parser()
    args = parser.parse_args(argv)
    load_credentials(saved=args.command not in {"serve", "login", "version"})

    # Group commands (`linear`, `claude`) carry the leaf name in
    # `subcommand`; top-level commands only set `command`. Leaf names are
    # unique across groups, so one dispatch chain covers both.
    command = getattr(args, "subcommand", None) or args.command

    if command == "version":
        print(f"langslice {langslice.__version__}")
    elif command == "login":
        from langslice.providers.openai_oauth import login

        print(f"Signed in. Credentials saved to {login()}")
    elif command in HOST_COMMANDS:
        _host_command(command, 1)(args)
    elif command == "run":
        run_linear(args)
    elif command == "mcp":
        run_mcp(args)
    elif command == "prepare":
        run_claude_prepare(args)
    else:
        parser.print_help()
        sys.exit(1)

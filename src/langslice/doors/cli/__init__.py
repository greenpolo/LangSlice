"""The ``langslice`` command, one module per command group.

``langslice.cli:main`` (the installed entry point) is :func:`main` here.
Groups: :mod:`~langslice.doors.cli.linear` (``linear run``, ``linear
quick-affine`` and the shared job flags), :mod:`~langslice.doors.cli.register`
(``nonlinear register``), :mod:`~langslice.doors.cli.claude` (``mcp``,
``claude prepare``), the host commands (``abba``, ``serve``:
:mod:`langslice.hosts.cli`, loaded by module path through
``HOST_COMMANDS``, so the doors never import a host), and the agent CLI:
:mod:`~langslice.doors.cli.job` (``job FOLDER VERB``) and
:mod:`~langslice.doors.cli.catalog` (``ops``, ``schema``).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

import langslice
from langslice.doors.cli.catalog import add_parsers as add_catalog_parsers
from langslice.doors.cli.catalog import run_ops, run_schema
from langslice.doors.cli.claude import (
    add_claude_prepare_parser,
    add_mcp_parser,
    run_claude_prepare,
    run_mcp,
)
from langslice.doors.cli.job import add_parser as add_job_parser
from langslice.doors.cli.job import run as run_job
from langslice.doors.cli.linear import (
    add_quick_affine_parser,
    add_run_parser,
    run_linear,
    run_quick_affine,
)
from langslice.doors.cli.register import add_register_parser, run_register

#: The agent CLI's commands: JSON on stdout, an exit code returned.
AGENT_COMMANDS = {"job": run_job, "ops": run_ops, "schema": run_schema}

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
    add_quick_affine_parser(linear_sub)

    # langslice nonlinear <cmd> — image-gen registration
    nonlinear = subparsers.add_parser(
        "nonlinear",
        help="Nonlinear methods: image-gen registration",
    )
    nonlinear_sub = nonlinear.add_subparsers(dest="subcommand", required=True)
    add_register_parser(nonlinear_sub)

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

    # The agent CLI: langslice job FOLDER VERB, langslice ops, langslice schema
    add_job_parser(subparsers)
    add_catalog_parsers(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int | None:
    """Run one ``langslice`` command; the agent CLI's commands (``job``,
    ``ops``, ``schema``) return their exit code (0, 2, 3 or 4)."""
    # `.env` holds the API keys (GEMINI_API_KEY, OPENAI_API_KEY); every lane
    # reads it, not only the one whose module happens to be imported.
    from langslice.providers.openai_config import _load_dotenv

    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in AGENT_COMMANDS:
        if args.command == "job" and args.verb.replace("-", "_") == "trace_borders":
            from langslice.doors.api.setup import apply_saved_credentials

            apply_saved_credentials()  # the image model's keys
        return AGENT_COMMANDS[args.command](args)
    if args.command not in {"serve", "login", "version"}:
        from langslice.doors.api.setup import apply_saved_credentials

        apply_saved_credentials()

    # Group commands (`linear`, `nonlinear`) carry the leaf name in
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
    elif command == "register":
        run_register(args)
    elif command == "quick-affine":
        run_quick_affine(args)
    elif command == "run":
        run_linear(args)
    elif command == "mcp":
        run_mcp(args)
    elif command == "prepare":
        run_claude_prepare(args)
    else:
        parser.print_help()
        sys.exit(1)
    return None

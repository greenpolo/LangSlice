"""The ``langslice-job`` command: the agent CLI, the MCP tools as shell commands.

``langslice-job FOLDER VERB [arguments]``
(:mod:`langslice.doors.agent_cli.calls`), ``langslice-job FOLDER python SCRIPT
[ARGS ...]`` (the agent's own script, run by the job's environment:
:func:`langslice.doors.agent_cli.environment.python`), ``langslice-job ops``
and ``langslice-job schema [VERB]`` (:mod:`langslice.doors.agent_cli.catalog`).
stdout holds the one JSON envelope: whatever else prints there while a command
runs (a library's notices, native code) goes to stderr
(``envelope.stdout_to_stderr``). It works on job folders and nothing else: it
starts no agent, server or host and signs nobody in. Those are the
``langslice`` command's (:mod:`langslice.doors.commands.parser`), so a coding
agent allowed to run ``langslice-job`` can register a brain without being able
to start another agent. ``ops`` and ``schema`` are read as commands, so a job
folder of either name is given as a path (``./ops``).

Run as a program (:func:`main` without arguments: the installed command, a
background run's child), a job folder's command is answered by the job's warm
process when one is up (:mod:`langslice.doors.agent_cli.warm`); this module
imports nothing else until that is decided, so a warm call starts in
milliseconds.
"""

from __future__ import annotations

import argparse
import sys

#: The catalog commands, read before a job folder.
CATALOG = ("ops", "schema")


def build_parser(catalog: bool) -> argparse.ArgumentParser:
    from langslice.doors.agent_cli.calls import add_arguments
    from langslice.doors.agent_cli.catalog import (
        add_parsers as add_catalog_parsers,
    )

    parser = argparse.ArgumentParser(
        prog="langslice-job",
        description="The agent CLI: one verb on a LangSlice job folder, JSON on stdout. "
        "`langslice-job ops` lists the verbs, `langslice-job schema VERB` gives one's "
        "arguments.",
    )
    if catalog:
        add_catalog_parsers(parser.add_subparsers(dest="command", required=True))
    else:
        add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one agent-CLI command; its exit code (0, 2, 3 or 4; a script's own
    under ``python``). Without *argv* (the process's own command line) the
    job's warm process answers when it can; with *argv* the call runs here."""
    if argv is None:
        from langslice.doors.agent_cli import warm

        code = warm.run(sys.argv[1:])
        if code is not None:
            return code
    return run(list(sys.argv[1:] if argv is None else argv))


def run(argv: list[str]) -> int:
    """One agent-CLI command in this process (a cold call, or a warm
    process's child); its exit code."""
    # `.env` holds the API keys; a verb that needs a model gets the saved
    # keys when its job opens (doors.jobs).
    from langslice.doors.agent_cli.envelope import emit, stdout_to_stderr
    from langslice.doors.credentials import load_credentials

    catalog = bool(argv) and argv[0] in CATALOG
    args = build_parser(catalog).parse_args(argv)
    load_credentials(saved=False)
    if catalog:
        from langslice.doors.agent_cli.catalog import ops, schema

        with stdout_to_stderr():  # a job's atlas opened by schema prints nothing on stdout
            envelope = {"ops": ops, "schema": schema}[args.command](args)
        return emit(envelope)
    from langslice.doors.agent_cli.calls import run as run_job

    return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())

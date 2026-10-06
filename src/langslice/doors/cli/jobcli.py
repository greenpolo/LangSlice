"""The ``langslice-job`` command: the agent CLI, the MCP tools as shell commands.

``langslice-job FOLDER VERB [arguments]`` (:mod:`langslice.doors.cli.job`),
``langslice-job ops`` and ``langslice-job schema [VERB]``
(:mod:`langslice.doors.cli.catalog`). stdout holds the one JSON envelope:
whatever else prints there while a command runs (a library's notices,
native code) goes to stderr (``envelope.stdout_to_stderr``). It works on job folders and nothing
else: it starts no agent, server or host and signs nobody in. Those are the
``langslice`` command's (:mod:`langslice.doors.cli`), so a coding agent
allowed to run ``langslice-job`` can register a brain without being able to
start another agent. ``ops`` and ``schema`` are read as commands, so a job
folder of either name is given as a path (``./ops``).
"""

from __future__ import annotations

import argparse
import sys

from langslice.doors.cli.catalog import add_parsers as add_catalog_parsers
from langslice.doors.cli.catalog import ops, schema
from langslice.doors.cli.envelope import emit, stdout_to_stderr
from langslice.doors.cli.job import add_arguments
from langslice.doors.cli.job import run as run_job

#: The catalog commands, read before a job folder.
CATALOG = {"ops": ops, "schema": schema}


def build_parser(catalog: bool) -> argparse.ArgumentParser:
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
    """Run one agent-CLI command; its exit code (0, 2, 3 or 4)."""
    # `.env` holds the API keys; a verb that needs a model gets the saved
    # keys when its job opens (doors.jobs).
    from langslice.doors.api.setup import load_credentials

    argv = list(sys.argv[1:] if argv is None else argv)
    catalog = bool(argv) and argv[0] in CATALOG
    args = build_parser(catalog).parse_args(argv)
    load_credentials(saved=False)
    if catalog:
        with stdout_to_stderr():  # a job's atlas opened by schema prints nothing on stdout
            envelope = CATALOG[args.command](args)
        return emit(envelope)
    return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())

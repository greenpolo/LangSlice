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
folder of either name is given as a path (``./ops``). Every call runs in
its own process.
"""

from __future__ import annotations

import argparse
import sys

from langslice.doors.agent_cli import calls, catalog
from langslice.doors.agent_cli.envelope import emit, stdout_to_stderr
from langslice.doors.credentials import load_credentials

#: The catalog commands, read before a job folder.
CATALOG = ("ops", "schema")


def build_parser(of_catalog: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langslice-job",
        description="The agent CLI: one verb on a LangSlice job folder, JSON on stdout. "
        "`langslice-job ops` lists the verbs, `langslice-job schema VERB` gives one's "
        "arguments.",
    )
    if of_catalog:
        catalog.add_parsers(parser)
    else:
        calls.add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one agent-CLI command (*argv*, else the process's own command
    line); its exit code (0, 2, 3 or 4; a script's own under ``python``)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    of_catalog = bool(argv) and argv[0] in CATALOG
    args = build_parser(of_catalog).parse_args(argv)
    # `.env` holds the API keys; a verb that needs a model gets the saved
    # keys when its job opens (doors.jobs).
    load_credentials(saved=False)
    if of_catalog:
        # A job's atlas opened by schema prints nothing on stdout.
        with stdout_to_stderr():
            commands = {"ops": catalog.ops, "schema": catalog.schema}
            envelope = commands[argv[0]](args)
        return emit(envelope)
    return calls.run(args)


if __name__ == "__main__":
    raise SystemExit(main())

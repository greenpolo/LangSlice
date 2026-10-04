"""``langslice ops`` and ``langslice schema [VERB]``: what the agent CLI offers.

Both read the registry (:data:`langslice.ops.registry.VERBS`) and the
verbs' declarations (:mod:`langslice.doors.declarations`), the same ones the
agent and MCP tools are built from, so the list and the schemas are what
``langslice job FOLDER VERB`` accepts. ``schema`` is versioned
(:data:`SCHEMA_VERSION`); a verb is never renamed once shipped. A hidden
verb (``registry.Verb.hidden``) is in neither list; ``schema VERB`` still
answers for it by name.
"""

from __future__ import annotations

import argparse
from typing import Any

from langslice.doors.cli.envelope import Envelope, emit

#: The version of what ``langslice schema`` prints (bumped when a verb's
#: arguments change incompatibly).
SCHEMA_VERSION = 1

#: The agent CLI's own commands besides the verbs (``langslice job FOLDER ...``).
JOB_COMMANDS: dict[str, str] = {
    "init": "Create the job for a folder of section images (the job flags of "
            "`langslice linear run`; --resume continues an existing job).",
    "runs [ID]": "The background runs (newest first), or one run: running, finished "
                "(with its answer) or lost.",
    "wait [ID]": "Wait for a background run (the latest without ID); --timeout SECONDS.",
}


def canonical_verb(name: str) -> str:
    """A verb's name from what a caller typed (kebab-case accepted)."""
    return name.strip().replace("-", "_")


def verbs_table() -> list[dict[str, str]]:
    """Every verb: name, kind, group, one-line description."""
    from langslice.doors.declarations import summary
    from langslice.ops.registry import listed

    return [{"name": name, "kind": verb.kind, "group": verb.group, "summary": summary(name)}
            for name, verb in listed().items()]


def schema_of(name: str, job: str | None = None) -> dict[str, Any]:
    """The JSON schema of *name*'s arguments as the CLI accepts them (the
    caller sizes pictures: ``view.resolution``); with *job*, as that job's
    spec declares the verb (e.g. no ``engine`` where the user fixed it)."""
    from langslice.doors.declarations import Variant, arguments_schema

    variant = Variant(auto=True)
    if job:
        from langslice.doors.jobs import find, read_spec

        variant = Variant.of(read_spec(find(job)), auto=True)
    return arguments_schema(name, variant)


def add_parsers(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser("ops", help="List the verbs of the agent CLI (JSON)")
    schema = subparsers.add_parser("schema", help="The JSON schema of a verb's arguments")
    schema.add_argument("verb", nargs="?", default=None, help="One verb (default: every verb)")
    schema.add_argument("--job", default=None, metavar="FOLDER",
                        help="Declare the verb as this job's settings do")


def run_ops(_args: argparse.Namespace) -> int:
    return emit(Envelope(result={
        "verbs": verbs_table(),
        "usage": "langslice job FOLDER VERB --args '{...}' [--key value] [--dry-run] "
                 "[--background] [--verbose]",
        "job_commands": JOB_COMMANDS,
    }, next=["langslice schema VERB"]))


def run_schema(args: argparse.Namespace) -> int:
    from langslice.doors.jobs import NoJob
    from langslice.ops.registry import VERBS, listed

    try:
        if args.verb is None:  # every listed verb; a hidden one only by name
            result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "verbs": {
                name: schema_of(name, args.job) for name in listed()}}
        else:
            name = canonical_verb(args.verb)
            if name not in VERBS:
                return emit(Envelope.failure("UNKNOWN_VERB", f"No verb {args.verb!r}."))
            result = {"schema_version": SCHEMA_VERSION, "verb": name,
                      "arguments": schema_of(name, args.job)}
    except NoJob as exc:
        return emit(Envelope.failure("NO_JOB", str(exc)))
    return emit(Envelope(result=result))

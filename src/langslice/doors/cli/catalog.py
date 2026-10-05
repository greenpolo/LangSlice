"""``langslice ops`` and ``langslice schema [VERB]``: what the agent CLI offers.

Both read the registry (:data:`langslice.ops.registry.VERBS`) and the
verbs' declarations (:mod:`langslice.doors.declarations`), the same ones the
agent and MCP tools are built from, so the list and the schemas are what
``langslice job FOLDER VERB`` accepts. ``ops`` lists every verb (name, kind,
group, one line, ``long`` where it computes outside the job lock and may
take minutes). ``schema`` gives per verb what a model reads of it: the
whole declared description, the one-line summary, the argument schema, the
picture options described once (``picture_options``, the job statement's
text, with the ``view.resolution`` range) and ``long``; declared as the
job's settings declare it with ``--job FOLDER``, or the job of the folder it
runs in, otherwise every argument (``hint`` says how to narrow it).
``schema`` is versioned (:data:`SCHEMA_VERSION`); a verb is never renamed
once shipped. A hidden verb (``registry.Verb.hidden``) is in neither list;
``schema VERB`` still answers for it by name.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from langslice.doors.cli.envelope import EXIT_INTERNAL, EXIT_REFUSED, Envelope, emit

logger = logging.getLogger(__name__)

#: The version of what ``langslice schema`` prints (bumped when a verb's
#: arguments change incompatibly). 2: each verb an object with its
#: description, summary, arguments, picture options and ``long``.
SCHEMA_VERSION = 2

#: The agent CLI's own commands besides the verbs (``langslice job FOLDER ...``).
JOB_COMMANDS: dict[str, str] = {
    "init": "Create the job for a folder of section images (the job flags of "
            "`langslice linear run`, plus --notes TEXT and --viewer claude|codex|openai; "
            "a job already in the folder is continued). Answers with the job statement.",
    "brief": "Start here: LangSlice's job statement for this job (the one its own "
             "agent gets), the status table, the user's notes, and the opening pictures "
             "saved as files (artifacts of kind `opening`, in reading order); also "
             "written to BRIEF.md in the job folder.",
    "runs [ID]": "The background runs (newest first), or one run: running, finished "
                "(with its answer) or lost.",
    "wait [ID]": "Wait for a background run (the latest without ID); --timeout SECONDS.",
}


def canonical_verb(name: str) -> str:
    """A verb's name from what a caller typed (kebab-case accepted)."""
    return name.strip().replace("-", "_")


def verbs_table() -> list[dict[str, Any]]:
    """Every listed verb: name, kind, group, one-line description, and
    ``long: true`` for a long verb."""
    from langslice.doors.declarations import summary
    from langslice.ops.registry import listed

    return [{"name": name, "kind": verb.kind, "group": verb.group, "summary": summary(name),
             **({"long": True} if verb.long else {})}
            for name, verb in listed().items()]


class Declared:
    """How the verbs are declared for one ``schema`` call: the job's run (an
    open job: its settings, channels and viewer) or every argument."""

    def __init__(self, job: str | None) -> None:
        from langslice.doors.declarations import Variant

        self.folder: str | None = None
        self.variant = Variant(auto=True, door="cli")
        self.channels: Any = None
        self.atlas_channels: tuple[str, ...] | None = None
        self.max_edge: int | None = None
        found = job or _job_here()
        if found:
            self._open(found)

    def _open(self, folder: str) -> None:
        from langslice.agent.prompt import display_facts
        from langslice.doors.declarations import Variant
        from langslice.doors.jobs import open_folder

        opened = open_folder(folder, persist=False, door="cli")
        try:
            self.folder = str(opened.job.folder)
            self.variant = Variant.of(opened.spec, auto=True, door="cli",
                                      image_model=opened.image_model_connected)
            facts = display_facts(opened.ctx, opened.job.state)
            self.channels, self.atlas_channels = facts["channels"], facts["atlas_channels"]
            self.max_edge = opened.max_view_edge
        finally:
            opened.close()

    def entry(self, name: str) -> dict[str, Any]:
        """One verb as this call declares it."""
        from langslice.agent.prompt import PICTURE_TOOLS, display_lines
        from langslice.core.opening import DEFAULT_VIEWER, VIEWER_LIMITS
        from langslice.doors.declarations import arguments_schema, declaration
        from langslice.ops.registry import VERBS

        declared = declaration(name, self.variant)
        verb = VERBS[name]
        entry: dict[str, Any] = {
            "summary": declared.summary, "description": declared.doc,
            "kind": verb.kind, "group": verb.group, "long": verb.long,
            "arguments": arguments_schema(name, self.variant),
        }
        if name in PICTURE_TOOLS:
            edge = self.max_edge or VIEWER_LIMITS[DEFAULT_VIEWER][1]
            entry["picture_options"] = "\n".join(display_lines(
                [name], channels=self.channels, atlas_channels=self.atlas_channels,
                resolution=edge))
        return entry

    def facts(self) -> dict[str, Any]:
        """Which job the verbs are declared for, or how to declare them for one."""
        if self.folder:
            return {"job": self.folder}
        return {"job": None, "hint": "Declared without a job: every argument and option, "
                "pictures up to the default viewer's size. Add --job FOLDER (or run it in "
                "a job folder) for a job's own declarations, channels and picture sizes."}


def _job_here() -> str | None:
    """The job of the folder this runs in, if any."""
    from langslice.doors.jobs import NoJob, find

    try:
        return str(find(os.getcwd()))
    except (NoJob, OSError):
        return None


def add_parsers(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser("ops", help="List the verbs of the agent CLI (JSON)")
    schema = subparsers.add_parser(
        "schema", help="A verb's description, arguments and picture options (JSON)")
    schema.add_argument("verb", nargs="?", default=None, help="One verb (default: every verb)")
    schema.add_argument("--job", default=None, metavar="FOLDER",
                        help="Declare the verb as this job's settings do (default: the job "
                        "of the current folder, if any)")


def run_ops(_args: argparse.Namespace) -> int:
    return emit(Envelope(result={
        "verbs": verbs_table(),
        "usage": "langslice job FOLDER VERB --args '{...}' [--key value] [--dry-run] "
                 "[--background] [--verbose]",
        "long": "A long verb computes outside the job lock and may take minutes: run it "
                "with --background (answers at once with a run id), then `wait ID`.",
        "job_commands": JOB_COMMANDS,
    }, next=["langslice job FOLDER brief", "langslice schema VERB"]))


def run_schema(args: argparse.Namespace) -> int:
    from langslice.doors.jobs import NoJob
    from langslice.ops.registry import VERBS, listed

    try:
        declared = Declared(args.job)
        if args.verb is None:  # every listed verb; a hidden one only by name
            result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, **declared.facts(),
                                      "verbs": {name: declared.entry(name) for name in listed()}}
            nexts: list[str] = []
        else:
            name = canonical_verb(args.verb)
            if name not in VERBS:
                return emit(Envelope.failure("UNKNOWN_VERB", f"No verb {args.verb!r}."))
            result = {"schema_version": SCHEMA_VERSION, "verb": name, **declared.facts(),
                      **declared.entry(name)}
            folder = declared.folder or "FOLDER"
            nexts = [f"langslice job {folder} {name} --args '{{...}}'"
                     + (" --background" if VERBS[name].long else "")]
    except NoJob as exc:
        return emit(Envelope.failure("NO_JOB", str(exc)))
    except (FileNotFoundError, ValueError) as exc:
        return emit(Envelope.failure("JOB_UNREADABLE", str(exc), exit=EXIT_REFUSED))
    except Exception as exc:  # the envelope reports it; the traceback goes to stderr
        logger.exception("langslice schema failed")
        return emit(Envelope.failure("INTERNAL", f"{type(exc).__name__}: {exc}",
                                     exit=EXIT_INTERNAL))
    return emit(Envelope(result=result, next=nexts))

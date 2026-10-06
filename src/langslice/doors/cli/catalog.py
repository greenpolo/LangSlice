"""``langslice-job ops`` and ``langslice-job schema [VERB]``: what the agent CLI offers.

Both read the registry (:data:`langslice.ops.registry.VERBS`) and the
verbs' declarations (:mod:`langslice.doors.declarations`), the same ones the
agent and MCP tools are built from, so the list and the schemas are what
``langslice-job FOLDER VERB`` accepts. ``ops`` lists every verb (name, kind,
group, one line, ``long`` where it computes outside the job lock and may
take minutes). ``schema`` gives per verb what a model reads of it: the
whole declared description, the one-line summary, the argument schema, the
picture options described once (``picture_options``, the job statement's
text, with the ``view.resolution`` range) and ``long``; declared as the
job's settings declare it with ``--job FOLDER``, or the job of the folder it
runs in, otherwise every argument (``hint`` says how to narrow it).
``schema`` is versioned (:data:`SCHEMA_VERSION`); a verb is never renamed
once shipped. A hidden verb (``registry.Verb.hidden``) is in neither list;
``schema VERB`` still answers for it by name. ``schema`` of a job command
(``init``, ``brief``, ``runs``, ``wait``: :data:`JOB_COMMANDS`) gives its
summary, usage and flags (``init``'s are its parser's,
:func:`langslice.doors.cli.job.init_parser`); ``langslice-job FOLDER NAME
--help`` answers as ``schema NAME`` does (:func:`describe`). Both return
the envelope; :mod:`langslice.doors.cli.jobcli` prints it.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from langslice.doors.cli.envelope import EXIT_INTERNAL, EXIT_REFUSED, Envelope

logger = logging.getLogger(__name__)

#: The version of what ``langslice-job schema`` prints (bumped when a verb's
#: arguments change incompatibly). 2: each verb an object with its
#: description, summary, arguments, picture options and ``long``.
SCHEMA_VERSION = 2

#: The agent CLI's own commands besides the verbs (``langslice-job FOLDER ...``).
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


def command_names() -> list[str]:
    """The job commands by name (``init``, ``brief``, ``runs``, ``wait``)."""
    return [key.split()[0] for key in JOB_COMMANDS]


def flags_table(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    """An argparse parser's options as data: per option its ``flag`` (the
    long form), ``aliases``, ``help``, ``default``, ``choices``,
    ``takes_value`` and ``metavar``."""
    formatter = parser._get_formatter()
    rows: list[dict[str, Any]] = []
    for action in parser._actions:
        if not action.option_strings or isinstance(action, argparse._HelpAction):
            continue
        strings = sorted(action.option_strings, key=lambda text: (-len(text), text))
        try:
            text = formatter._expand_help(action) if action.help else ""
        except (KeyError, TypeError, ValueError):
            text = action.help or ""
        rows.append({
            "flag": strings[0], "aliases": strings[1:], "help": " ".join(text.split()),
            "default": action.default if action.default is not argparse.SUPPRESS else None,
            "choices": list(action.choices) if action.choices is not None else None,
            "takes_value": action.nargs != 0,
            "metavar": action.metavar if isinstance(action.metavar, str) else None,
        })
    return rows


def command_entry(name: str) -> dict[str, Any]:
    """A job command as ``schema`` gives it: its summary, usage and flags."""
    key = next(key for key in JOB_COMMANDS if key.split()[0] == name)
    entry: dict[str, Any] = {"summary": JOB_COMMANDS[key], "kind": "command",
                             "usage": f"langslice-job FOLDER {key}"}
    if name == "init":
        from langslice.doors.cli.job import init_parser

        entry["usage"] = "langslice-job IMAGE_FOLDER init [--flag value ...]"
        entry["flags"] = flags_table(init_parser())
    elif name == "wait":
        entry["flags"] = [{"flag": "--timeout", "aliases": [], "default": None,
                           "help": "Seconds to wait before answering with the run still "
                                   "running", "choices": None, "takes_value": True,
                           "metavar": "SECONDS"}]
    else:
        entry["flags"] = []
    return entry


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


def ops(_args: argparse.Namespace) -> Envelope:
    """``langslice-job ops``."""
    return Envelope(result={
        "verbs": verbs_table(),
        "usage": "langslice-job FOLDER VERB --args '{...}' [--key value] [--dry-run] "
                 "[--background] [--verbose]",
        "long": "A long verb computes outside the job lock and may take minutes: run it "
                "with --background (answers at once with a run id), then `wait ID`.",
        "job_commands": JOB_COMMANDS,
    }, next=["langslice-job FOLDER brief", "langslice-job schema VERB"])


def schema(args: argparse.Namespace) -> Envelope:
    """``langslice-job schema [VERB] [--job FOLDER]``."""
    from langslice.doors.jobs import NoJob
    from langslice.ops.registry import VERBS, listed

    if args.verb is not None and canonical_verb(args.verb) in command_names():
        name = canonical_verb(args.verb)  # init, brief, runs, wait: no job needed
        entry = command_entry(name)
        return Envelope(result={"schema_version": SCHEMA_VERSION, "verb": name, **entry},
                        next=[entry["usage"]])
    try:
        declared = Declared(args.job)
        if args.verb is None:  # every listed verb; a hidden one only by name
            result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, **declared.facts(),
                                      "verbs": {name: declared.entry(name) for name in listed()}}
            nexts: list[str] = []
        else:
            name = canonical_verb(args.verb)
            if name not in VERBS:
                return Envelope.failure("UNKNOWN_VERB", f"No verb {args.verb!r}.")
            result = {"schema_version": SCHEMA_VERSION, "verb": name, **declared.facts(),
                      **declared.entry(name)}
            folder = declared.folder or "FOLDER"
            nexts = [f"langslice-job {folder} {name} --args '{{...}}'"
                     + (" --background" if VERBS[name].long else "")]
    except NoJob as exc:
        return Envelope.failure("NO_JOB", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return Envelope.failure("JOB_UNREADABLE", str(exc), exit=EXIT_REFUSED)
    except Exception as exc:  # the envelope reports it; the traceback goes to stderr
        logger.exception("langslice-job schema failed")
        return Envelope.failure("INTERNAL", f"{type(exc).__name__}: {exc}",
                                exit=EXIT_INTERNAL)
    return Envelope(result=result, next=nexts)


def describe(name: str, folder: str) -> Envelope:
    """What ``langslice-job FOLDER NAME --help`` answers: ``schema NAME``,
    declared for FOLDER's job when it has one."""
    from langslice.doors.jobs import NoJob, find

    job: str | None = None
    if canonical_verb(name) not in command_names():
        try:
            job = str(find(folder))
        except (NoJob, OSError):
            job = None
    return schema(argparse.Namespace(verb=name, job=job))

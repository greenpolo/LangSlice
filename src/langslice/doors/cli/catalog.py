"""``langslice-job ops`` and ``langslice-job schema [VERB ...]``: what the agent CLI offers.

Both read the registry (:data:`langslice.ops.registry.VERBS`) and the
verbs' declarations (:mod:`langslice.doors.declarations`), the same ones the
agent and MCP tools are built from, so the list and the schemas are what
``langslice-job FOLDER VERB`` accepts. ``ops`` lists every verb (name, kind,
group, one line, ``long`` where it computes outside the job lock and may
take minutes). ``schema`` gives per verb what a model reads of it: the
whole declared description, the one-line summary, the argument schema and
``long``; one verb's entry as the result, several verbs' (separate
arguments or comma-separated) or every verb's under ``verbs``; declared as
the job's settings declare it with ``--job FOLDER`` (or as
``langslice-job FOLDER schema [VERB ...]``, :mod:`langslice.doors.cli.job`),
or the job of the folder it runs in, otherwise every argument (``hint``
says how to narrow it), and then a job's plugin verbs are unknown (the
refusal says how to name the job). ``schema`` is versioned (:data:`SCHEMA_VERSION`).
``schema`` of a job command
(``init``, ``brief``, ``runs``, ``wait``, ``env``, ``python``: :data:`JOB_COMMANDS`) gives its
summary, usage and flags (``init``'s are its parser's,
:func:`langslice.doors.cli.job.init_parser`); ``langslice-job FOLDER NAME
--help`` answers as ``schema NAME`` does (:func:`describe`). Both return
the envelope; :mod:`langslice.doors.cli.jobcli` prints it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from langslice.doors.cli.envelope import EXIT_INTERNAL, EXIT_REFUSED, Envelope

logger = logging.getLogger(__name__)

#: The version of what ``langslice-job schema`` prints (bumped when a verb's
#: arguments change incompatibly). 3: the redesigned toolbox (look, zoom,
#: one tool per registration method); each verb an object with its
#: description, summary, arguments and ``long``.
SCHEMA_VERSION = 3

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
    "wait [ID]": "Wait for a background run (the latest without ID): its answer, or "
                "after 30 s its state `running` (wait again); --timeout SECONDS.",
    "ops": "This job's own verbs: each one's kind, group, one-line summary and whether it "
           "is long.",
    "schema [VERB ...]": "This job's verbs as `langslice-job schema` gives them, its tool "
                         "plugins' included: several at once, every verb without one.",
    "env [install PACKAGE ...|remove]": "The job's own Python environment (made when "
        "missing; it also imports LangSlice's packages): its python, and `install` puts "
        "pip packages in it, listed in scripts/requirements.txt; `remove` deletes it and "
        "keeps that list, from which `env` rebuilds it.",
    "python SCRIPT [ARGS ...]": "Run your own Python script with the job's environment's "
        "python (`env`, made when missing): its output and exit code are the script's. "
        "Faster than calling that python yourself: LangSlice and the atlas are usually "
        "loaded already.",
}


def command_names() -> list[str]:
    """The job commands by name (``init``, ``brief``, ``runs``, ``wait``,
    ``ops``, ``schema``, ``env``, ``python``)."""
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
        # background.WAIT_S, not imported: `schema` runs without the job modules.
        entry["flags"] = [{"flag": "--timeout", "aliases": [], "default": 30.0,
                           "help": "Seconds to wait before answering with the run still "
                                   "running", "choices": None, "takes_value": True,
                           "metavar": "SECONDS"}]
    else:
        entry["flags"] = []
    return entry


def canonical_verb(name: str) -> str:
    """A verb's name from what a caller typed (kebab-case accepted)."""
    return name.strip().replace("-", "_")


#: What ``ops`` says of a long verb.
LONG_WORDS = ("A long verb computes outside the job lock and may take minutes; a call "
              "still running after 30 s answers with its run id and state running (the "
              "run goes on; `wait ID` collects its answer), and --background answers at "
              "once.")


def verbs_table(spec: Any = None) -> list[dict[str, Any]]:
    """Every verb: name, kind, group, one-line description, and
    ``long: true`` for a long verb. For the job *spec* of a folder that loads
    tool plugins, their tools follow the native ones."""
    from langslice.core.toolset import TOOLS
    from langslice.doors.declarations import Variant, summary
    from langslice.ops.registry import verbs_for

    variant = Variant.of(spec, auto=True, door="cli") if spec is not None and spec.plugins \
        else None
    return [{"name": name, "kind": verb.kind, "group": verb.group,
             "summary": summary(name, variant) if variant is not None and name not in TOOLS
             else summary(name),
             **({"long": True} if verb.long else {})}
            for name, verb in verbs_for(spec).items()]


class Declared:
    """How the verbs are declared for one ``schema`` call: the job's run (an
    open job: its settings, channels and viewer) or every argument."""

    def __init__(self, job: str | None) -> None:
        from langslice.doors.declarations import FULL, Variant

        self.folder: str | None = None
        self.spec: Any = None
        self.variant = Variant(auto=True, door="cli", uncapped=True, prompt=FULL.prompt)
        self.disabled: tuple[str, ...] = ()
        #: The open job's verbs (``Opened.listed_verbs``); None without a job.
        self.verbs: tuple[str, ...] | None = None
        found = job or _job_here()
        if found:
            self._open(found)

    def _open(self, folder: str) -> None:
        from langslice.doors.declarations import Variant
        from langslice.doors.jobs import open_folder

        opened = open_folder(folder, persist=False, door="cli")
        try:
            self.folder = str(opened.job.folder)
            self.spec = opened.spec
            self.variant = Variant.of(opened.spec, auto=True, door="cli",
                                      image_model=opened.image_model_connected,
                                      brain_scale=opened.ctx.brain_scale)
            self.disabled = tuple(opened.spec.switched_off)
            self.verbs = tuple(opened.listed_verbs())
        finally:
            opened.close()

    def has(self, name: str) -> bool:
        """The verb is the job's (every verb without a job)."""
        return self.verbs is None or name in self.verbs

    def entry(self, name: str) -> dict[str, Any]:
        """One verb as this call declares it."""
        from langslice.doors.declarations import arguments_schema, declaration
        from langslice.ops.registry import verbs_for

        declared = declaration(name, self.variant)
        verb = verbs_for(self.spec)[name]
        return {
            "summary": declared.summary, "description": declared.doc,
            "kind": verb.kind, "group": verb.group, "long": verb.long,
            "arguments": arguments_schema(name, self.variant),
        }

    def facts(self) -> dict[str, Any]:
        """Which job the verbs are declared for, or how to declare them for one."""
        if self.folder:
            return {"job": self.folder}
        return {"job": None, "hint": "Declared without a job: every argument. Add --job "
                "FOLDER (or run it in a job folder) for a job's own declarations."}


def _spec_here() -> Any:
    """The settings of the job in the current folder, if there is one and it
    reads (what ``ops`` lists the plugin tools of); None otherwise."""
    from langslice.doors.jobs import NoJob, find, read_spec

    try:
        return read_spec(find(os.getcwd()))
    except (NoJob, OSError, ValueError, FileNotFoundError):
        return None


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
        "schema", help="A verb's description and arguments (JSON)")
    schema.add_argument("verb", nargs="*", default=[],
                        help="One or more verbs (default: every verb)")
    schema.add_argument("--job", default=None, metavar="FOLDER",
                        help="Declare the verb as this job's settings do (default: the job "
                        "of the current folder, if any)")


def ops(args: argparse.Namespace) -> Envelope:
    """``langslice-job ops`` (the verbs, with the plugin tools of the job in the
    current folder), or ``langslice-job FOLDER ops`` (``args.job``: that job's
    own verbs only, as its settings declare them)."""
    folder = getattr(args, "job", None)
    if folder:
        from langslice.doors.jobs import NoJob

        try:
            declared = Declared(folder)
        except NoJob as exc:
            return Envelope.failure("NO_JOB", str(exc))
        except (FileNotFoundError, ValueError) as exc:
            return Envelope.failure("JOB_UNREADABLE", str(exc), exit=EXIT_REFUSED)
        from langslice.doors.declarations import summary

        table = [{**row, "summary": summary(row["name"], declared.variant)}
                 for row in verbs_table(declared.spec) if declared.has(row["name"])]
        job = declared.folder or folder
        return Envelope(result={
            "job": job, "verbs": table,
            "usage": f"langslice-job {job} VERB --args '{{...}}' [--key value] [--dry-run] "
                     "[--background] [--verbose]",
            "long": LONG_WORDS, "job_commands": JOB_COMMANDS,
        }, next=[f"langslice-job {job} brief", f"langslice-job {job} schema VERB [VERB ...]"])
    return Envelope(result={
        "verbs": verbs_table(_spec_here()),
        "usage": "langslice-job FOLDER VERB --args '{...}' [--key value] [--dry-run] "
                 "[--background] [--verbose]",
        "long": LONG_WORDS,
        "job_commands": JOB_COMMANDS,
    }, next=["langslice-job FOLDER brief", "langslice-job FOLDER schema VERB [VERB ...]"])


def _type_words(schema: dict[str, Any], defs: dict[str, Any]) -> str:
    """A JSON schema's type in words: ``string``, ``list of numbers``, one of
    the enum's values, an object's keys."""
    if "$ref" in schema:
        return _type_words(defs.get(schema["$ref"].rsplit("/", 1)[-1], {}), defs)
    if "enum" in schema:
        return "one of " + ", ".join(json.dumps(value) for value in schema["enum"])
    if "const" in schema:
        return json.dumps(schema["const"])
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return " or ".join(_type_words(part, defs) for part in schema[key])
    kind = schema.get("type")
    if kind == "array":
        return "list of " + _type_words(schema.get("items") or {}, defs)
    if kind == "object":
        keys = list((schema.get("properties") or {}).keys())
        if keys:
            need = set(schema.get("required") or [])
            return "{" + ", ".join(key + ("" if key in need else "?") for key in keys) + "}"
        return "object"
    return {"string": "string", "integer": "whole number", "number": "number",
            "boolean": "true or false", "null": "null"}.get(str(kind), str(kind or "any"))


def argument_lines(name: str, variant: Any) -> list[str]:
    """Verb *name*'s arguments as declared in *variant*, one line each: its
    name, type in words, and default (or ``required``)."""
    from langslice.doors.declarations import arguments_schema

    schema = arguments_schema(name, variant)
    defs = schema.get("$defs") or {}
    need = set(schema.get("required") or [])
    lines = []
    for argument, entry in (schema.get("properties") or {}).items():
        default = ("required" if argument in need
                   else f"default {json.dumps(entry.get('default'))}")
        lines.append(f"- `{argument}`: {_type_words(entry, defs)}; {default}")
    return lines


def tools_text(folder: str, names: list[str], variant: Any, spec: Any) -> str:
    """``TOOLS.md``: every verb of *names* (the job's, in order) as declared
    in *variant*, the run's declarations: its summary line, kind and group,
    its description and its arguments, the same words ``langslice-job FOLDER
    schema`` gives as JSON."""
    from langslice.doors.declarations import declaration
    from langslice.ops.registry import verbs_for

    verbs = verbs_for(spec)
    lines = [
        "# LangSlice tools",
        "",
        f"Every verb of this job as `langslice-job {folder} schema VERB` declares it, "
        "written by `brief`; the same words, as text. Call a verb as "
        f"`langslice-job {folder} VERB --args '{{...}}'`.",
        "",
    ]
    for name in names:
        if name not in verbs:
            continue
        declared = declaration(name, variant)
        verb = verbs[name]
        head = declared.doc.split("\n", 1)
        body = head[1] if len(head) > 1 else ""
        lines += [f"## {name}", "",
                  f"{declared.summary} ({verb.kind}, {verb.group}"
                  + (", long" if verb.long else "") + ")", ""]
        text = "\n".join(line[8:] if line.startswith(" " * 8) else line.strip()
                         for line in body.split("\n")).strip()
        if text:
            lines += [text, ""]
        arguments = argument_lines(name, variant)
        if arguments:
            lines += ["Arguments:", *arguments, ""]
    return "\n".join(lines).rstrip() + "\n"


def _names(given: Any) -> list[str]:
    """The verbs ``schema`` was given (one argument each, or comma-separated)."""
    if given is None:
        return []
    items = [given] if isinstance(given, str) else list(given)
    return [canonical_verb(part) for item in items for part in str(item).split(",")
            if part.strip()]


def schema(args: argparse.Namespace) -> Envelope:
    """``langslice-job schema [VERB ...] [--job FOLDER]`` (also
    ``langslice-job FOLDER schema [VERB ...]``): one verb's entry, several
    verbs' under ``verbs``, or every verb's."""
    from langslice.doors.jobs import NoJob
    from langslice.ops.registry import verbs_for

    names = _names(args.verb)
    commands = command_names()
    if len(names) == 1 and names[0] in commands:
        name = names[0]  # init, brief, runs, wait: no job needed
        entry = command_entry(name)
        return Envelope(result={"schema_version": SCHEMA_VERSION, "verb": name, **entry},
                        next=[entry["usage"]])
    try:
        declared = Declared(args.job)
        verbs = verbs_for(declared.spec)
        unknown = [name for name in names if name not in verbs and name not in commands]
        if unknown:
            message = f"No verb {', '.join(repr(name) for name in unknown)}."
            if declared.folder is None:
                message += (" A job's tool plugins add verbs of their own, known with its "
                            "folder: `langslice-job FOLDER schema "
                            + " ".join(names) + "`.")
            return Envelope.failure("UNKNOWN_VERB", message)
        off = [name for name in names if name in verbs and not declared.has(name)]
        if off:
            return Envelope.failure(
                "VERB_OFF", f"This job's settings have no {', '.join(off)}.",
                job=declared.folder or "<folder>", verb=off[0])
        nexts: list[str] = []
        if len(names) == 1:
            name = names[0]
            result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "verb": name,
                                      **declared.facts(), **declared.entry(name)}
            folder = declared.folder or "FOLDER"
            nexts = [f"langslice-job {folder} {name} --args '{{...}}'"]
        else:  # several verbs, or every verb
            chosen = names or [name for name in verbs if declared.has(name)]
            result = {"schema_version": SCHEMA_VERSION, **declared.facts(),
                      "verbs": {name: command_entry(name) if name in commands
                                else declared.entry(name) for name in chosen}}
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
    return schema(argparse.Namespace(verb=[name], job=job))

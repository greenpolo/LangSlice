"""``langslice job FOLDER ops`` and ``schema [VERB ...]``: what the job commands offer.

Both read the registry (:data:`langslice.tools.catalog.VERBS`) and the verbs'
declarations (:mod:`langslice.tools.descriptions.render`), the same ones the
agent and MCP tools are built from, so the list and the schemas are what
``langslice job FOLDER VERB`` accepts. ``ops`` lists every verb (name, kind,
group, one line, ``long`` where it computes outside the job lock and may take
minutes). ``schema`` gives per verb what a model reads of it: the whole
declared description, the one-line summary, the argument schema and ``long``;
one verb's entry as the result, several verbs' (separate arguments or
comma-separated) or every verb's under ``verbs``; declared as the folder's job
settings declare it, or, where the folder holds no job, every argument
(``hint`` says so, and a job's plugin verbs are unknown). ``schema`` is
versioned (:data:`SCHEMA_VERSION`). ``schema`` of a job command (``init``,
``launch``, ``runs``, ``wait``, ``env``, ``python``: :data:`JOB_COMMANDS`)
gives its summary, usage and flags (:func:`langslice.doors.commands.job.parsers.flag_rows_of`);
``langslice job FOLDER NAME --help`` answers as ``schema NAME`` does
(:func:`describe`). Both return the envelope; :func:`langslice.doors.commands.job.calls.run`
prints it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langslice.core.refusal import RefusalCode
from langslice.doors.commands.job.envelope import (
    EXIT_INTERNAL,
    EXIT_REFUSED,
    Envelope,
)
from langslice.doors.commands.job.parsers import FLAG_GROUPS, flag_rows_of
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tools_on import ToolsOn
from langslice.tools.descriptions.description import DescriptionFacts
from langslice.tools.descriptions.render import (
    FULL,
    arguments_schema,
    declaration,
    summary,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

CLI_VERBS = frozenset({"status", "export_maps", "review_comments"})


def on_command_line(verb: str) -> bool:
    """Tell whether ``langslice job FOLDER`` runs a verb (:data:`CLI_VERBS`).

    Parameters
    ----------
    verb
        The verb's name.

    Returns
    -------
    bool
        True for the verbs of a session that sets a job up and checks it.
    """
    return verb in CLI_VERBS


JOB_COMMANDS: dict[str, str] = {
    "init": "Create the job for a folder of section images (the job flags of "
    "`langslice linear run`, plus --notes TEXT and --viewer claude|codex|openai; "
    "a job already in the folder is continued).",
    "launch --role register|register-scripting": "The exact prompt that launches a "
    "registration subagent (`result.prompt`): pass it unchanged as the subagent's "
    "whole prompt. `register` is the subagent with the MCP tools alone, "
    "`register-scripting` the one with a shell for scripts of its own as "
    "well (its launch makes the job's Python environment); the prompt is "
    "also saved in logs/launches/. Run it again "
    "for each launch: it reflects the job as it stands.",
    "runs [ID]": "The background runs (newest first), or one run: running, finished "
    "(with its answer) or lost.",
    "wait [ID]": "Wait for a background run (the latest without ID): its answer, or "
    "after 30 s its state `running` (wait again); --timeout SECONDS.",
    "ops": "This job's own verbs: each one's kind, group, one-line summary and whether it is long.",
    "schema [VERB ...]": "This job's verbs as `langslice job FOLDER schema` gives them, its tool "
    "plugins' included: several at once, every verb without one.",
    "env [install PACKAGE ...|remove]": "The job's own Python environment (made when "
    "missing; it also imports LangSlice's packages): its python, and `install` puts "
    "pip packages in it, listed in scripts/requirements.txt; `remove` deletes it and "
    "keeps that list, from which `env` rebuilds it.",
    "python SCRIPT [ARGS ...]": "Run your own Python script with the job's environment's "
    "python (`env`, made when missing): its output and exit code are the script's.",
}


def command_names() -> list[str]:
    """The job commands by name (``init``, ``launch``, ``runs``, ``wait``,
    ``ops``, ``schema``, ``env``, ``python``)."""
    return [key.split()[0] for key in JOB_COMMANDS]


def command_entry(name: str) -> dict[str, Any]:
    """A job command as ``schema`` gives it: its summary, usage and flags."""
    key = next(key for key in JOB_COMMANDS if key.split()[0] == name)
    entry: dict[str, Any] = {
        "summary": JOB_COMMANDS[key],
        "kind": "command",
        "usage": f"langslice job FOLDER {key}",
    }
    if name == "init":
        entry["usage"] = "langslice job IMAGE_FOLDER init [--flag value ...]"
    entry["flags"] = flag_rows_of(name) if name in FLAG_GROUPS else []
    return entry


def canonical_verb(name: str) -> str:
    """A verb's name from what a caller typed (kebab-case accepted)."""
    return name.strip().replace("-", "_")


LONG_WORDS = (
    "A long verb computes outside the job lock and may take minutes; a call "
    "still running after 30 s answers with its run id and state running (the "
    "run goes on; `wait ID` collects its answer), and --background answers at "
    "once."
)


def verbs_table(spec: JobSpec | None = None) -> list[dict[str, Any]]:
    """Every verb: name, kind, group, one-line description, and
    ``long: true`` for a long verb. The engines' tools follow the others
    (for the job *spec* of a folder, the engines its run has a tool of),
    then the tools its plugins add."""
    from langslice.tools.catalog import VERBS, verbs_for

    facts = FULL
    if spec is not None:
        facts = DescriptionFacts.from_spec(spec, auto=True, door="cli")
    return [
        {
            "name": name,
            "kind": verb.kind,
            "group": verb.group,
            "summary": summary(name, FULL if name in VERBS else facts),
            **({"long": True} if verb.long else {}),
        }
        for name, verb in verbs_for(spec).items()
        if on_command_line(name)
    ]


class Declared:
    """How the verbs are declared for one ``schema`` call: the job's run (an
    open job: its settings, channels and viewer) or every argument."""

    def __init__(self, job: str | None) -> None:
        self.folder: str | None = None
        self.spec: JobSpec | None = None
        self.variant = DescriptionFacts(auto=True, door="cli", uncapped=True, prompt=FULL.prompt)
        self.on = ToolsOn()
        self.verbs: tuple[str, ...] | None = None
        if job:
            self._open(job)

    def _open(self, folder: str) -> None:
        from langslice.doors.jobs import open_folder

        opened = open_folder(folder, persist=False, door="cli")
        try:
            self.folder = str(opened.job.folder)
            self.spec = opened.spec
            self.variant = DescriptionFacts.from_spec(
                opened.spec,
                auto=True,
                door="cli",
                image_model=opened.image_model_connected,
                brain_scale=opened.ctx.brain_scale,
            )
            self.on = opened.spec.tools_on
            self.verbs = tuple(opened.listed_verbs())
        finally:
            opened.close()

    def has(self, name: str) -> bool:
        """The verb is the job's (every verb without a job)."""
        return self.verbs is None or name in self.verbs

    def entry(self, name: str) -> dict[str, Any]:
        """One verb as this call declares it."""
        from langslice.tools.catalog import verbs_for

        declared = declaration(name, self.variant)
        verb = verbs_for(self.spec)[name]
        return {
            "summary": declared.summary,
            "description": declared.doc,
            "kind": verb.kind,
            "group": verb.group,
            "long": verb.long,
            "arguments": arguments_schema(name, self.variant),
        }

    def facts(self) -> dict[str, Any]:
        """Which job the verbs are declared for, or how to declare them for one."""
        if self.folder:
            return {"job": self.folder}
        return {
            "job": None,
            "hint": "Declared without a job: every argument. Give the job's folder "
            "for a job's own declarations.",
        }


def _job_of(folder: str) -> str | None:
    """The job *folder* names, or None when it holds none."""
    from langslice.doors.jobs import NoJob, find

    try:
        return str(find(folder))
    except (NoJob, OSError):
        return None


def ops(folder: str) -> Envelope:
    """``langslice job FOLDER ops``: the job's own verbs, as its settings
    declare them."""
    from langslice.doors.jobs import NoJob, find

    try:
        declared = Declared(str(find(folder)))
    except NoJob as exc:
        return Envelope.failure(RefusalCode.NOT_FOUND, str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return Envelope.failure(RefusalCode.IO_ERROR, str(exc), exit=EXIT_REFUSED)
    table = [
        {**row, "summary": summary(row["name"], declared.variant)}
        for row in verbs_table(declared.spec)
        if declared.has(row["name"])
    ]
    job = declared.folder or folder
    return Envelope(
        result={
            "job": job,
            "verbs": table,
            "usage": f"langslice job {job} VERB --args '{{...}}' [--key value] [--dry-run] "
            "[--background] [--verbose]",
            "long": LONG_WORDS,
            "job_commands": JOB_COMMANDS,
        },
        next=[f"langslice job {job} schema VERB [VERB ...]"],
    )


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
    return {
        "string": "string",
        "integer": "whole number",
        "number": "number",
        "boolean": "true or false",
        "null": "null",
    }.get(str(kind), str(kind or "any"))


def argument_lines(name: str, variant: Any) -> list[str]:
    """Verb *name*'s arguments as declared in *variant*, one line each: its
    name, type in words, and default (or ``required``)."""
    schema = arguments_schema(name, variant)
    defs = schema.get("$defs") or {}
    need = set(schema.get("required") or [])
    lines = []
    for argument, entry in (schema.get("properties") or {}).items():
        default = "required" if argument in need else f"default {json.dumps(entry.get('default'))}"
        lines.append(f"- `{argument}`: {_type_words(entry, defs)}; {default}")
    return lines


def tools_text(folder: str, names: list[str], variant: Any, spec: Any) -> str:
    """``TOOLS.md``: every verb of *names* (the job's, in order) as declared
    in *variant*, the run's declarations: its summary line, kind and group,
    its description and its arguments, the same words ``langslice job FOLDER
    schema`` gives as JSON."""
    from langslice.tools.catalog import verbs_for

    verbs = verbs_for(spec)
    lines = [
        "# LangSlice tools",
        "",
        f"Every tool of this job as `langslice job {folder} schema TOOL` declares it, "
        "written by `init`; the same words, as text. A registration agent has "
        "them as MCP tools; a script calls one as "
        f'`langslice.open_job(r"{folder}").TOOL(...)`, with the same arguments.',
        "",
    ]
    for name in names:
        if name not in verbs:
            continue
        declared = declaration(name, variant)
        verb = verbs[name]
        lines += [
            f"## {name}",
            "",
            f"{declared.summary} ({verb.kind}, {verb.group}"
            + (", long" if verb.long else "")
            + ")",
            "",
        ]
        if declared.text:
            lines += [declared.text, ""]
        arguments = argument_lines(name, variant)
        if arguments:
            lines += ["Arguments:", *arguments, ""]
    return "\n".join(lines).rstrip() + "\n"


def _names(given: list[str]) -> list[str]:
    """The verbs ``schema`` was given (one argument each, or comma-separated)."""
    return [canonical_verb(part) for item in given for part in item.split(",") if part.strip()]


def schema(folder: str, given: list[str]) -> Envelope:
    """``langslice job FOLDER schema [VERB ...]``: one verb's entry, several
    verbs' under ``verbs``, or every verb's; declared for the folder's job,
    or with every argument when it holds none."""
    from langslice.doors.jobs import NoJob
    from langslice.tools.catalog import verbs_for

    names = _names(given)
    commands = command_names()
    if len(names) == 1 and names[0] in commands:
        name = names[0]
        entry = command_entry(name)
        return Envelope(
            result={"schema_version": SCHEMA_VERSION, "verb": name, **entry}, next=[entry["usage"]]
        )
    try:
        declared = Declared(_job_of(folder))
        verbs = verbs_for(declared.spec)
        unknown = [name for name in names if name not in verbs and name not in commands]
        if unknown:
            message = f"No verb {', '.join(repr(name) for name in unknown)}."
            if declared.folder is None:
                message += (
                    " This folder holds no job, and a job's tool plugins add verbs of their own."
                )
            return Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                f"{message} List the verbs with `langslice job FOLDER ops`.",
            )
        off = [name for name in names if name in verbs and not declared.has(name)]
        if off:
            return Envelope.failure(
                RefusalCode.TOOL_OFF,
                f"This job's settings have no {', '.join(off)}. "
                f"`langslice job {declared.folder or '<folder>'} status` lists the job's verbs "
                "under `verbs`.",
            )
        nexts: list[str] = []
        if len(names) == 1:
            name = names[0]
            result: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "verb": name,
                **declared.facts(),
                **declared.entry(name),
            }
            folder = declared.folder or "FOLDER"
            nexts = [f"langslice job {folder} {name} --args '{{...}}'"]
            if not on_command_line(name):
                nexts = [f'langslice.open_job(r"{folder}").{name}(...)']
        else:
            chosen = names or [name for name in verbs if declared.has(name)]
            result = {
                "schema_version": SCHEMA_VERSION,
                **declared.facts(),
                "verbs": {
                    name: command_entry(name) if name in commands else declared.entry(name)
                    for name in chosen
                },
            }
    except NoJob as exc:
        return Envelope.failure(RefusalCode.NOT_FOUND, str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return Envelope.failure(RefusalCode.IO_ERROR, str(exc), exit=EXIT_REFUSED)
    except Exception as exc:
        logger.exception("langslice job schema failed")
        return Envelope.failure(
            RefusalCode.INTERNAL, f"{type(exc).__name__}: {exc}", exit=EXIT_INTERNAL
        )
    return Envelope(result=result, next=nexts)


def describe(name: str, folder: str) -> Envelope:
    """What ``langslice job FOLDER NAME --help`` answers: ``schema NAME``."""
    return schema(folder, [name])

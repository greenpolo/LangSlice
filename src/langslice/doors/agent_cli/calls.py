"""``langslice-job FOLDER VERB [arguments]``: one verb on a job folder, for agents.

The agent CLI: the verbs of a session that sets a job up and checks it
(:data:`langslice.doors.agent_cli.catalog.CLI_VERBS`, kebab-case accepted:
``export-maps``; any other verb is a registration tool and is refused,
``TOOL_OFF``), plus ``init`` (create the job for a folder of images),
``launch --role ROLE`` (the launch prompt of a
registration subagent, :mod:`langslice.doors.agent_cli.launch`), ``runs [ID]``
and ``wait [ID]`` (background runs; ``status`` is only the verb). FOLDER is the
job folder or the image folder beside it.

Arguments are the verb's declared arguments
(:mod:`langslice.tools.descriptions.render`, ``langslice-job schema VERB``): a
JSON object with ``--args '{...}'`` or ``--args @file.json``, and/or one flag
per argument, ``--name value`` (kebab or snake case; a JSON value, or plain
text for a text argument; a list argument takes the flag again for each item).
Options: ``--dry-run`` (``export_maps`` lists the files it would write and
writes none; not with ``--background``), ``--background`` (answer at once with a run id; without
it a long verb runs as a background run all the same where a process may
outlive its command, the call waiting up to ``background.WAIT_S`` for it,
:func:`_detached`), ``--verbose`` (the whole reply: whole-stack rows, the
descriptions written for a model), ``--timeout SECONDS`` (``wait``).

Each call opens the job as it stands on disk (``doors.jobs.open_folder``), runs
the verb through the same tool a running agent has and closes it, so a call
sees what a running agent has written (``Job.sync``). The answer is the JSON
envelope of :mod:`langslice.doors.agent_cli.envelope` on stdout; anything a
library prints goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langslice.core.refusal import FAILED_STATUSES, RefusalCode, ReplyStatus
from langslice.doors.agent_cli import background
from langslice.doors.agent_cli.catalog import (
    CLI_VERBS,
    canonical_verb,
    on_command_line,
)
from langslice.doors.agent_cli.envelope import (
    EXIT_ARGUMENTS,
    EXIT_INTERNAL,
    EXIT_OK,
    EXIT_REFUSED,
    Envelope,
    dumps,
    emit,
    stdout_to_stderr,
)
from langslice.doors.jobs import Opened
from langslice.job.folder.lock import JobBusy
from langslice.tools.descriptions.door_text import NOTES_KEY, image_model_state
from langslice.tools.descriptions.render import Declaration, declaration

if TYPE_CHECKING:
    from collections.abc import Callable

    from langslice.doors.commands.flags import FlagGroup
    from langslice.job.job import Job

logger = logging.getLogger(__name__)

WARN_KEYS = (
    "unknown_ids",
    "unknown",
    "rejected",
    "render_failed",
    "clamped",
    "deformation_cleared",
    "truncated",
    "dropped_positions_mm",
    "not_shown",
)
HELP_FLAGS = frozenset({"--help", "-h"})
ROW_VERBS = frozenset({"status"})
INIT_FLAG_GROUPS: tuple[FlagGroup, ...] = ("job", "resume", "init")


def progress(message: str) -> None:
    """The job's progress lines: stderr, never stdout."""
    print(message, file=sys.stderr, flush=True)


class _Refusal(Exception):
    """Carry the agent CLI's own refusal out of a call, its envelope made.

    The envelope is made where the refusal is raised and :func:`execute`
    answers it as it stands.
    """

    def __init__(self, envelope: Envelope) -> None:
        super().__init__(envelope.error)
        self.envelope = envelope


class _Parser(argparse.ArgumentParser):
    """argparse that raises instead of exiting (the envelope reports it)."""

    def error(self, message: str) -> typing.NoReturn:
        raise _Refusal(Envelope.failure(RefusalCode.BAD_ARGUMENT, message))


def add_arguments(p: argparse.ArgumentParser) -> None:
    """``FOLDER VERB [arguments]``, the ``langslice-job`` command's
    (:mod:`langslice.doors.agent_cli.main`)."""
    p.add_argument("folder", help="The job folder, or the image folder beside it")
    p.add_argument(
        "verb",
        help="A verb (`langslice-job ops`), init, launch --role ROLE, "
        "runs [ID], wait [ID], env or python SCRIPT [ARGS ...]",
    )
    p.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="--args JSON|@file, --name value, --dry-run, --background, "
        "--verbose, --timeout SECONDS",
    )


def run(args: argparse.Namespace) -> int:
    if canonical_verb(args.verb) == "python" and not HELP_FLAGS & set(args.rest[:1]):
        from langslice.doors.agent_cli.environment import python

        return python(args.folder, list(args.rest))
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    with stdout_to_stderr():
        envelope = execute(args.folder, args.verb, list(args.rest))
    return emit(envelope)


def execute(folder: str, verb: str, rest: list[str], *, atlas_loader: Any = None) -> Envelope:
    """One agent-CLI call (see the module text); never raises. Every call on
    a job is logged in its folder (:func:`record`)."""
    started = time.perf_counter()
    name = canonical_verb(verb)
    run_id = None if name == "init" else _run_id(rest)
    try:
        envelope = _execute(folder, name, rest, atlas_loader=atlas_loader)
    except _Refusal as refusal:
        envelope = refusal.envelope
    except JobBusy as exc:
        envelope = Envelope.failure(RefusalCode.BUSY, str(exc), exit=EXIT_REFUSED)
    except Exception as exc:
        logger.exception("langslice-job %s %s failed", folder, verb)
        envelope = Envelope.failure(
            RefusalCode.INTERNAL, f"{type(exc).__name__}: {exc}", exit=EXIT_INTERNAL
        )
    if run_id:
        _finish_run(folder, run_id, envelope)
    record(folder, name, rest, envelope, seconds=time.perf_counter() - started, run_id=run_id)
    return envelope


def _run_id(rest: list[str]) -> str | None:
    """A background child's ``--run-id`` (:mod:`.background`)."""
    for index, token in enumerate(rest):
        if token.startswith("--run-id="):
            return token.partition("=")[2] or None
        if token in ("--run-id", "--run_id") and index + 1 < len(rest):
            return rest[index + 1]
    return None


def _execute(folder: str, name: str, rest: list[str], *, atlas_loader: Any) -> Envelope:
    if HELP_FLAGS & set(rest):
        from langslice.doors.agent_cli.catalog import describe

        return describe(name, folder)
    if name == "init":
        return init(folder, rest, atlas_loader=atlas_loader)
    if name == "ops":
        from langslice.doors.agent_cli.catalog import ops
        from langslice.doors.jobs import NoJob, find

        if rest:
            raise _Refusal(Envelope.failure(RefusalCode.BAD_ARGUMENT, "ops takes no arguments."))
        try:
            return ops(argparse.Namespace(job=str(find(folder))))
        except NoJob as exc:
            return Envelope.failure(RefusalCode.NOT_FOUND, str(exc))
    if name == "schema":
        from langslice.doors.agent_cli.catalog import schema
        from langslice.doors.jobs import NoJob, find

        try:
            job = str(find(folder))
        except NoJob as exc:
            return Envelope.failure(RefusalCode.NOT_FOUND, str(exc))
        named = [
            token
            for index, token in enumerate(rest)
            if not token.startswith("-") and (index == 0 or rest[index - 1] != "--job")
        ]
        return schema(argparse.Namespace(verb=named, job=job))
    if name == "launch":
        return launch(folder, rest, atlas_loader=atlas_loader)
    options, arguments, positional = parse(rest)
    if name in ("runs", "wait"):
        return runs(folder, name, positional, options)
    if name == "env":
        from langslice.doors.agent_cli import environment as jobenv
        from langslice.doors.jobs import NoJob, find
        from langslice.job.folder.layout import JobLayout

        try:
            layout = JobLayout(find(folder))
        except NoJob as exc:
            return Envelope.failure(RefusalCode.NOT_FOUND, str(exc))
        return jobenv.command(layout, positional)
    if positional:
        raise _Refusal(
            Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                f"Unexpected value(s) {positional}: give arguments as --name value or --args JSON.",
            )
        )
    return call(folder, name, arguments, options, atlas_loader=atlas_loader)


def record(
    folder: str,
    verb: str,
    rest: list[str],
    envelope: Envelope,
    *,
    seconds: float,
    run_id: str | None = None,
) -> None:
    """Log one call in the job folder (``logs/calls.jsonl``: the verb, its
    arguments as given, the envelope's outcome and message, the artifacts'
    paths, and ``reply``: the envelope itself as printed, kept in
    ``logs/calls/``, :func:`langslice.doors.call_log.save_reply`) and, with
    ``LANGSLICE_TRACE_DIR`` set, trace it as the MCP door traces a tool call
    (:mod:`langslice.doors.call_log`). Never raises; a folder without a job
    logs
    nothing."""
    import os

    from langslice.doors.call_log import (
        TRACE_DIR_ENV,
        cli_trace,
        log_call,
        save_reply,
    )
    from langslice.doors.jobs import NoJob, find
    from langslice.job.folder.layout import JobLayout

    try:
        layout = JobLayout(find(folder))
    except (NoJob, OSError, ValueError):
        return
    try:
        paths = [str(item.get("path")) for item in envelope.artifacts]
        kept = save_reply(layout.logs_dir, verb, dumps(envelope))
        log_call(
            layout.logs_dir,
            {
                "verb": verb,
                "arguments": list(rest),
                "ok": envelope.ok,
                "exit": envelope.exit,
                **(
                    {"error": envelope.error.get("code"), "message": envelope.error.get("message")}
                    if envelope.error
                    else {}
                ),
                **({"run": run_id} if run_id else {}),
                "warnings": len(envelope.warnings),
                "artifacts": paths,
                **({"reply": layout.relative(kept)} if kept is not None else {}),
                "seconds": round(seconds, 3),
            },
        )
        trace = cli_trace(layout.folder, os.environ.get(TRACE_DIR_ENV))
        if trace is not None:
            content: list[dict[str, Any]] = [{"text": dumps(envelope)}]
            for item in envelope.artifacts:
                if item.get("kind") in ("view", "opening"):
                    path = Path(str(item.get("path")))
                    content.append(
                        {
                            "image": "image/jpeg",
                            "path": str(path),
                            "bytes": path.stat().st_size if path.exists() else None,
                        }
                    )
            trace.write(
                "tool_result",
                name=verb,
                args=envelope.call or {"argv": list(rest)},
                content=content,
            )
    except Exception:
        logger.warning("Could not log the call %s %s", folder, verb, exc_info=True)


def _finish_run(folder: str, run_id: str, envelope: Envelope) -> None:
    from langslice.doors.jobs import find
    from langslice.job.folder.layout import JobLayout

    try:
        background.finish(JobLayout(find(folder)), run_id, envelope.to_dict(), envelope.exit)
    except Exception:
        logger.exception("Could not record background run %s", run_id)


def _load_args(value: str) -> dict[str, Any]:
    text = value
    if value.startswith("@"):
        try:
            text = Path(value[1:]).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise _Refusal(
                Envelope.failure(
                    RefusalCode.BAD_ARGUMENT,
                    f"--args {value}: {exc}. Pass --args a JSON object, or @path/to/file.json.",
                )
            ) from exc
    try:
        loaded = json.loads(text)
    except ValueError as exc:
        raise _Refusal(
            Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                f"--args is not JSON: {exc}. Pass --args a JSON object, or @path/to/file.json.",
            )
        ) from exc
    if not isinstance(loaded, dict):
        raise _Refusal(
            Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                "--args must be a JSON object: pass --args a JSON object, or @path/to/file.json.",
            )
        )
    return loaded


def parse(rest: list[str]) -> tuple[dict[str, Any], dict[str, list[str]], list[str]]:
    """``(options, flags, positional)``: the CLI's options, every other
    ``--name value`` as raw strings per argument name (snake case), and
    values given without a flag."""
    options: dict[str, Any] = {}
    flags: dict[str, list[str]] = {}
    positional: list[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        index += 1
        if not token.startswith("--") or token == "--":
            positional.append(token)
            continue
        key, _, inline = token[2:].partition("=")
        key = key.replace("-", "_")
        if key in ("dry_run", "background", "verbose"):
            options[key] = True
            continue
        if inline:
            value = inline
        elif index < len(rest) and not rest[index].startswith("--"):
            value = rest[index]
            index += 1
        else:
            value = ""
        if key == "args":
            options["args"] = _load_args(value)
        elif key == "timeout":
            try:
                options["timeout"] = float(value)
            except ValueError as exc:
                raise _Refusal(
                    Envelope.failure(RefusalCode.BAD_ARGUMENT, "--timeout takes seconds")
                ) from exc
        elif key == "run_id":
            options["run_id"] = value
        else:
            flags.setdefault(key, []).append(value)
    return options, flags, positional


def _flag_value(annotation: Any, values: list[str]) -> Any:
    """One argument's flag values as its declared type reads them."""
    origin = typing.get_origin(annotation)
    if annotation is bool:
        raw = values[-1].strip().lower()
        if raw in ("", "true", "1", "yes"):
            return True
        if raw in ("false", "0", "no"):
            return False
        return raw
    if annotation is str:
        return values[-1]
    if origin in (list, tuple):
        items: list[Any] = []
        numbers = typing.get_args(annotation)[:1] in ((float,), (int,))
        for value in values:
            try:
                loaded = json.loads(value)
            except ValueError:
                loaded = value
                if numbers and "," in value:
                    try:
                        loaded = [float(v) for v in value.strip("[]() ").split(",")]
                    except ValueError:
                        pass
            if isinstance(loaded, list):
                items.extend(loaded)
            else:
                items.append(value if typing.get_args(annotation)[:1] == (str,) else loaded)
        return items
    try:
        return json.loads(values[-1])
    except ValueError:
        return values[-1]


SHORT_SIGNATURE = 5


ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "cf.", "vs.")


def _first_sentence(text: str) -> str:
    """*text* up to the first full stop and space that ends a sentence: not
    one inside an abbreviation (:data:`ABBREVIATIONS`) or a bracket."""
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "." and text[index + 1 : index + 2] == " " and depth == 0:
            head = text[: index + 1]
            if not head.lower().endswith(ABBREVIATIONS):
                return head.strip()
    return text.strip()


def _flag_text(value: Any) -> str:
    """One value as a flag takes it (JSON for a list, an object or a number)."""
    import shlex

    return shlex.quote(value if isinstance(value, str) else json.dumps(value))


def corrected_call(declared: Declaration, verb: str, arguments: dict[str, Any], job: str) -> str:
    """The fix of an ``BAD_ARGUMENT`` refusal: every argument *verb*
    takes, with the first sentence of its description and its default, then
    the call made of the arguments it knows (*arguments*, the unknown ones
    left out), the required ones it lacks as placeholders and, for a short
    signature (:data:`SHORT_SIGNATURE`), the others at their defaults."""
    import inspect

    parameters = declared.signature.parameters
    takes = []
    for name, parameter in parameters.items():
        about = _first_sentence(declared.arguments.get(name, "")).rstrip(".")
        default = (
            "required"
            if parameter.default is inspect.Parameter.empty
            else f"default {json.dumps(parameter.default)}"
        )
        takes.append(f"{name} ({about + '; ' if about else ''}{default})")
    parts = [f"langslice-job {job} {verb}"]
    short = len(parameters) <= SHORT_SIGNATURE
    for name, parameter in parameters.items():
        flag = "--" + name.replace("_", "-")
        if name in arguments:
            parts.append(f"{flag} {_flag_text(arguments[name])}")
        elif parameter.default is inspect.Parameter.empty:
            parts.append(f"{flag} <{name}>")
        elif short:
            parts.append(f"{flag} {_flag_text(parameter.default)}")
    return f"{verb} takes {'; '.join(takes) or 'no arguments'}. Call it as: `{' '.join(parts)}`."


def arguments_for(
    tool: Any,
    declaration: Declaration,
    options: dict[str, Any],
    flags: dict[str, list[str]],
    verb: str,
    job: str = "FOLDER",
) -> dict[str, Any]:
    """The call's arguments: ``--args``, then each flag over it, read as the
    tool declares them; refused (exit 2) when one is unknown (the fix: what
    the verb takes and the call without the unknown ones,
    :func:`corrected_call`) or missing."""
    import inspect

    from langslice.tools.arguments import argument_refusal, normalize_arguments

    signature = inspect.signature(tool)
    declared = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "tool_context"
    }
    arguments = dict(options.get("args") or {})
    for key, values in flags.items():
        parameter = declared.get(key)
        arguments[key] = (
            _flag_value(parameter.annotation, values) if parameter is not None else values[-1]
        )
    refusal = argument_refusal(tool, arguments)
    if refusal is not None:
        problems = cast("list[dict[str, object]]", refusal.facts.get("problems") or [])
        top_level = all(problem.get("argument") == "(top level)" for problem in problems)
        known = {name: value for name, value in arguments.items() if name in declared}
        fix = None
        if top_level:
            fix = corrected_call(declaration, verb, known, job)
        raise _Refusal(
            Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                f"{refusal.message} Check the argument names with `langslice-job schema {verb}`.",
                result=refusal.payload(),
                fix=fix,
            )
        )
    missing = [
        name
        for name, parameter in declared.items()
        if parameter.default is inspect.Parameter.empty and name not in arguments
    ]
    if missing:
        raise _Refusal(
            Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                f"{verb} needs {', '.join(missing)}. Give every required argument; see "
                f"`langslice-job schema {verb}`.",
                result={"missing": missing},
            )
        )
    return normalize_arguments(tool, arguments)


def _spec_of(folder: str) -> Any:
    """The settings of the job at *folder* (its tool plugins loaded), or None
    when it has no readable ones (the call then fails on its own)."""
    from langslice.doors.jobs import find, read_spec

    try:
        return read_spec(find(folder))
    except (LookupError, OSError, TypeError, ValueError):
        return None


def _open(folder: str, *, persist: bool, atlas_loader: Any) -> Opened:
    from langslice.doors.jobs import NoJob, open_folder

    try:
        return open_folder(
            folder, atlas_loader=atlas_loader, emit=progress, persist=persist, door="cli"
        )
    except NoJob as exc:
        raise _Refusal(Envelope.failure(RefusalCode.NOT_FOUND, str(exc))) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise _Refusal(Envelope.failure(RefusalCode.IO_ERROR, str(exc), exit=EXIT_REFUSED)) from exc


def call(
    folder: str,
    verb: str,
    flags: dict[str, list[str]],
    options: dict[str, Any],
    *,
    atlas_loader: Any = None,
) -> Envelope:
    """Run *verb* on the job *folder* names (see the module text)."""
    from langslice.tools.catalog import verbs_for

    if verb not in verbs_for(_spec_of(folder)):
        raise _Refusal(
            Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                f"No verb {verb!r}. List the verbs with `langslice-job ops`.",
            )
        )
    if not on_command_line(verb):
        raise _not_a_command(verb)
    dry_run = bool(options.get("dry_run"))
    if dry_run and options.get("background"):
        raise _Refusal(
            Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                "--dry-run and --background do not go together: a dry run answers at once.",
            )
        )
    opened = _open(folder, persist=not dry_run, atlas_loader=atlas_loader)
    job_folder = str(opened.job.folder)
    try:
        box = opened.tools()
        tools = {tool.__name__: tool for tool in box.tools}
        offered = opened.listed_verbs()
        if verb not in tools:
            spec = opened.spec
            tasks = spec.tasks
            raise _Refusal(
                Envelope.failure(
                    RefusalCode.TOOL_OFF,
                    f"This job's settings (tasks {tasks}"
                    + (f", disabled {spec.disabled}" if spec.disabled else "")
                    + f") have no {verb}. `langslice-job {job_folder} status` lists the job's "
                    "verbs under `verbs`.",
                    result={"verbs": offered},
                )
            )
        tool = tools[verb]
        declared = declaration(verb, box.variant)
        arguments = arguments_for(tool, declared, options, flags, verb, job_folder)
        long = verbs_for(opened.spec)[verb].long
        if options.get("run_id"):
            background.begin(opened.job.layout, options["run_id"])
        if options.get("background") and not options.get("run_id"):
            run_id = background.start(
                opened.job.layout, verb, arguments, verbose=bool(options.get("verbose"))
            )
            envelope = Envelope(
                result={"run": run_id, "verb": verb, "state": "running"},
                next=[f"langslice-job {job_folder} wait {run_id}"],
            )
        elif long and not dry_run and not options.get("run_id") and background.detaching():
            envelope = _detached(
                opened.job.layout, job_folder, verb, arguments, verbose=bool(options.get("verbose"))
            )
        else:
            envelope = _run(
                opened, verb, tool, arguments, dry_run=dry_run, verbose=bool(options.get("verbose"))
            )
        if dry_run and envelope.ok and long and not envelope.next:
            envelope.next = [_command(job_folder, verb, arguments)]
        envelope.call = arguments
        return envelope
    finally:
        opened.close()


def _not_a_command(verb: str) -> _Refusal:
    """Return the refusal of a registration tool called as a command.

    Parameters
    ----------
    verb
        The tool's name.

    Returns
    -------
    _Refusal
        ``TOOL_OFF``, its message saying where the tool is.
    """
    runs = ", ".join(sorted(CLI_VERBS))
    return _Refusal(
        Envelope.failure(
            RefusalCode.TOOL_OFF,
            f"{verb} is a registration tool, not a command: the command "
            f"line sets a job up and checks it ({runs}, init, launch, runs). A registration "
            f"agent calls {verb} as its MCP tool; a script calls "
            f"`langslice.open_job(FOLDER).{verb}(...)`.",
        )
    )


def _detached(
    layout: Any, job_folder: str, verb: str, arguments: dict[str, Any], *, verbose: bool
) -> Envelope:
    """A long verb's call where a process may outlive its command
    (:func:`background.detaching`): the verb runs as a background run and the
    call waits for it, at most :data:`background.WAIT_S`, then answers as
    ``wait`` does. A command ended meanwhile ends only the waiting (the run
    goes on); Ctrl-C ends the run too."""
    run_id = background.start(layout, verb, arguments, verbose=verbose)
    progress(
        f"langslice-job: {verb} runs as run {run_id}; if this command is ended, the "
        f"run goes on: `langslice-job {job_folder} wait {run_id}`"
    )
    try:
        record = background.wait(layout, run_id, background.WAIT_S)
    except KeyboardInterrupt:
        background.cancel(layout, run_id)
        raise
    if record is None:
        return _no_run(job_folder, run_id)
    return run_answer(job_folder, run_id, record, background.WAIT_S)


def _no_run(job_folder: str, run_id: str) -> Envelope:
    """The refusal for a run id the job does not hold."""
    return Envelope.failure(
        RefusalCode.NOT_FOUND,
        f"No run {run_id}. List the background runs with `langslice-job {job_folder} runs`.",
    )


def _command(job_folder: str, verb: str, arguments: dict[str, Any]) -> str:
    """The command line that calls *verb* with *arguments* on the job."""
    return f"langslice-job {job_folder} {verb} --args '{json.dumps(arguments, default=str)}'"


def _run(
    opened: Any, verb: str, tool: Any, arguments: dict[str, Any], *, dry_run: bool, verbose: bool
) -> Envelope:
    from langslice.job.pictures import artifacts as picture_artifacts
    from langslice.job.pictures import captured
    from langslice.tools.catalog import verbs_for
    from langslice.tools.pipeline import work_pictures

    verbs = verbs_for(opened.spec)
    job = opened.job
    with captured() as saved:
        reply = tool(**arguments)
        work = cast("Job", job).background
        work.wait_all()
    if isinstance(reply, dict):
        from langslice.tools.pipeline import with_notices

        reply = with_notices(job, reply)
    artifacts: list[dict[str, Any]] = []
    warnings: list[str] = []
    ok = not (isinstance(reply, dict) and reply.get("status") in FAILED_STATUSES)
    if verb == "export_maps" and isinstance(reply, dict) and not dry_run:
        artifacts += [dict(item) for item in reply.get("files") or []]
    job.views.flush()
    pictures, unsaved = picture_artifacts(saved, verb)
    artifacts += pictures
    warnings += unsaved
    artifacts += work_pictures(cast("Job", job), cast(object, reply), start=len(pictures))
    result = shape(verb, reply, verbose=verbose)
    if verb == "status" and isinstance(result, dict):
        result["verbs"] = opened.listed_verbs()
        state = image_model_state(opened.spec, connected=opened.image_model_connected)
        if state is not None:
            result["image_model"] = state
    if dry_run and isinstance(result, dict):
        result["dry_run"] = True
        if verbs[verb].scripting:
            warnings.append(
                f"{verb} wrote nothing under --dry-run; `files` lists what it would write."
            )
        else:
            warnings.append(f"{verb} only reads; --dry-run changes nothing about it.")
    if isinstance(reply, dict):
        for key in WARN_KEYS:
            value = reply.get(key)
            if value:
                warnings.append(f"{key}: {json.dumps(value, default=str)}")
        if ok and isinstance(reply.get("results"), list):
            warnings += [
                f"{row.get('id')}: {row.get('error')}"
                for row in reply["results"]
                if isinstance(row, dict) and row.get("status") == ReplyStatus.ERROR
            ]
    if not ok:
        unnamed = RefusalCode.INTERNAL
        told = cast("dict[str, object]", reply)
        code = str(told.get("error") or unnamed)
        message = str(told.get("message") or "")
        rows = told.get("results")
        if not message and isinstance(rows, list):
            message = "; ".join(
                f"{row.get('id')}: {row.get('error')}"
                for row in rows
                if isinstance(row, dict) and row.get("error")
            )
        failed = Envelope.failure(code, message, result=result)
        failed.artifacts, failed.warnings = artifacts, warnings
        return failed
    return Envelope(result=result, artifacts=artifacts, warnings=warnings)


def shape(verb: str, reply: Any, *, verbose: bool) -> Any:
    """A tool reply for the CLI: pictures out (they are artifacts); status
    rows uniform (every field, null where absent:
    :func:`langslice.core.stack.status.with_uniform_rows`); concise unless
    *verbose* (no whole-stack rows on a write; a reply with pictures keeps
    what they are as one line, ``picture_note``; *verbose* keeps the
    description and the pictures' text lines as written for a model)."""
    from langslice.core.stack.status import with_uniform_rows
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    if not isinstance(reply, dict):
        return reply
    body = with_uniform_rows(dict(reply))
    media = body.pop(TOOL_MEDIA_PARTS_KEY, None)
    if verbose:
        texts = [item for item in media if isinstance(item, str)] if isinstance(media, list) else []
        if texts:
            body["media_texts"] = texts
        return body
    description = body.pop("description", None)
    pictured = isinstance(media, list) and any(not isinstance(item, str) for item in media)
    if pictured and isinstance(description, str) and description.strip():
        body["picture_note"] = " ".join(description.split())
    if verb not in ROW_VERBS and isinstance(body.get("rows"), list):
        body["n_rows"] = len(body.pop("rows"))
    if body.get("files_written") and isinstance(body.get("files"), list):
        body["n_files"] = len(body.pop("files"))
    return body


def init_parser() -> argparse.ArgumentParser:
    """``init``'s flags: the job flags of ``langslice linear run``, plus
    ``--notes`` and ``--viewer`` (``langslice-job schema init`` lists them)."""
    from langslice.core.pictures.opening import VIEWER_LIMITS
    from langslice.doors.commands.flags import add_flags

    parser = _Parser(prog="langslice-job FOLDER init", add_help=False)
    for group in INIT_FLAG_GROUPS:
        add_flags(parser, group, choices={"--viewer": sorted(VIEWER_LIMITS)})
    return parser


def init_flags() -> list[dict[str, Any]]:
    """``init``'s flags as data (:func:`langslice.doors.commands.flags.flag_rows`),
    the ones :func:`init_parser` adds."""
    from langslice.core.pictures.opening import VIEWER_LIMITS
    from langslice.doors.commands.flags import flag_rows

    return flag_rows(INIT_FLAG_GROUPS, choices={"--viewer": sorted(VIEWER_LIMITS)})


def init(folder: str, rest: list[str], *, atlas_loader: Any = None) -> Envelope:
    """Create (or continue) the job for the image folder *folder*: the job
    flags of ``langslice linear run``; ingest as every host does. With
    ``--registration FILE`` the file's linear registration is the job's
    supplied placement (``doors.jobs.with_registration``): the result's
    ``registration`` says what was placed and what was not, and its
    warnings are the envelope's (``BAD_ARGUMENT`` when it cannot be
    read, matched one to one, or places no section)."""
    from langslice.core.sections.discovery import discover_slices
    from langslice.doors.commands.linear import spec_from_args
    from langslice.doors.jobs import (
        NoJob,
        create,
        find,
        read_spec,
        with_registration,
    )
    from langslice.job.host_inputs import InputsChanged

    args = init_parser().parse_args(rest)
    images = Path(folder).expanduser().resolve()
    if not images.is_dir() or not discover_slices(str(images)):
        return Envelope.failure(
            RefusalCode.NOT_FOUND,
            f"No section images in {images}. Point init at a folder of section images "
            "(TIFF, PNG or JPEG).",
        )
    try:
        spec = spec_from_args(args, str(images))
    except ValueError as exc:
        return Envelope.failure(RefusalCode.BAD_ARGUMENT, str(exc))
    imported: dict[str, Any] | None = None
    if args.registration:
        try:
            spec, imported = with_registration(
                spec, args.registration, atlas_loader=atlas_loader, emit=progress
            )
        except (OSError, ValueError) as exc:
            return Envelope.failure(
                RefusalCode.BAD_ARGUMENT,
                f"{exc} Give --registration a QuickNII/VisuAlign JSON or XML, a DeepSlice "
                "CSV/JSON/XML or a LangSlice registration.json whose entries name this "
                "folder's section images (by file name, stem or _sNNN number) on the job's "
                "atlas.",
            )
    if spec.resume:
        try:
            existing = find(spec.job_dir or images)
            saved = read_spec(existing).to_dict()
        except NoJob:
            existing, saved = None, None
        if existing is not None and saved is not None:
            asked = spec.to_dict()
            changed = sorted(
                key
                for key in set(asked) | set(saved)
                if key not in ("resume", "inputs", "image_folder")
                and asked.get(key) != saved.get(key)
            )
            if changed:
                return Envelope.failure(
                    RefusalCode.STALE_INPUT,
                    f"The job in this folder was made with other settings: {', '.join(changed)}. "
                    "Give init the job flags it was made with (job.json, spec) to continue "
                    "it, or --no-resume to start it over (its work is discarded).",
                )
    try:
        opened = create(spec, atlas_loader=atlas_loader, emit=progress, door="cli")
    except InputsChanged as exc:
        return Envelope.failure(
            RefusalCode.STALE_INPUT,
            f"{exc} Rerun `langslice-job {images} init` with --no-resume to start a new job "
            "from these inputs (the old one's work is replaced), or with the inputs the job "
            "was made from (job.json, spec.inputs) to continue it.",
        )
    try:
        from langslice.doors.agent_cli import launch as launches
        from langslice.doors.jobs import VIEWER_KEY, job_viewer
        from langslice.job.folder.layout import write_job_file

        layout = opened.job.layout
        state = opened.job.state
        settings = {
            **({NOTES_KEY: args.notes} if args.notes is not None else {}),
            **({VIEWER_KEY: args.viewer} if args.viewer else {}),
        }
        if settings:
            write_job_file(layout, **settings)
            opened.viewer = job_viewer(layout)
        listed_verbs = opened.listed_verbs()
        result = {
            "job_folder": str(layout.folder),
            "image_folder": str(images),
            "sections": [record.id for record in state.in_order()],
            "tasks": list(opened.spec.tasks),
            "atlas": opened.spec.atlas,
            "plane": opened.spec.plane,
            "verbs": listed_verbs,
            "resumed": bool(opened.job.undo.steps or opened.job.undo.redo_steps),
            **launches.facts(opened),
        }
        if imported is not None:
            result["registration"] = {
                key: value for key, value in imported.items() if key != "warnings"
            }
        artifacts = [
            {"path": str(layout.folder / name), "kind": "card"}
            for name in ("AGENTS.md", "CLAUDE.md")
        ]
        artifacts.append({"path": str(layout.state_file), "kind": "state"})
        tools = launches.write_tools(layout.folder, opened.tools())
        if tools is not None:
            artifacts.append({"path": str(tools), "kind": "tools"})
        return Envelope(
            result=result,
            artifacts=artifacts,
            warnings=list((imported or {}).get("warnings") or []),
            next=[
                f"langslice-job {layout.folder} launch --role register",
                f"langslice-job {layout.folder} launch --role register-scripting",
            ],
        )
    finally:
        opened.close()


def launch(folder: str, rest: list[str], *, atlas_loader: Any = None) -> Envelope:
    """``launch --role register|register-scripting``: the exact prompt that
    launches a registration subagent, saved in ``logs/launches/``
    (:mod:`langslice.doors.agent_cli.launch`)."""
    from langslice.doors.agent_cli import launch as launches
    from langslice.doors.jobs import NoJob, find
    from langslice.job.folder.layout import JobLayout

    parser = _Parser(prog="launch", add_help=False)
    parser.add_argument("--role", required=True, choices=launches.ROLES)
    role = parser.parse_args(rest).role
    try:
        layout = JobLayout(find(folder))
    except NoJob as exc:
        return Envelope.failure(RefusalCode.NOT_FOUND, str(exc))
    try:
        if role == "register":
            prompt = launches.mcp_prompt(layout.folder, atlas_loader)
        else:
            prompt = launches.scripting_prompt(
                layout.folder,
                atlas_loader=cast("Callable[[str], object] | None", atlas_loader),
            )
    except ImportError as exc:
        return Envelope.failure(
            RefusalCode.ENGINE_UNAVAILABLE,
            f"The {role} role needs the `mcp` package ({exc}); install it in LangSlice's "
            "environment (pip install mcp).",
        )
    except (FileNotFoundError, ValueError) as exc:
        return Envelope.failure(RefusalCode.IO_ERROR, str(exc), exit=EXIT_REFUSED)
    except (OSError, subprocess.SubprocessError) as exc:
        return Envelope.failure(RefusalCode.IO_ERROR, str(exc))
    made = launches.Launch(role=role, prompt=prompt)
    saved = launches.save(layout, role, made.prompt)
    return Envelope(
        result={
            "role": role,
            "job_folder": str(layout.folder),
            "prompt": made.prompt,
        },
        artifacts=[{"path": str(saved), "kind": "launch"}],
    )


def runs(folder: str, verb: str, positional: list[str], options: dict[str, Any]) -> Envelope:
    """``runs [ID]`` (every run, or one) and ``wait [ID]`` (the latest without ID)."""
    from langslice.doors.jobs import NoJob, find
    from langslice.job.folder.layout import JobLayout

    try:
        layout = JobLayout(find(folder))
    except NoJob as exc:
        return Envelope.failure(RefusalCode.NOT_FOUND, str(exc))
    job_folder = str(layout.folder)
    if len(positional) > 1:
        return Envelope.failure(RefusalCode.BAD_ARGUMENT, "Give one run id.")
    if verb == "runs" and not positional:
        listed = background.listing(layout)
        return Envelope(
            result={"runs": listed},
            next=[
                f"langslice-job {job_folder} wait {run['id']}"
                for run in listed
                if run.get("state") == "running"
            ][:1],
        )
    run_id = positional[0] if positional else background.latest(layout)
    if run_id is None:
        return Envelope.failure(
            RefusalCode.NOT_FOUND,
            "This job has no background runs. Start one with a long verb and --background.",
        )
    if verb == "runs":
        record = background.read(layout, run_id)
        if record is None:
            return _no_run(job_folder, run_id)
        return Envelope(
            result={
                **background.brief(record),
                **({"envelope": record["envelope"]} if "envelope" in record else {}),
            },
            next=[]
            if record.get("state") != "running"
            else [f"langslice-job {job_folder} wait {run_id}"],
        )
    waited = float(options.get("timeout", background.WAIT_S))
    record = background.wait(layout, run_id, waited)
    if record is None:
        return _no_run(job_folder, run_id)
    return run_answer(job_folder, run_id, record, waited)


def run_answer(job_folder: str, run_id: str, record: dict[str, Any], waited: float) -> Envelope:
    """What ``wait`` (or a long verb's detached call) answers after waiting
    *waited* seconds for run *run_id*: its answer once it has one; while it
    runs, ``ok`` with its state ``running`` and the command that waits again;
    ``INTERNAL`` when its process ended without one."""
    state = record.get("state")
    if state == "running":
        return Envelope(
            result={
                **background.brief(record),
                "waited_s": waited,
                "message": f"Still running after {waited:g} s; the run goes on. Wait "
                "again for its answer.",
            },
            next=[f"langslice-job {job_folder} wait {run_id}"],
        )
    if state != "finished" or not isinstance(record.get("envelope"), dict):
        return Envelope.failure(
            RefusalCode.INTERNAL,
            f"Run {run_id} ended without an answer; see its log under logs/runs/ and run the "
            "verb again. If every background run is lost, your shell ends the processes a "
            "command leaves running when it returns (sandboxes do this): run the verb as "
            "`LANGSLICE_DETACH=0 langslice-job ...`, without --background, in your harness's "
            "own background mode if it has one.",
            result=background.brief(record),
            exit=EXIT_INTERNAL,
        )
    answer = Envelope.from_dict(record["envelope"], int(record.get("exit", EXIT_OK)))
    if isinstance(answer.result, dict):
        answer.result = {**answer.result, "run": run_id}
    return answer


__all__ = ["EXIT_ARGUMENTS", "add_arguments", "execute", "run"]

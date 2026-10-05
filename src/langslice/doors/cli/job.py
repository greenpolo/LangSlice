"""``langslice job FOLDER VERB [arguments]``: one verb on a job folder, for agents.

The agent CLI: every verb of :data:`langslice.ops.registry.VERBS`
under the tool's own name (kebab-case accepted: ``set-positions``), plus
``init`` (create the job for a folder of images), ``brief`` (the job
statement and the opening pictures, :mod:`langslice.doors.cli.brief`),
``runs [ID]`` and ``wait [ID]`` (background runs; ``status`` is only the
verb). FOLDER is the job folder or the image folder beside it.

Arguments are the verb's declared arguments
(:mod:`langslice.doors.declarations`, ``langslice schema VERB``): a JSON
object with ``--args '{...}'`` or ``--args @file.json``, and/or one flag per
argument, ``--name value`` (kebab or snake case; a JSON value, or plain text
for a text argument; a list argument takes the flag again for each item).
Options: ``--dry-run`` (a write verb runs on the job without writing
anything and reports what would change; ``fit_deformable`` and
``trace_borders`` are only checked, not run; not with ``--background``),
``--background`` (answer at
once with a run id), ``--verbose`` (the whole reply: whole-stack rows, the
descriptions written for a model), ``--timeout SECONDS`` (``wait``).

Each call opens the job as it stands on disk (``doors.jobs.open_folder``),
runs the verb through the same tool a running agent has (the
look-before-commit gates off: they are tool-only; ``view.resolution`` any
size from 128 px to the source's own pixels) and closes it, so calls and a
running agent interleave on one folder: each picks up the other's writes
(``Job.sync``) and every write is one undo step. The answer is the JSON
envelope of :mod:`langslice.doors.cli.envelope` on stdout; anything a
library prints goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import typing
from pathlib import Path
from typing import Any

from langslice.doors.cli import background
from langslice.doors.cli.catalog import canonical_verb
from langslice.doors.cli.envelope import (
    EXIT_ARGUMENTS,
    EXIT_INTERNAL,
    EXIT_OK,
    EXIT_REFUSED,
    Envelope,
    dumps,
    emit,
    stdout_to_stderr,
)
from langslice.job.lock import JobBusy

logger = logging.getLogger(__name__)

#: Verbs a dry run checks without running (a model call, a long fit that
#: saves records).
CHECKED_ONLY = frozenset({"trace_borders", "trace_from_atlas", "fit_deformable"})
#: Reply keys worth a warning when present and not empty.
WARN_KEYS = ("unknown_ids", "unknown", "rejected", "render_failed", "clamped",
             "deformation_cleared", "truncated", "dropped_positions_mm", "not_shown")
#: Verbs whose whole-stack ``rows`` are their answer (kept when concise).
ROW_VERBS = frozenset({"status", "view_stack"})


def progress(message: str) -> None:
    """The job's progress lines: stderr, never stdout."""
    print(message, file=sys.stderr, flush=True)


class _Refusal(Exception):
    def __init__(self, envelope: Envelope) -> None:
        super().__init__(envelope.error)
        self.envelope = envelope


class _Parser(argparse.ArgumentParser):
    """argparse that raises instead of exiting (the envelope reports it)."""

    def error(self, message: str) -> typing.NoReturn:
        raise _Refusal(Envelope.failure("BAD_ARGUMENTS", message))


# --- the command ------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "job", help="Agent CLI: one verb on a job folder, JSON on stdout "
        "(see `langslice ops`)")
    p.add_argument("folder", help="The job folder, or the image folder beside it")
    p.add_argument("verb", help="A verb (`langslice ops`), init, runs [ID] or wait [ID]")
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="--args JSON|@file, --name value, --dry-run, --background, "
                   "--verbose, --timeout SECONDS")


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    with stdout_to_stderr():
        envelope = execute(args.folder, args.verb, list(args.rest))
    return emit(envelope)


def execute(folder: str, verb: str, rest: list[str], *,
            atlas_loader: Any = None) -> Envelope:
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
        envelope = Envelope.failure("JOB_BUSY", str(exc), exit=EXIT_REFUSED)
    except Exception as exc:  # the envelope reports it; the traceback goes to stderr
        logger.exception("langslice job %s %s failed", folder, verb)
        envelope = Envelope.failure("INTERNAL", f"{type(exc).__name__}: {exc}",
                                    exit=EXIT_INTERNAL)
    if run_id:
        _finish_run(folder, run_id, envelope)
    record(folder, name, rest, envelope, seconds=time.perf_counter() - started, run_id=run_id)
    return envelope


def _run_id(rest: list[str]) -> str | None:
    """A background child's ``--run-id`` (:mod:`langslice.doors.cli.background`)."""
    for index, token in enumerate(rest):
        if token.startswith("--run-id="):
            return token.partition("=")[2] or None
        if token in ("--run-id", "--run_id") and index + 1 < len(rest):
            return rest[index + 1]
    return None


def _execute(folder: str, name: str, rest: list[str], *, atlas_loader: Any) -> Envelope:
    if name == "init":
        return init(folder, rest, atlas_loader=atlas_loader)
    options, arguments, positional = parse(rest)
    if name == "brief":
        if arguments or positional:
            raise _Refusal(Envelope.failure(
                "BAD_ARGUMENTS", "brief takes no arguments.", verb=name))
        return brief(folder, atlas_loader=atlas_loader)
    if name in ("runs", "wait"):
        return runs(folder, name, positional, options)
    if positional:
        raise _Refusal(Envelope.failure(
            "BAD_ARGUMENTS", f"Unexpected value(s) {positional}: give arguments as "
            "--name value or --args JSON.", verb=name))
    return call(folder, name, arguments, options, atlas_loader=atlas_loader)


def record(folder: str, verb: str, rest: list[str], envelope: Envelope, *,
           seconds: float, run_id: str | None = None) -> None:
    """Log one call in the job folder (``logs/calls.jsonl``: the verb, its
    arguments as given, the envelope's outcome, the artifacts' paths) and,
    with ``LANGSLICE_TRACE_DIR`` set, trace it as the MCP door traces a tool
    call (:mod:`langslice.doors.trace`). Never raises; a folder without a
    job, or a lean job (which keeps no logs), logs nothing."""
    import os

    from langslice.doors.jobs import NoJob, find
    from langslice.doors.trace import TRACE_DIR_ENV, cli_trace, log_call
    from langslice.job.layout import JobLayout, read_job_file

    try:
        layout = JobLayout(find(folder))
        held = read_job_file(layout) or {}
    except (NoJob, OSError, ValueError):
        return
    try:
        if (held.get("spec") or {}).get("output_level") == "lean":
            return
        paths = [str(item.get("path")) for item in envelope.artifacts]
        log_call(layout.logs_dir, {
            "verb": verb, "arguments": list(rest), "ok": envelope.ok, "exit": envelope.exit,
            **({"error": envelope.error.get("code")} if envelope.error else {}),
            **({"run": run_id} if run_id else {}),
            "warnings": len(envelope.warnings), "artifacts": paths,
            "seconds": round(seconds, 3),
        })
        trace = cli_trace(layout.folder, os.environ.get(TRACE_DIR_ENV))
        if trace is not None:
            content: list[dict[str, Any]] = [{"text": dumps(envelope)}]
            for item in envelope.artifacts:
                if item.get("kind") in ("view", "opening"):
                    path = Path(str(item.get("path")))
                    content.append({"image": "image/jpeg", "path": str(path),
                                    "bytes": path.stat().st_size if path.exists() else None})
            trace.write("tool_result", name=verb, args=envelope.call or {"argv": list(rest)},
                        content=content)
    except Exception:
        logger.warning("Could not log the call %s %s", folder, verb, exc_info=True)


def _finish_run(folder: str, run_id: str, envelope: Envelope) -> None:
    from langslice.doors.jobs import find
    from langslice.job.layout import JobLayout

    try:
        background.finish(JobLayout(find(folder)), run_id, envelope.to_dict(), envelope.exit)
    except Exception:
        logger.exception("Could not record background run %s", run_id)


# --- arguments --------------------------------------------------------------------


def _load_args(value: str) -> dict[str, Any]:
    text = value
    if value.startswith("@"):
        try:
            text = Path(value[1:]).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise _Refusal(Envelope.failure("BAD_JSON", f"--args {value}: {exc}")) from exc
    try:
        loaded = json.loads(text)
    except ValueError as exc:
        raise _Refusal(Envelope.failure("BAD_JSON", f"--args is not JSON: {exc}")) from exc
    if not isinstance(loaded, dict):
        raise _Refusal(Envelope.failure("BAD_JSON", "--args must be a JSON object"))
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
            value = ""  # a bare flag: true for a yes/no argument
        if key == "args":
            options["args"] = _load_args(value)
        elif key == "timeout":
            try:
                options["timeout"] = float(value)
            except ValueError as exc:
                raise _Refusal(Envelope.failure(
                    "BAD_ARGUMENTS", "--timeout takes seconds")) from exc
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
        for value in values:
            try:
                loaded = json.loads(value)
            except ValueError:
                loaded = value
            if isinstance(loaded, list):
                items.extend(loaded)
            else:
                items.append(value if typing.get_args(annotation)[:1] == (str,)
                             else loaded)
        return items
    try:
        return json.loads(values[-1])
    except ValueError:
        return values[-1]


def arguments_for(tool: Any, options: dict[str, Any],
                  flags: dict[str, list[str]], verb: str) -> dict[str, Any]:
    """The call's arguments: ``--args``, then each flag over it, read as the
    tool declares them; refused (exit 2) when one is unknown or missing."""
    import inspect

    from langslice.doors.tools.arguments import argument_refusal, normalize_arguments

    signature = inspect.signature(tool)
    declared = {name: parameter for name, parameter in signature.parameters.items()
                if name != "tool_context"}
    arguments = dict(options.get("args") or {})
    for key, values in flags.items():
        parameter = declared.get(key)
        arguments[key] = (_flag_value(parameter.annotation, values) if parameter is not None
                          else values[-1])
    refusal = argument_refusal(tool, arguments)
    if refusal is not None:
        raise _Refusal(Envelope.failure("UNKNOWN_ARGUMENTS", refusal["message"],
                                        result=refusal, verb=verb))
    missing = [name for name, parameter in declared.items()
               if parameter.default is inspect.Parameter.empty and name not in arguments]
    if missing:
        raise _Refusal(Envelope.failure(
            "MISSING_ARGUMENTS", f"{verb} needs {', '.join(missing)}.",
            result={"missing": missing}, verb=verb))
    return normalize_arguments(tool, arguments)


# --- one verb ---------------------------------------------------------------------


def _open(folder: str, *, persist: bool, atlas_loader: Any) -> Any:
    from langslice.doors.jobs import NoJob, open_folder

    try:
        return open_folder(folder, atlas_loader=atlas_loader, emit=progress, persist=persist,
                           door="cli")
    except NoJob as exc:
        raise _Refusal(Envelope.failure("NO_JOB", str(exc))) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise _Refusal(Envelope.failure("JOB_UNREADABLE", str(exc), exit=EXIT_REFUSED)) \
            from exc


def call(folder: str, verb: str, flags: dict[str, list[str]], options: dict[str, Any], *,
         atlas_loader: Any = None) -> Envelope:
    """Run *verb* on the job *folder* names (see the module text)."""
    from langslice.ops.registry import VERBS

    if verb not in VERBS:
        raise _Refusal(Envelope.failure("UNKNOWN_VERB", f"No verb {verb!r}.", verb=verb))
    dry_run = bool(options.get("dry_run"))
    if dry_run and options.get("background"):
        # A background child runs the verb for real; a dry run answers at once.
        raise _Refusal(Envelope.failure(
            "BAD_ARGUMENTS", "--dry-run and --background do not go together: a dry run "
            "answers at once.", verb=verb))
    opened = _open(folder, persist=not dry_run, atlas_loader=atlas_loader)
    job_folder = str(opened.job.folder)
    try:
        # The job's verbs as the tool door builds them (a hidden verb is
        # called by name, never listed: ``Verb.hidden``).
        tools = {tool.__name__: tool for tool in opened.tools().tools}
        offered = opened.listed_verbs()
        if verb not in tools:
            spec = opened.spec
            if (VERBS[verb].image_model and spec.has("nonlinear")
                    and spec.nonlinear.uses_image_model and not opened.image_model_connected):
                raise _Refusal(Envelope.failure(
                    "IMAGE_MODEL_OFF", f"{verb} needs the job's image model "
                    f"({spec.nonlinear.provider}), which is not connected to LangSlice here "
                    "(no key or login).", result={"verbs": offered}, job=job_folder, verb=verb))
            raise _Refusal(Envelope.failure(
                "VERB_OFF", f"This job's settings (tasks {spec.tasks}) have no {verb}.",
                result={"verbs": offered}, job=job_folder, verb=verb))
        tool = tools[verb]
        arguments = arguments_for(tool, options, flags, verb)
        if options.get("run_id"):
            background.begin(opened.job.layout, options["run_id"])
        if options.get("background") and not options.get("run_id"):
            run_id = background.start(opened.job.layout, verb, arguments,
                                      verbose=bool(options.get("verbose")))
            envelope = Envelope(result={"run": run_id, "verb": verb, "state": "running"},
                                next=[f"langslice job {job_folder} wait {run_id}"])
        elif dry_run and verb in CHECKED_ONLY:
            envelope = _checked(opened, verb, arguments)
        else:
            envelope = _run(opened, verb, tool, arguments, dry_run=dry_run,
                            verbose=bool(options.get("verbose")))
        if dry_run and envelope.ok and VERBS[verb].long:
            # A long verb: the real call may take minutes; --background
            # answers at once and `wait` collects the answer.
            envelope.next = [*envelope.next, _command(job_folder, verb, arguments)
                             + " --background"]
        envelope.call = arguments
        return envelope
    finally:
        opened.close()


def _command(job_folder: str, verb: str, arguments: dict[str, Any]) -> str:
    """The command line that calls *verb* with *arguments* on the job."""
    return (f"langslice job {job_folder} {verb} --args "
            f"'{json.dumps(arguments, default=str)}'")


def _checked(opened: Any, verb: str, arguments: dict[str, Any]) -> Envelope:
    """A dry run of a verb that is not simulated: its sections resolved."""
    refs = list(arguments.get("slices") or []) + (
        [arguments["id"]] if arguments.get("id") not in (None, "") else [])
    known = [opened.job.state.resolve(ref) for ref in refs]
    unknown = [str(ref) for ref, record in zip(refs, known, strict=True) if record is None]
    if unknown:
        return Envelope.failure("UNKNOWN_SLICE_IDS", "No such section(s).",
                                result={"unknown": unknown}, job=str(opened.job.folder))
    return Envelope(
        result={"dry_run": True, "simulated": False,
                "sections": [record.id for record in known if record is not None]},
        warnings=[f"{verb} is checked, not run, by --dry-run: the arguments and sections "
                  "are valid; nothing ran and nothing was written."])


def _run(opened: Any, verb: str, tool: Any, arguments: dict[str, Any], *,
         dry_run: bool, verbose: bool) -> Envelope:
    from langslice.job.views import captured
    from langslice.ops.registry import VERBS

    job = opened.job
    job_folder = str(job.folder)
    before = job.snapshot() if dry_run else None
    with captured() as saved:
        reply = tool(**arguments)
    if VERBS[verb].image_model:
        job.settle_image_corrections()  # this process ends: the calls land now
    artifacts: list[dict[str, Any]] = []
    warnings: list[str] = []
    ok = not (isinstance(reply, dict) and reply.get("status") in ("error", "refused"))
    if ok and verb == "submit" and not dry_run:
        from langslice.job.formats import derived_files

        job.emit_results(progress)
        artifacts.append({"path": str(Path(job.results_path).resolve()), "kind": "results"})
        artifacts += [{"path": str(path), "kind": kind}
                      for path, kind in derived_files(job.layout, job.state)]
    if verb == "export_maps" and isinstance(reply, dict) and not dry_run:
        artifacts += [dict(item) for item in reply.get("files") or []]
    job.views.flush()
    for picture in saved:
        # ``index`` is the picture's place among the call's pictures, the
        # number the reply's ``image_indexes`` give; ``label`` what it shows.
        label = ", ".join(picture.sections) + (f" ({picture.mode})" if picture.mode else "")
        for path, kind in picture.files():
            if path.exists():
                artifacts.append({"path": str(path), "kind": kind, "index": picture.index,
                                  **({"label": label or verb} if kind == "view" else {})})
            elif kind == "view":
                warnings.append(f"picture not saved: {path}")
    result = shape(verb, reply, verbose=verbose)
    if verb == "trace_borders" and isinstance(result, dict):
        result["image_correction"] = _landed(job, result.get("id"))
    if verb == "trace_from_atlas" and isinstance(result, dict):
        for row in result.get("results") or []:  # each landed call's outcome
            if isinstance(row, dict) and row.get("error") is None:
                landed = _landed(job, row.get("id"))
                if landed:
                    row["image_correction"] = landed
    if verb == "status" and isinstance(result, dict):
        from langslice.doors.statement import image_model_state

        result["verbs"] = opened.listed_verbs()
        state = image_model_state(opened.spec, connected=opened.image_model_connected)
        if state is not None:
            result["image_model"] = state
    if dry_run and isinstance(result, dict):
        result["dry_run"] = True
        if VERBS[verb].kind == "write":
            result["would_change"] = changes(before or {}, job.state.to_dict(),
                                             verbose=verbose)
        elif VERBS[verb].scripting:
            warnings.append(f"{verb} wrote nothing under --dry-run; `files` lists what it "
                            "would write.")
        else:
            warnings.append(f"{verb} only reads; --dry-run changes nothing about it.")
    if isinstance(reply, dict):
        for key in WARN_KEYS:
            value = reply.get(key)
            if value:
                warnings.append(f"{key}: {json.dumps(value, default=str)}")
        if ok and isinstance(reply.get("results"), list):  # per-section refusals
            warnings += [f"{row.get('id')}: {row.get('error')}" for row in reply["results"]
                         if isinstance(row, dict) and row.get("status") == "error"]
    if not ok:
        code = str(reply.get("error") or "ERROR")
        message = str(reply.get("message") or reply.get("detail") or "")
        rows = reply.get("results")
        if not message and isinstance(rows, list):  # per-section refusals
            message = "; ".join(f"{row.get('id')}: {row.get('error')}" for row in rows
                                if isinstance(row, dict) and row.get("error"))
        failed = Envelope.failure(code, message, result=result, job=job_folder, verb=verb)
        failed.artifacts, failed.warnings = artifacts, warnings
        return failed
    nexts = ([_command(job_folder, verb, arguments)] if dry_run
             and VERBS[verb].kind == "write" else [])
    return Envelope(result=result, artifacts=artifacts, warnings=warnings, next=nexts)


def _landed(job: Any, section: Any) -> dict[str, Any]:
    """A section's image-model call as it landed (status, error, message,
    cached, attempt); empty for an unknown section or none."""
    record = job.state.resolve(str(section)) if section not in (None, "") else None
    held = (record.image_correction or {}) if record is not None else {}
    return {key: held[key] for key in ("status", "error", "message", "cached", "attempt")
            if key in held}


def shape(verb: str, reply: Any, *, verbose: bool) -> Any:
    """A tool reply for the CLI: pictures out (they are artifacts); concise
    unless *verbose* (no whole-stack rows on a write; a reply with pictures
    keeps what they are as one line, ``picture_note``; *verbose* keeps the
    description and the pictures' text lines as written for a model)."""
    from langslice.doors.tools import TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY

    if not isinstance(reply, dict):
        return reply
    body = dict(reply)
    media = body.pop(TOOL_MEDIA_PARTS_KEY, None)
    body.pop(TOOL_MEDIA_DELIVERY_ID_KEY, None)
    if verbose:
        texts = [item for item in media if isinstance(item, str)] if isinstance(media, list) \
            else []
        if texts:
            body["media_texts"] = texts
        return body
    description = body.pop("description", None)
    pictured = isinstance(media, list) and any(not isinstance(item, str) for item in media)
    if pictured and isinstance(description, str) and description.strip():
        # What the pictures are, on one line (the artifacts list them).
        body["picture_note"] = " ".join(description.split())
    if verb not in ROW_VERBS and isinstance(body.get("rows"), list):
        body["n_rows"] = len(body.pop("rows"))
    if body.get("files_written") and isinstance(body.get("files"), list):
        body["n_files"] = len(body.pop("files"))  # listed under artifacts
    return body


def changes(before: dict[str, Any], after: dict[str, Any], *,
            verbose: bool) -> dict[str, Any]:
    """What a write changed in the state: per section the fields (with both
    values when *verbose*), and the stack's own fields."""
    def by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {str(item.get("id")): item for item in state.get("slices") or []}

    sections: dict[str, Any] = {}
    old, new = by_id(before), by_id(after)
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name, {}), new.get(name, {})
        fields = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
        if fields:
            sections[name] = ({key: {"from": a.get(key), "to": b.get(key)} for key in fields}
                              if verbose else fields)
    stack = sorted(key for key in set(before) | set(after)
                   if key != "slices" and before.get(key) != after.get(key))
    return {"sections": sections, "stack": stack}


# --- init ---------------------------------------------------------------------------


def init(folder: str, rest: list[str], *, atlas_loader: Any = None) -> Envelope:
    """Create (or continue) the job for the image folder *folder*: the job
    flags of ``langslice linear run``; ingest as every host does. With
    ``--registration FILE`` the file's linear registration is the job's
    supplied placement (``doors.jobs.with_registration``): the result's
    ``registration`` says what was placed and what was not, and its
    warnings are the envelope's (``BAD_REGISTRATION`` when it cannot be
    read, matched one to one, or places no section)."""
    from langslice.core.discovery import discover_slices
    from langslice.core.opening import VIEWER_LIMITS
    from langslice.doors.cli.linear import add_linear_arguments, spec_from_args
    from langslice.doors.jobs import create, with_registration
    from langslice.job.job import InputsChanged

    parser = _Parser(prog="langslice job FOLDER init", add_help=False)
    add_linear_arguments(parser)
    parser.add_argument("--notes", default=None, metavar="TEXT",
                        help="The user's notes for this job (job.json), which every door "
                        "gives the registration agent")
    parser.add_argument("--viewer", default=None, choices=sorted(VIEWER_LIMITS),
                        help="Which model reads the pictures (job.json; default claude): "
                        "it sets their largest size")
    args = parser.parse_args(rest)
    images = Path(folder).expanduser().resolve()
    if not images.is_dir() or not discover_slices(str(images)):
        return Envelope.failure("NO_IMAGES", f"No section images in {images}.")
    try:
        spec = spec_from_args(args, str(images))
    except ValueError as exc:
        return Envelope.failure("BAD_ARGUMENTS", str(exc))
    imported: dict[str, Any] | None = None
    if args.registration:
        try:
            spec, imported = with_registration(spec, args.registration,
                                               atlas_loader=atlas_loader, emit=progress)
        except (OSError, ValueError) as exc:
            return Envelope.failure("BAD_REGISTRATION", str(exc), job=str(images))
    try:
        opened = create(spec, atlas_loader=atlas_loader, emit=progress, door="cli")
    except InputsChanged as exc:
        return Envelope.failure("INPUTS_CHANGED", str(exc), job=str(images))
    try:
        from langslice.doors.cli import brief as briefs
        from langslice.doors.jobs import VIEWER_KEY, job_viewer
        from langslice.doors.statement import NOTES_KEY
        from langslice.job.layout import write_job_file

        layout = opened.job.layout
        state = opened.job.state
        settings = {**({NOTES_KEY: args.notes} if args.notes is not None else {}),
                    **({VIEWER_KEY: args.viewer} if args.viewer else {})}
        if settings:
            write_job_file(layout, **settings)
            opened.viewer = job_viewer(layout)
        written = briefs.build(opened, pictures=False)
        listed_verbs = opened.listed_verbs()
        result = {
            "job_folder": str(layout.folder), "image_folder": str(images),
            "sections": [record.id for record in state.in_order()],
            "tasks": list(spec.tasks), "atlas": spec.atlas, "plane": spec.plane,
            "verbs": listed_verbs, "resumed": bool(opened.job.undo_stack
                                                   or opened.job.redo_stack),
            **written.facts,
            "statement": written.statement,
        }
        if imported is not None:
            result["registration"] = {key: value for key, value in imported.items()
                                      if key != "warnings"}
        artifacts = [{"path": str(layout.folder / name), "kind": "card"}
                     for name in ("AGENTS.md", "CLAUDE.md")]
        artifacts.append({"path": str(layout.state_file), "kind": "state"})
        artifacts += written.artifacts
        return Envelope(result=result, artifacts=artifacts,
                        warnings=list((imported or {}).get("warnings") or []),
                        next=[f"langslice job {layout.folder} brief"])
    finally:
        opened.close()


# --- brief ----------------------------------------------------------------------------


def brief(folder: str, *, atlas_loader: Any = None) -> Envelope:
    """``brief``: the job statement, the opening pictures saved as files
    (artifacts of kind ``opening``, in reading order) and ``BRIEF.md``
    (:mod:`langslice.doors.cli.brief`)."""
    from langslice.doors.cli import brief as briefs

    opened = _open(folder, persist=True, atlas_loader=atlas_loader)
    try:
        written = briefs.build(opened, pictures=True)
        pictures = [item for item in written.artifacts if item["kind"] == "opening"]
        expected = sum(1 for entry in written.opening if "picture" in entry)
        warnings = ([] if len(pictures) == expected else [
            "The opening pictures were not saved (a lean job keeps no pictures); "
            "`langslice.open_job` gives them to a script."])
        return Envelope(result={
            "job_folder": str(opened.job.folder),
            "statement": written.statement,
            "opening": written.opening,
            "verbs": opened.listed_verbs(),
            **written.facts,
        }, artifacts=written.artifacts, warnings=warnings)
    finally:
        opened.close()


# --- background runs ------------------------------------------------------------------


def runs(folder: str, verb: str, positional: list[str], options: dict[str, Any]) -> Envelope:
    """``runs [ID]`` (every run, or one) and ``wait [ID]`` (the latest without ID)."""
    from langslice.doors.jobs import NoJob, find
    from langslice.job.layout import JobLayout

    try:
        layout = JobLayout(find(folder))
    except NoJob as exc:
        return Envelope.failure("NO_JOB", str(exc))
    job_folder = str(layout.folder)
    if len(positional) > 1:
        return Envelope.failure("BAD_ARGUMENTS", "Give one run id.", job=job_folder)
    if verb == "runs" and not positional:
        listed = background.listing(layout)
        return Envelope(result={"runs": listed},
                        next=[f"langslice job {job_folder} wait {run['id']}"
                              for run in listed if run.get("state") == "running"][:1])
    run_id = positional[0] if positional else background.latest(layout)
    if run_id is None:
        return Envelope.failure("UNKNOWN_RUN", "This job has no background runs.",
                                job=job_folder)
    if verb == "runs":
        record = background.read(layout, run_id)
        if record is None:
            return Envelope.failure("UNKNOWN_RUN", f"No run {run_id}.", job=job_folder)
        return Envelope(result={**background.brief(record),
                                **({"envelope": record["envelope"]}
                                   if "envelope" in record else {})},
                        next=[] if record.get("state") != "running"
                        else [f"langslice job {job_folder} wait {run_id}"])
    record = background.wait(layout, run_id, options.get("timeout"))
    if record is None:
        return Envelope.failure("UNKNOWN_RUN", f"No run {run_id}.", job=job_folder)
    state = record.get("state")
    if state == "running":
        return Envelope.failure("STILL_RUNNING", f"Run {run_id} is still running.",
                                result=background.brief(record), job=job_folder,
                                exit=EXIT_REFUSED)
    if state != "finished" or not isinstance(record.get("envelope"), dict):
        return Envelope.failure("RUN_LOST", f"Run {run_id} ended without an answer.",
                                result=background.brief(record), job=job_folder,
                                exit=EXIT_INTERNAL)
    answer = Envelope.from_dict(record["envelope"], int(record.get("exit", EXIT_OK)))
    if isinstance(answer.result, dict):
        answer.result = {**answer.result, "run": run_id}
    return answer


__all__ = ["EXIT_ARGUMENTS", "add_parser", "execute", "run"]

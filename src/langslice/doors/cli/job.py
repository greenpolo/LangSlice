"""``langslice job FOLDER VERB [arguments]``: one verb on a job folder, for agents.

The agent CLI (phase 5): every verb of :data:`langslice.ops.registry.VERBS`
under the tool's own name (kebab-case accepted: ``set-positions``), plus
``init`` (create the job for a folder of images), ``runs [ID]`` and ``wait
[ID]`` (background runs; ``status`` is only the verb). FOLDER is the job folder or the image folder
beside it.

Arguments are the verb's declared arguments
(:mod:`langslice.doors.declarations`, ``langslice schema VERB``): a JSON
object with ``--args '{...}'`` or ``--args @file.json``, and/or one flag per
argument, ``--name value`` (kebab or snake case; a JSON value, or plain text
for a text argument; a list argument takes the flag again for each item).
Options: ``--dry-run`` (a write verb runs on the job without writing
anything and reports what would change; ``fit_deformable`` and
``trace_borders`` are only checked, not run), ``--background`` (answer at
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
    emit,
    stdout_to_stderr,
)
from langslice.job.lock import JobBusy

logger = logging.getLogger(__name__)

#: Verbs a dry run checks without running (a model call, a long fit that
#: saves records).
CHECKED_ONLY = frozenset({"trace_borders", "fit_deformable"})
#: Reply keys worth a warning when present and not empty.
WARN_KEYS = ("unknown_ids", "unknown", "rejected", "render_failed", "clamped",
             "deformation_cleared", "truncated", "dropped_positions_mm", "not_shown")
#: Verbs whose whole-stack ``rows`` are their answer (kept when concise).
ROW_VERBS = frozenset({"status", "view_stack"})
#: The CLI's own options (the rest are the verb's arguments).
OPTIONS = {"args", "dry_run", "background", "verbose", "timeout", "run_id"}


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
    """One agent-CLI call (see the module text); never raises."""
    name = canonical_verb(verb)
    run_id: str | None = None
    try:
        if name == "init":
            return init(folder, rest, atlas_loader=atlas_loader)
        options, arguments, positional = parse(rest)
        run_id = options.get("run_id")
        if name in ("runs", "wait"):
            return runs(folder, name, positional, options)
        if positional:
            raise _Refusal(Envelope.failure(
                "BAD_ARGUMENTS", f"Unexpected value(s) {positional}: give arguments as "
                "--name value or --args JSON.", verb=name))
        envelope = call(folder, name, arguments, options, atlas_loader=atlas_loader)
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
    return envelope


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
        return open_folder(folder, atlas_loader=atlas_loader, emit=progress, persist=persist)
    except NoJob as exc:
        raise _Refusal(Envelope.failure("NO_JOB", str(exc))) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise _Refusal(Envelope.failure("JOB_UNREADABLE", str(exc), exit=EXIT_REFUSED)) \
            from exc


def call(folder: str, verb: str, flags: dict[str, list[str]], options: dict[str, Any], *,
         atlas_loader: Any = None) -> Envelope:
    """Run *verb* on the job *folder* names (see the module text)."""
    from langslice.ops.registry import VERBS, enabled

    if verb not in VERBS:
        raise _Refusal(Envelope.failure("UNKNOWN_VERB", f"No verb {verb!r}.", verb=verb))
    dry_run = bool(options.get("dry_run"))
    opened = _open(folder, persist=not dry_run, atlas_loader=atlas_loader)
    job_folder = str(opened.job.folder)
    try:
        offered = enabled(opened.spec, scripting=True)
        if verb not in offered:
            raise _Refusal(Envelope.failure(
                "VERB_OFF", f"This job's settings (tasks {opened.spec.tasks}) have no {verb}.",
                result={"verbs": offered}, job=job_folder, verb=verb))
        tools = {tool.__name__: tool for tool in opened.tools().tools}
        tool = tools[verb]
        arguments = arguments_for(tool, options, flags, verb)
        if options.get("run_id"):
            background.begin(opened.job.layout, options["run_id"])
        elif options.get("background"):
            run_id = background.start(opened.job.layout, verb, arguments,
                                      verbose=bool(options.get("verbose")))
            return Envelope(result={"run": run_id, "verb": verb, "state": "running"},
                            next=[f"langslice job {job_folder} wait {run_id}"])
        if dry_run and verb in CHECKED_ONLY:
            return _checked(opened, verb, arguments)
        return _run(opened, verb, tool, arguments, dry_run=dry_run,
                    verbose=bool(options.get("verbose")))
    finally:
        opened.close()


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
    if verb == "trace_borders":
        job.settle_image_corrections()  # this process ends: the call lands now
    artifacts: list[dict[str, str]] = []
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
        for path, kind in picture.files():
            if path.exists():
                artifacts.append({"path": str(path), "kind": kind})
            elif kind == "view":
                warnings.append(f"picture not saved: {path}")
    result = shape(verb, reply, verbose=verbose)
    if verb == "trace_borders" and isinstance(result, dict):
        record = job.state.resolve(str(result.get("id", ""))) if result.get("id") else None
        held = (record.image_correction or {}) if record is not None else {}
        result["image_correction"] = {key: held[key] for key in (
            "status", "error", "message", "cached", "attempt") if key in held}
    if verb == "status" and isinstance(result, dict):
        from langslice.ops.registry import enabled

        result["verbs"] = enabled(opened.spec, scripting=True)
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
    nexts = ([f"langslice job {job_folder} {verb} --args "
              f"'{json.dumps(arguments, default=str)}'"] if dry_run
             and VERBS[verb].kind == "write" else [])
    return Envelope(result=result, artifacts=artifacts, warnings=warnings, next=nexts)


def shape(verb: str, reply: Any, *, verbose: bool) -> Any:
    """A tool reply for the CLI: pictures out (they are artifacts); concise
    unless *verbose* (no model-facing descriptions, no whole-stack rows on a
    write)."""
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
    body.pop("description", None)
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
    flags of ``langslice linear run``; ingest as every host does."""
    from langslice.core.discovery import discover_slices
    from langslice.doors.cli.linear import add_linear_arguments, build_linear_spec
    from langslice.doors.jobs import create
    from langslice.ops.registry import enabled

    parser = _Parser(prog="langslice job FOLDER init", add_help=False)
    add_linear_arguments(parser)
    args = parser.parse_args(rest)
    images = Path(folder).expanduser().resolve()
    if not images.is_dir() or not discover_slices(str(images)):
        return Envelope.failure("NO_IMAGES", f"No section images in {images}.")
    try:
        spec = build_linear_spec(args, str(images))
    except ValueError as exc:
        return Envelope.failure("BAD_ARGUMENTS", str(exc))
    opened = create(spec, atlas_loader=atlas_loader, emit=progress)
    try:
        layout = opened.job.layout
        state = opened.job.state
        result = {
            "job_folder": str(layout.folder), "image_folder": str(images),
            "sections": [record.id for record in state.in_order()],
            "tasks": list(spec.tasks), "atlas": spec.atlas, "plane": spec.plane,
            "verbs": enabled(spec, scripting=True), "resumed": bool(opened.job.undo_stack
                                                    or opened.job.redo_stack),
        }
        artifacts = [{"path": str(layout.folder / name), "kind": "card"}
                     for name in ("AGENTS.md", "CLAUDE.md")]
        artifacts.append({"path": str(layout.state_file), "kind": "state"})
        return Envelope(result=result, artifacts=artifacts,
                        next=[f"langslice job {layout.folder} status"])
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

"""The agent CLI's answer: one JSON envelope on stdout, an exit code.

Every agent command (``langslice ops``, ``schema``, ``job FOLDER ...``)
prints exactly one JSON object on stdout and nothing else; progress and logs
go to stderr. The envelope::

    {"ok": true|false,
     "result": {...},                       # the verb's reply, concise
     "artifacts": [{"path": "...", "kind": "view"}, ...],
     "warnings": ["..."],
     "next": ["langslice job ... wait <id>"],
     "error": {"code": "...", "message": "...", "fix": "..."}}   # only when not ok

Exit codes: 0 ok, 2 bad arguments (the call itself is wrong), 3 refused by
the job (a gate or a rule: the stack is not in a state the verb accepts),
4 internal error. Pictures are never inlined: they are saved in the job
folder and listed under ``artifacts`` by absolute path and kind (``view``
the JPEG, ``view_json`` its frame, ``labels`` the uint32 atlas ids,
``borders`` the drawn borders, ``results`` an export, ``card`` the
reference card).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

EXIT_OK = 0
EXIT_ARGUMENTS = 2
EXIT_REFUSED = 3
EXIT_INTERNAL = 4

#: Refusal codes that mean the call's arguments are wrong (exit 2) rather
#: than the job refusing it (exit 3), besides every ``BAD_*``, ``UNKNOWN_*``
#: and ``TOO_MANY_*`` code.
ARGUMENT_CODES = frozenset({
    "MISSING_ARGUMENTS", "VIEW_KEY_UNUSED", "DUPLICATE_SLICE_IDS", "ZOOM_UNSUPPORTED",
    "INVALID_BORDER_STYLE", "RESOLUTION_FIXED", "NO_SIDES", "LABEL_MAP_ANTS_ONLY",
    "ENGINE_FIXED", "EMPTY_RESULT", "NO_JOB", "NO_IMAGES", "NOT_A_RUN",
})

#: What to do about a code (the envelope's ``error.fix``); ``{job}`` is the
#: job folder, ``{verb}`` the verb.
FIXES: dict[str, str] = {
    "UNKNOWN_VERB": "List the verbs with `langslice ops`.",
    "VERB_OFF": "This job's tasks do not include this verb; `langslice job {job} status` "
                "lists the job's verbs under `verbs`.",
    "UNKNOWN_ARGUMENTS": "Check the argument names with `langslice schema {verb}`.",
    "MISSING_ARGUMENTS": "Give every required argument; see `langslice schema {verb}`.",
    "BAD_JSON": "Pass --args a JSON object, or @path/to/file.json.",
    "UNKNOWN_SLICE_IDS": "Use a filename or corrected index from "
                         "`langslice job {job} status`.",
    "UNKNOWN_SECTION": "Use a filename or corrected index from `langslice job {job} status`.",
    "NO_JOB": "Create the job first: `langslice job <image folder> init`.",
    "JOB_EXISTS": "Open it with `langslice job {job} status`, or pass --resume to init.",
    "NO_IMAGES": "Point init at a folder of section images (TIFF, PNG or JPEG).",
    "NOTHING_TO_UNDO": "Nothing was written since the job opened its history.",
    "NOTHING_TO_REDO": "Redo follows an undo only.",
    "NO_POSITION": "Write the section's position first (set_positions).",
    "MISSING_POSITIONS": "Write every section's position first (set_positions).",
    "MISSING_TRANSFORMS": "Give every section a transform (fit_affine or adjust_transforms).",
    "MISSING_DEFORMATIONS": "Give every section a deformation (fit_deformable), or record "
                            "that its linear placement stands (fit_deformable with "
                            "keep_linear).",
    "ORDER_POSITION_MISMATCH": "Make positions run one way along the order "
                               "(reorder_slices or set_positions).",
    "LOCKED": "The user locked this section's in-plane alignment; leave it out.",
    "UNKNOWN_RUN": "List the background runs with `langslice job {job} status`.",
    "STILL_RUNNING": "Wait again: `langslice job {job} wait <id>`.",
    "RUN_LOST": "The background process ended without an answer; see its log under "
                "logs/runs/ and run the verb again.",
}


def exit_code(code: str) -> int:
    """The exit code of a refusal *code*."""
    if code in ARGUMENT_CODES or code.startswith(("BAD_", "UNKNOWN_", "TOO_MANY_")):
        return EXIT_ARGUMENTS
    return EXIT_REFUSED


@dataclass
class Envelope:
    """One answer (see the module text)."""

    ok: bool = True
    result: Any = None
    artifacts: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    exit: int = EXIT_OK

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, "result": self.result,
                               "artifacts": self.artifacts, "warnings": self.warnings,
                               "next": self.next}
        if self.error is not None:
            out["error"] = self.error
        return out

    @classmethod
    def failure(cls, code: str, message: str = "", *, exit: int | None = None,
                fix: str | None = None, result: Any = None, job: str = "<folder>",
                verb: str = "VERB") -> Envelope:
        """A refused or failed call; *fix* defaults to :data:`FIXES`."""
        status = exit if exit is not None else exit_code(code)
        hint = fix if fix is not None else FIXES.get(code, (
            f"See `langslice schema {verb}` for the arguments." if status == EXIT_ARGUMENTS
            else f"See the stack with `langslice job {job} status`, then retry."
            if status == EXIT_REFUSED else "Report it; the log is on stderr."))
        hint = hint.replace("{job}", job).replace("{verb}", verb)
        return cls(ok=False, result=result, exit=status,
                   error={"code": code, "message": message, "fix": hint},
                   next=[])

    @classmethod
    def from_dict(cls, data: dict[str, Any], exit: int) -> Envelope:
        return cls(ok=bool(data.get("ok")), result=data.get("result"),
                   artifacts=list(data.get("artifacts") or []),
                   warnings=list(data.get("warnings") or []), next=list(data.get("next") or []),
                   error=data.get("error"), exit=exit)


def dumps(envelope: Envelope) -> str:
    """The envelope as one line of JSON."""
    return json.dumps(envelope.to_dict(), default=str, ensure_ascii=False)


def emit(envelope: Envelope) -> int:
    """Print the envelope on stdout; its exit code."""
    sys.stdout.write(dumps(envelope) + "\n")
    sys.stdout.flush()
    return envelope.exit


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Everything printed to stdout in the block (Python or native libraries:
    Elastix, ITK) lands on stderr, so stdout holds the envelope alone."""
    sys.stdout.flush()
    saved: int | None = None
    try:
        fd = sys.stdout.fileno()
        saved = os.dup(fd)
        os.dup2(sys.stderr.fileno(), fd)
    except (AttributeError, OSError, io.UnsupportedOperation):
        saved = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        if saved is not None:
            os.dup2(saved, sys.stdout.fileno())
            os.close(saved)

"""The agent CLI's answer: one JSON envelope on stdout, an exit code.

Every agent command (``langslice-job ops``, ``schema``, ``job FOLDER ...``)
prints exactly one JSON object on stdout and nothing else; progress and logs
go to stderr. The envelope::

    {"ok": true|false,
     "picture_files": ["/.../view.jpg", ...],   # only when the call drew pictures
     "result": {...},                       # the verb's reply, concise
     "artifacts": [{"path": "...", "kind": "view"}, ...],
     "warnings": ["..."],
     "next": ["langslice-job ... wait <id>"],
     "error": {"code": "...", "message": "...", "fix": "..."}}   # only when not ok

Exit codes: 0 ok, 2 bad arguments (the call itself is wrong), 3 refused by
the job (a gate or a rule: the stack is not in a state the verb accepts),
4 internal error. Pictures are never inlined: they are saved in the job
folder and listed under ``artifacts`` by absolute path and kind (``view``
the JPEG, ``view_json`` its frame, ``labels`` the uint32 atlas ids,
``borders`` the drawn borders, ``results`` an export, ``card`` the
reference card); ``picture_files`` repeats the JPEGs' paths (kinds
:data:`PICTURE_KINDS`), in order, second in the envelope, where an agent
reading the reply sees them first.
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langslice.job.settings.needs import render
from langslice.tools.descriptions.fixes import FIXES

#: The artifacts that are pictures to open (``picture_files``).
PICTURE_KINDS = ("view", "opening")

EXIT_OK = 0
EXIT_ARGUMENTS = 2
EXIT_REFUSED = 3
EXIT_INTERNAL = 4

#: Refusal codes that mean the call's arguments are wrong (exit 2) rather
#: than the job refusing it (exit 3), besides every ``BAD_*``, ``UNKNOWN_*``
#: and ``TOO_MANY_*`` code.
ARGUMENT_CODES = frozenset({
    "MISSING_ARGUMENTS", "DUPLICATE_SLICE_IDS", "NO_SIDES", "WRONG_SIDE_PAIR", "EMPTY_BOX",
    "NO_JOB", "NO_IMAGES", "ATLAS_ENDPOINT_OUTSIDE", "NOT_FOUND",
})

#: The disabled list of the job the call is about, once it is open (set by
#: ``doors.agent_cli.calls``): the fixes name no tool the job disables.
DISABLED: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "langslice_disabled", default=())


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
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    exit: int = EXIT_OK
    #: The call's arguments as the verb read them (the trace's ``args``);
    #: never printed.
    call: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok}
        pictures = [str(item.get("path")) for item in self.artifacts
                    if item.get("kind") in PICTURE_KINDS]
        if pictures:
            out["picture_files"] = pictures
        out.update({"result": self.result, "artifacts": self.artifacts,
                    "warnings": self.warnings, "next": self.next})
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
            f"See `langslice-job schema {verb}` for the arguments." if status == EXIT_ARGUMENTS
            else f"See the stack with `langslice-job {job} status`, then retry."
            if status == EXIT_REFUSED else "Report it; the log is on stderr."))
        hint = render(hint, DISABLED.get()).replace("{job}", job).replace("{verb}", verb)
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
    return json.dumps(envelope.to_dict(), default=str, ensure_ascii=True)


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

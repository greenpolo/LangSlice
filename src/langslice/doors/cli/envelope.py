"""The agent CLI's answer: one JSON envelope on stdout, an exit code.

Every agent command (``langslice-job ops``, ``schema``, ``job FOLDER ...``)
prints exactly one JSON object on stdout and nothing else; progress and logs
go to stderr. The envelope::

    {"ok": true|false,
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
    "MISSING_ARGUMENTS", "DUPLICATE_SLICE_IDS", "NO_SIDES", "EMPTY_BOX", "NO_JOB",
    "NO_IMAGES", "RETIRED_TOOL",
})

#: What to do about a code (the envelope's ``error.fix``); ``{job}`` is the
#: job folder, ``{verb}`` the verb.
FIXES: dict[str, str] = {
    "UNKNOWN_VERB": "List the verbs with `langslice-job ops`.",
    "RETIRED_TOOL": "Call the verb named under result.use instead (see the message); "
                    "`langslice-job ops` lists the verbs.",
    "VERB_OFF": "This job's tasks do not include this verb; `langslice-job {job} status` "
                "lists the job's verbs under `verbs`.",
    "UNKNOWN_ARGUMENTS": "Check the argument names with `langslice-job schema {verb}`.",
    "MISSING_ARGUMENTS": "Give every required argument; see `langslice-job schema {verb}`.",
    "BAD_JSON": "Pass --args a JSON object, or @path/to/file.json.",
    "UNKNOWN_SLICE_IDS": "Name sections by filename, as `langslice-job {job} status` "
                         "lists them.",
    "UNKNOWN_SECTION": "Name sections by filename, as `langslice-job {job} status` "
                       "lists them.",
    "NO_JOB": "Create the job first: `langslice-job <image folder> init`.",
    "NO_IMAGES": "Point init at a folder of section images (TIFF, PNG or JPEG).",
    "NOTHING_TO_UNDO": "The job's history holds no earlier step.",
    "NOTHING_TO_REDO": "Redo follows an undo only.",
    "NO_POSITION": "Write the section's position first (position_sections).",
    "MISSING_POSITIONS": "Write every section's own position first (position_sections); "
                         "a starting position the job gave a section does not count.",
    "MISSING_TRANSFORMS": "Give every section a transform (elastix_affine or "
                          "interactive_transform).",
    "MISSING_DEFORMATIONS": "Give every section a deformation (ants_syn or trace_borders), "
                            "or name it in submit's left_linear with the reason its linear "
                            "placement stands.",
    "DEFORMATION_REQUIRED": "The user requires a deformation on every section: fit one "
                            "(ants_syn or trace_borders) instead of naming it in "
                            "left_linear.",
    "HAS_DEFORMATION": "That section carries a deformation; leave it out of left_linear.",
    "POSITIONS_SUPPLIED": "The user supplied the positions; leave sections out.",
    "ANGLES_SUPPLIED": "The user supplied the cutting angles; leave cutting_angles out.",
    "LOCKED": "The user locked this section's in-plane alignment; leave it out.",
    "STALE_INPUT": "The section changed while this ran; run the verb again for it.",
    "JOB_BUSY": "Another writer held the job folder's lock; retry.",
    "NOTHING_FITTED": "Each section's row under result.results names its problem.",
    "NOTHING_WRITTEN": "Each section's row under result.results names its problem.",
    "UNKNOWN_RUN": "List the background runs with `langslice-job {job} runs`.",
    "STILL_RUNNING": "Wait again: `langslice-job {job} wait <id>`.",
    "RUN_LOST": "The background process ended without an answer; see its log under "
                "logs/runs/ and run the verb again.",
    "BAD_REGISTRATION": "Give --registration a QuickNII/VisuAlign JSON or XML, a DeepSlice "
                        "CSV/JSON/XML or a LangSlice registration.json whose entries name "
                        "this folder's section images (by file name, stem or _sNNN number) "
                        "on the job's atlas.",
    "INPUTS_CHANGED": "Rerun `langslice-job {job} init` with --fresh to start a new job "
                      "from these inputs (the old one's work is replaced), or with the "
                      "inputs the job was made from (job.json, spec.inputs) to continue it.",
    "IMAGE_MODEL_OFF": "Connect the job's image model to LangSlice (`langslice login` for "
                       "openai-oauth, or its API key) and call it again; without it, fit "
                       "each section's deformation to its stain with ants_syn.",
    "NOTHING_TRACED": "Each section's row under result.results names its problem; a "
                      "section needs a position and a transform (elastix_affine or "
                      "interactive_transform) before trace_borders.",
    "NOT_COMPARED": "Look at each section in mode overlay or positioning since its last "
                    "write, then write its position again.",
    "NOT_REVIEWED": "Run look in mode positioning over every section after the last "
                    "position_sections write, then submit again.",
    "INTERVAL_BREAKS_UNSUPPORTED": "Report a break only where the written spacing exceeds "
                                   "1.5x the stack's median spacing; check the positions "
                                   "on both sides first.",
    "STRICT_INTERVAL": "Make every spacing within 10% of the nominal interval and report "
                       "no break (the user asked for a strict interval).",
    "FLIP_DISABLED": "Flipping is switched off for this job; leave flip out.",
    "KEEPS_HOST_WARP": "The user keeps this section's own deformation; leave it out of "
                       "the nonlinear verbs.",
    "NONLINEAR_SKIPPED": "The user left this section out of the nonlinear task; leave it "
                         "out.",
    "INVALID_LINEAR_PLACEMENT": "Give the section a position and a transform "
                                "(elastix_affine or interactive_transform) first.",
    "ANTS_MISSING": "ANTs (antspyx) does not import on this host; reinstall LangSlice "
                    "there (see the message), or name the sections in submit's "
                    "left_linear.",
    "FIT_FAILED": "See the message; try other settings or regions.",
    "RENDER_FAILED": "See the message; check the section file opens, then retry.",
    "UNAVAILABLE": "Not installed on this host (see the message); use what the job "
                   "statement offers instead.",
    "FIT_ATLAS_UNAVAILABLE": "Use atlas_image \"template\" (the reference template).",
    "NO_PICTURE": "Look first: zoom takes a box on a picture already drawn.",
    "UNKNOWN_PICTURE": "Use a picture number from a reply's `pictures`, or views.jsonl.",
    "PICTURE_NOT_SAVED": "This job keeps no pictures between calls; draw it again with look "
                         "and zoom in the same process.",
    "EMPTY_BOX": "Give a box with area inside the picture: [x0, y0, x1, y1] in its pixels.",
    "NOTHING_EXPORTED": "Each skipped section's reason is under result.skipped; place it "
                        "first.",
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
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    exit: int = EXIT_OK
    #: The call's arguments as the verb read them (the trace's ``args``);
    #: never printed.
    call: dict[str, Any] | None = field(default=None, repr=False, compare=False)

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
            f"See `langslice-job schema {verb}` for the arguments." if status == EXIT_ARGUMENTS
            else f"See the stack with `langslice-job {job} status`, then retry."
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

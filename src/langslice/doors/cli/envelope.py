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
    "ENGINE_FIXED", "EMPTY_RESULT", "NO_JOB", "NO_IMAGES",
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
    "NO_IMAGES": "Point init at a folder of section images (TIFF, PNG or JPEG).",
    "NOTHING_TO_UNDO": "The job's history holds no earlier step.",
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
    "STALE_INPUT": "The section changed while this ran; run the verb again for it.",
    "JOB_BUSY": "Another writer held the job folder's lock; retry.",
    "NOTHING_FITTED": "Each section's row under result.results names its problem.",
    "NOTHING_ADJUSTED": "Each section's row under result.results names its problem.",
    "NOTHING_WRITTEN": "See result.unknown_ids and result.rejected.",
    "UNKNOWN_RUN": "List the background runs with `langslice job {job} runs`.",
    "STILL_RUNNING": "Wait again: `langslice job {job} wait <id>`.",
    "RUN_LOST": "The background process ended without an answer; see its log under "
                "logs/runs/ and run the verb again.",
    "BAD_REGISTRATION": "Give --registration a QuickNII/VisuAlign JSON or XML, a DeepSlice "
                        "CSV/JSON/XML or a LangSlice registration.json whose entries name "
                        "this folder's section images (by file name, stem or _sNNN number) "
                        "on the job's atlas.",
    "INPUTS_CHANGED": "Rerun `langslice job {job} init` with --fresh to start a new job "
                      "from these inputs (the old one's work is replaced), or with the "
                      "inputs the job was made from (job.json, spec.inputs) to continue it.",
    "IMAGE_MODEL_OFF": "Connect the job's image model to LangSlice (`langslice login` for "
                       "openai-oauth, or its API key) and call it again; without it, fit "
                       "each section's deformation to its stain with fit_deformable.",
    "NO_IMAGE_MODEL": "This job has no image model: use fit_section \"fit\" (the stain).",
    "MISSING_IMAGE_CORRECTIONS": "Run trace_borders for each section listed under "
                                 "result.sections at its current placement (a placement "
                                 "change makes an earlier trace stale), then submit again.",
    "NOTHING_TRACED": "Each section's row under result.results names its problem; a "
                      "section needs a position and a transform (fit_affine or "
                      "adjust_transforms) before trace_borders.",
    "NO_TRACE": "Run trace_borders for the section at its current placement first, or fit "
                "with fit_section \"fit\".",
    "TRACE_STALE": "The section moved since its trace: run trace_borders again for it.",
    "TRACE_RUNNING": "The section's trace is still running: wait for it (fit_deformable "
                     "with a traced fit_section waits), or run trace_borders again.",
    "TRACE_TIMEOUT": "The trace did not land in time: call fit_deformable again later.",
    "TRACE_FAILED": "The image model failed (see the message): run trace_borders again, or "
                    "fit with fit_section \"fit\".",
    "NOT_REVIEWED": "Run view_stack after the last set_positions write, then submit again.",
    "INTERVAL_BREAKS_UNSUPPORTED": "Report a break only where the written spacing exceeds "
                                   "1.5x the stack's median spacing; check the positions "
                                   "on both sides first.",
    "STRICT_INTERVAL": "Make every spacing within 10% of the nominal interval and report "
                       "no break (the user asked for a strict interval).",
    "DAMAGED_REQUIRES_MANUAL_TRANSFORM": "Align each damaged section listed with "
                                         "adjust_transforms (a non-identity manual "
                                         "transform on its surviving anatomy).",
    "DAMAGED": "fit_affine refuses a damaged section unless given include or exclude "
               "regions; or align it with adjust_transforms.",
    "DAMAGE_SET_BY_USER": "The user marked this section damaged; leave the flag as it is.",
    "FLIP_DISABLED": "Flipping is switched off for this job; leave flip out.",
    "KEEPS_HOST_WARP": "The user keeps this section's own deformation; leave it out of "
                       "the nonlinear verbs.",
    "NONLINEAR_SKIPPED": "The user left this section out of the nonlinear task; leave it "
                         "out.",
    "INVALID_LINEAR_PLACEMENT": "Give the section a position and a transform (fit_affine "
                                "or adjust_transforms) first.",
    "NO_DEFORMATION": "start \"current\" needs an applied deformation; use start "
                      "\"linear\".",
    "FIT_FAILED": "See the message; try other settings, regions or engine.",
    "RENDER_FAILED": "See the message; check the section file opens, then retry.",
    "UNAVAILABLE": "Not installed on this host (see the message); use what the job "
                   "statement offers instead.",
    "FIT_ATLAS_UNAVAILABLE": "Use fit_atlas \"ara\" (the reference template).",
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

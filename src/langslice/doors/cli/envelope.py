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

from langslice.core.needs import render

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
    "NO_JOB", "NO_IMAGES", "ATLAS_ENDPOINT_OUTSIDE",
})

#: The disabled list of the job the call is about, once it is open (set by
#: ``doors.cli.job``): the fixes name no tool the job disables.
DISABLED: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "langslice_disabled", default=())


#: What to do about a code (the envelope's ``error.fix``), written with what
#: it needs (:mod:`langslice.core.needs`); ``{job}`` is the job folder,
#: ``{verb}`` the verb.
FIXES: dict[str, str] = {
    "UNKNOWN_VERB": "List the verbs with `langslice-job ops`.",
    "VERB_OFF": "This job's tasks do not include this verb; `langslice-job {job} status` "
                "lists the job's verbs under `verbs`.",
    "UNKNOWN_ARGUMENTS": "Check the argument names with `langslice-job schema {verb}`.",
    "MISSING_ARGUMENTS": "Give every required argument; see `langslice-job schema {verb}`.",
    "BAD_JSON": "Pass --args a JSON object, or @path/to/file.json.",
    "UNKNOWN_SLICE_IDS": "Name sections by filename, as `langslice-job {job} status` "
                         "lists them.",
    "UNKNOWN_SECTION": "Name sections by filename, as `langslice-job {job} status` "
                       "lists them.",
    "NO_JOB": "Create the job first: `langslice-job <image folder> init`; a job made "
              "with --job-dir is named by its own folder.",
    "NO_IMAGES": "Point init at a folder of section images (TIFF, PNG or JPEG).",
    "NOTHING_TO_UNDO": "The job's history holds no earlier step.",
    "NOTHING_TO_REDO": "Redo follows an undo only.",
    "UNKNOWN_REGISTRATION": "List the section's registrations[[?restore_registration : "
                            "`langslice-job {job} restore_registration --section NAME`]].",
    "REGISTRATION_ELSEWHERE": "Put the section back at the position and cutting angles the "
                              "message names, then call it again.",
    "NO_POSITION": "Write the section's position first[[?position_sections  "
                   "(position_sections)]].",
    "MISSING_POSITIONS": "Write every section's own position first[[?position_sections  "
                         "(position_sections)]].",
    "MISSING_TRANSFORMS": "Give every section a transform[[?elastix_affine|"
                          "interactive_transform  ([[%elastix_affine=elastix_affine||"
                          "interactive_transform=interactive_transform]])]].",
    "MISSING_DEFORMATIONS": "Give every section a deformation[[?fireants_syn|trace_borders  "
                            "([[%fireants_syn=fireants_syn||trace_borders=trace_borders]])]].",
    "FROZEN": "The user froze these properties of these sections (the reply names them): "
              "leave them out of the call.",
    "STALE_INPUT": "The section changed while this ran; run the verb again for it.",
    "SECTION_BUSY": "Background work is running on this section (the reply names it); "
    "write to it once the work lands.",
    "JOB_BUSY": "Another writer held the job folder's lock; retry.",
    "NOTHING_FITTED": "Each section's row under result.results names its problem.",
    "NOTHING_WRITTEN": "Each section's row under result.results names its problem.",
    "UNKNOWN_RUN": "List the background runs with `langslice-job {job} runs`.",
    "RUN_LOST": "The background process ended without an answer; see its log under "
                "logs/runs/ and run the verb again. If every background run is lost, "
                "your shell ends the processes a command leaves running when it returns "
                "(sandboxes do this): run the verb as `LANGSLICE_DETACH=0 langslice-job "
                "...`, without --background, in your harness's own background mode if it "
                "has one.",
    "BAD_REGISTRATION": "Give --registration a QuickNII/VisuAlign JSON or XML, a DeepSlice "
                        "CSV/JSON/XML or a LangSlice registration.json whose entries name "
                        "this folder's section images (by file name, stem or _sNNN number) "
                        "on the job's atlas.",
    "INPUTS_CHANGED": "Rerun `langslice-job {job} init` with --fresh to start a new job "
                      "from these inputs (the old one's work is replaced), or with the "
                      "inputs the job was made from (job.json, spec.inputs) to continue it.",
    "IMAGE_MODEL_OFF": "Connect the job's image model to LangSlice (`langslice login` for "
                       "openai-oauth, or its API key) and call it again[[?fireants_syn ; "
                       "without it, fit each section's deformation to its stain with "
                       "fireants_syn]].",
    "NOTHING_TRACED": "Each section's row under result.results names its problem"
                      "[[?trace_borders ; a section needs a position and a transform"
                      "[[?elastix_affine|interactive_transform  ([[%elastix_affine="
                      "elastix_affine||interactive_transform=interactive_transform]])]] "
                      "before trace_borders]].",
    "INTERVAL_BREAKS_UNSUPPORTED": "Report a break only where the written spacing exceeds "
                                   "1.5x the stack's median spacing; check the positions "
                                   "on both sides first.",
    "STRICT_INTERVAL": "Make every spacing within 10% of the nominal interval and report "
                       "no break (the user asked for a strict interval).",
    "OPTION_OFF": "That option is switched off for this job; use one of the others the "
                  "message lists.",
    "FLIP_DISABLED": "Flipping is switched off for this job; leave flip out.",
    "INVALID_LINEAR_PLACEMENT": "Give the section a position and a transform"
                                "[[?elastix_affine|interactive_transform  ([[%elastix_affine="
                                "elastix_affine||interactive_transform=interactive_transform"
                                "]])]] first.",
    "ENGINE_MISSING": "torch or fireants does not import on this host; reinstall "
                    "LangSlice (see the message).",
    "FIT_FAILED": "See the message; try other settings or regions.",
    "RENDER_FAILED": "See the message; check the section file opens, then retry.",
    "UNKNOWN_MASK": "Make the mask first[[?make_mask  with make_mask]]; `status` lists each "
                    "section's masks.",
    "STALE_MASK": "The section was turned or flipped since this mask was made; make the mask again"
                  "[[?make_mask  with make_mask]].",
    "UNKNOWN_IMAGE": "`status` lists the section images[[?make_image ; make one with "
                     "make_image]].",
    "MASK_UNSUPPORTED": "This engine sorts tissue itself and takes no mask; the message names "
                        "the engines that do.",
    "NO_DEFORMATION": "The section holds no deformation; fit one first, or make the mask "
                      "with source \"em\".",
    "NO_SORTING": "The section's deformation saved no sorting; make the mask with source "
                  "\"em\".",
    "HAS_DEFORMATION": "Undo the deformation first, or fit a warp on it.",
    "FIT_ATLAS_UNAVAILABLE": "Use atlas_image \"template\" (the reference template).",
    "NO_PICTURE": "Look first[[?zoom : zoom takes a box on a picture already drawn]].",
    "UNKNOWN_PICTURE": "Use a picture number from a reply's `pictures`, or views.jsonl.",
    "PICTURE_NOT_SAVED": "This job keeps no pictures between calls; draw it again with look"
                         "[[?zoom  and zoom]] in the same process.",
    "EMPTY_BOX": "Give a box with area inside the picture: [x0, y0, x1, y1] in its pixels.",
    "TOO_MANY_LANDMARK_VIEWS": "Nothing was written. Send the same pairs with `--view false`, "
                               "or in calls of four pairs or fewer.",
    "BAD_LANDMARK_POINT": "Give x,y inside the image content the message states[[?zoom  (on a "
                          "zoom, numbers its grid prints)]].",
    "ATLAS_ENDPOINT_OUTSIDE": "Nothing was written. Send the pairs again with each atlas "
                              "endpoint the message names at the point inside the atlas it "
                              "gives, or on the drawn border where you meant it.",
    "BAD_LANDMARK_PICTURE":"Draw this section with look (sections=[its filename]) and give "
                            "that picture's number; the message says which sections the "
                            "picture you named covers.",
    "STALE_LANDMARK_PLANE": "Draw the section's current atlas plane with look, then re-place "
                            "the pairs the message names (their atlas endpoints), or delete "
                            "them.",
    "UNKNOWN_LANDMARK": "Use a pair id the message lists[[?set_landmarks : set_landmarks "
                        "with only the section lists the pairs]].",
    "INSUFFICIENT_LANDMARKS": "Add pairs[[?set_landmarks  with set_landmarks]] until the "
                              "section has at least three, then apply again.",
    "LANDMARK_TRIANGULATION_FAILED": "Move or delete the pairs the message names"
                                     "[[?set_landmarks  with set_landmarks]], then apply "
                                     "again.",
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

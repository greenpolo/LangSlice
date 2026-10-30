"""The refusal: what LangSlice answers when it cannot do what it was asked.

A refusal is one exception, :class:`Refused`, carrying a
:class:`RefusalCode` (the machine-readable reason, the same string on every
door), a message saying what was attempted and what to do, and the plain
facts behind it (the ids, the allowed values). Nothing is written for a
refused call. The core, the job and the tools raise it; each door turns it
into its own answer once: the tool reply (:meth:`Refused.payload`), the
agent CLI's envelope, the MCP text block, the library's reply. A per-section
answer carries a refused section as its row, the other sections going on;
a crash is answered as a refusal whose code is the exception's class name.

The codes are strings the agent reads, so a member's value never changes.
A member's name is its value (one excepted, :attr:`RefusalCode.NO_SUCH_PATH`,
whose value names a word the style guide keeps out of names), so a reader
who meets a code in a reply finds it here with its meaning.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class TextEnum(StrEnum):
    """A string enum whose members read as their plain strings everywhere.

    ``repr`` is the value's, so a reply that holds a member prints exactly
    as one that holds the string; ``str``, formatting, comparison, hashing
    and JSON already are the string's.
    """

    def __repr__(self) -> str:
        return repr(self.value)


class ReplyStatus(TextEnum):
    """The ``status`` word of a reply or a per-section row."""

    #: The call did what it was asked.
    OK = "ok"
    #: The call could not run: nothing was written.
    ERROR = "error"
    #: A gate: the call is valid, but the job is not ready for it.
    REFUSED = "refused"


#: The reply statuses that mean a call failed (every other status, ``ok``,
#: ``started``, ``running``, ``submitted`` ..., is not a failure): each
#: ``tool_end`` event carries it as ``failed`` for the hosts, and the agent
#: CLI answers it with a refusal's envelope.
FAILED_STATUSES = (ReplyStatus.ERROR, ReplyStatus.REFUSED)


class RefusalCode(TextEnum):
    """Every refusal code, once, with its meaning."""

    #: A landmark's atlas endpoint lies outside the atlas.
    ATLAS_ENDPOINT_OUTSIDE = "ATLAS_ENDPOINT_OUTSIDE"
    #: The atlas plane at a section's position cannot be drawn.
    ATLAS_RENDER_FAILED = "ATLAS_RENDER_FAILED"
    #: A tool argument is malformed or out of range.
    BAD_ARGS = "BAD_ARGS"
    #: The agent CLI's command line cannot be parsed.
    BAD_ARGUMENTS = "BAD_ARGUMENTS"
    #: A zoom box is not four numbers.
    BAD_BOX = "BAD_BOX"
    #: A configured MCP plugin names an unknown rule or layer kind.
    BAD_CONFIG = "BAD_CONFIG"
    #: A deformation map given to the job is not a usable field.
    BAD_FIELD = "BAD_FIELD"
    #: The agent CLI's ``--args`` is not a JSON object.
    BAD_JSON = "BAD_JSON"
    #: A landmark endpoint names a picture that cannot carry it.
    BAD_LANDMARK_PICTURE = "BAD_LANDMARK_PICTURE"
    #: A landmark endpoint falls outside its picture's image.
    BAD_LANDMARK_POINT = "BAD_LANDMARK_POINT"
    #: A plugin's picture layer is not a layer its kind and frame allow.
    BAD_LAYER = "BAD_LAYER"
    #: A transform matrix given to the job is degenerate or not finite.
    BAD_MATRIX = "BAD_MATRIX"
    #: A path lies outside the job folder or cannot be read there.
    BAD_PATH = "BAD_PATH"
    #: A transform's pivot cannot be placed on the section's canvas.
    BAD_PIVOT = "BAD_PIVOT"
    #: A plane given by a plugin is not a plane through the atlas.
    BAD_PLANE = "BAD_PLANE"
    #: A region zoom asked of a picture that is not a one-section overlay.
    BAD_REGION_PICTURE = "BAD_REGION_PICTURE"
    #: An imported registration file cannot be read for this folder.
    BAD_REGISTRATION = "BAD_REGISTRATION"
    #: A quarter turn other than 0, 90, 180 or 270.
    BAD_ROTATION = "BAD_ROTATION"
    #: A fit's settings are not ones its engine takes.
    BAD_SETTINGS = "BAD_SETTINGS"
    #: ``look``'s ``warp`` is not one of its choices.
    BAD_WARP = "BAD_WARP"
    #: A section's saved deformation cannot be read.
    DEFORMATION_UNREADABLE = "DEFORMATION_UNREADABLE"
    #: A call names the same section twice.
    DUPLICATE_SLICE_IDS = "DUPLICATE_SLICE_IDS"
    #: A zoom box has no area inside its picture.
    EMPTY_BOX = "EMPTY_BOX"
    #: A plugin's registration engine ran and failed.
    ENGINE_FAILED = "ENGINE_FAILED"
    #: The deformable engine (FireANTs on torch) cannot import here.
    ENGINE_MISSING = "ENGINE_MISSING"
    #: A plugin's registration engine is not installed here.
    ENGINE_UNAVAILABLE = "ENGINE_UNAVAILABLE"
    #: The agent CLI could not set up or change the job's environment.
    ENV_FAILED = "ENV_FAILED"
    #: A failed reply that names no code of its own.
    ERROR = "ERROR"
    #: A fit's engine raised; the message is its error's first line.
    FIT_FAILED = "FIT_FAILED"
    #: A flip asked while the job switches flipping off.
    FLIP_DISABLED = "FLIP_DISABLED"
    #: A write would change a property the job froze.
    FROZEN = "FROZEN"
    #: An affine fit asked of a section that holds a deformation.
    HAS_DEFORMATION = "HAS_DEFORMATION"
    #: A trace's image call never answered: still running or interrupted.
    IMAGE_CALL_IN_PROGRESS_OR_INTERRUPTED = (
        "IMAGE_CALL_IN_PROGRESS_OR_INTERRUPTED"
    )
    #: A trace's record could not be written or read in the job folder.
    IMAGE_CORRECTION_IO_ERROR = "IMAGE_CORRECTION_IO_ERROR"
    #: A verb that needs the job's image model, which is not connected.
    IMAGE_MODEL_OFF = "IMAGE_MODEL_OFF"
    #: The folder's job was made from other supplied inputs.
    INPUTS_CHANGED = "INPUTS_CHANGED"
    #: A landmark warp asked with fewer than three pairs.
    INSUFFICIENT_LANDMARKS = "INSUFFICIENT_LANDMARKS"
    #: The agent CLI met an error of its own.
    INTERNAL = "INTERNAL"
    #: Interval breaks reported where the spacing shows no gap.
    INTERVAL_BREAKS_UNSUPPORTED = "INTERVAL_BREAKS_UNSUPPORTED"
    #: A section lacks the position or transform the call needs.
    INVALID_LINEAR_PLACEMENT = "INVALID_LINEAR_PLACEMENT"
    #: Another writer holds the job folder's lock.
    JOB_BUSY = "JOB_BUSY"
    #: The job folder's files cannot be read as a job.
    JOB_UNREADABLE = "JOB_UNREADABLE"
    #: The landmark mesh cannot be built (folded or degenerate).
    LANDMARK_TRIANGULATION_FAILED = "LANDMARK_TRIANGULATION_FAILED"
    #: The section's current warp cannot carry landmarks.
    LANDMARK_WARP_UNAVAILABLE = "LANDMARK_WARP_UNAVAILABLE"
    #: A plugin layer cannot be placed on the picture asked.
    LAYER_NOT_DRAWABLE = "LAYER_NOT_DRAWABLE"
    #: A plugin engine takes no section mask.
    MASK_UNSUPPORTED = "MASK_UNSUPPORTED"
    #: The ``mcp`` package is not installed for an MCP role.
    MCP_MISSING = "MCP_MISSING"
    #: A configured MCP server did not answer in time.
    MCP_TIMEOUT = "MCP_TIMEOUT"
    #: A configured MCP server answered with an error.
    MCP_TOOL_ERROR = "MCP_TOOL_ERROR"
    #: A configured MCP server cannot be reached.
    MCP_UNAVAILABLE = "MCP_UNAVAILABLE"
    #: The agent CLI call leaves out a required argument.
    MISSING_ARGUMENTS = "MISSING_ARGUMENTS"
    #: Submit gate: a section the host requires to warp holds no deformation.
    MISSING_DEFORMATIONS = "MISSING_DEFORMATIONS"
    #: Submit gate: a section has no position of its own.
    MISSING_POSITIONS = "MISSING_POSITIONS"
    #: Submit gate: a section has no transform.
    MISSING_TRANSFORMS = "MISSING_TRANSFORMS"
    #: A made image asked together with other images in one picture.
    MIXED_IMAGES = "MIXED_IMAGES"
    #: A stack engine asked of fewer sections than it needs.
    NEEDS_STACK = "NEEDS_STACK"
    #: A point asked of a picture that draws no atlas.
    NO_ATLAS_IN_PICTURE = "NO_ATLAS_IN_PICTURE"
    #: The section holds no deformation for the call to read.
    NO_DEFORMATION = "NO_DEFORMATION"
    #: An MCP call that names no job.
    NO_FOLDER = "NO_FOLDER"
    #: The folder holds no section images.
    NO_IMAGES = "NO_IMAGES"
    #: The folder holds no job.
    NO_JOB = "NO_JOB"
    #: No picture to zoom, and no saved copy to crop.
    NO_PICTURE = "NO_PICTURE"
    #: The section has no atlas position yet.
    NO_POSITION = "NO_POSITION"
    #: An atlas picture asked without positions or sections.
    NO_POSITIONS = "NO_POSITIONS"
    #: The section has no marked regions to draw.
    NO_REGIONS = "NO_REGIONS"
    #: A one-sided region on a section whose plane has no two sides.
    NO_SIDES = "NO_SIDES"
    #: The section's deformation saved no sorting to make a mask from.
    NO_SORTING = "NO_SORTING"
    #: The atlas has no region hierarchy.
    NO_STRUCTURES = "NO_STRUCTURES"
    #: The section's tissue cannot be told from its slide.
    NO_TISSUE = "NO_TISSUE"
    #: A traced fit asked of a section with no completed trace.
    NO_TRACE = "NO_TRACE"
    #: An interval break named where the spacing shows no gap.
    NOT_A_GAP = "NOT_A_GAP"
    #: A stack engine asked of sections that are not one series.
    NOT_A_STACK = "NOT_A_STACK"
    #: A plugin context call that a script cannot make.
    NOT_FOR_SCRIPTS = "NOT_FOR_SCRIPTS"
    #: A job-folder path that does not exist; its code is ``NOT_FOUND``.
    NO_SUCH_PATH = "NOT_FOUND"
    #: Background work could not be started for a section.
    NOT_STARTED = "NOT_STARTED"
    #: A plugin's write left the section unwritten.
    NOT_WRITTEN = "NOT_WRITTEN"
    #: No section could be exported.
    NOTHING_EXPORTED = "NOTHING_EXPORTED"
    #: No section of the call was fitted.
    NOTHING_FITTED = "NOTHING_FITTED"
    #: Redo asked with no undone step to redo.
    NOTHING_TO_REDO = "NOTHING_TO_REDO"
    #: Undo asked with no earlier step.
    NOTHING_TO_UNDO = "NOTHING_TO_UNDO"
    #: No section of the call was written.
    NOTHING_WRITTEN = "NOTHING_WRITTEN"
    #: A write before every opening page was read.
    OPENING_NOT_READ = "OPENING_NOT_READ"
    #: An option the job switches off.
    OPTION_OFF = "OPTION_OFF"
    #: A picture number with no saved image to crop.
    PICTURE_NOT_SAVED = "PICTURE_NOT_SAVED"
    #: A plugin tool's handler raised.
    PLUGIN_FAILED = "PLUGIN_FAILED"
    #: The positioning picture asked while it is switched off.
    POSITIONING_OFF = "POSITIONING_OFF"
    #: A plugin tool declared ``read`` tried to write.
    READ_TOOL_WRITES = "READ_TOOL_WRITES"
    #: A fit's record could not be saved in the job folder.
    RECORD_WRITE_FAILED = "RECORD_WRITE_FAILED"
    #: A region zoom names a region absent from the picture.
    REGION_NOT_IN_PLANE = "REGION_NOT_IN_PLANE"
    #: A restricted fit's regions are too small for the engine.
    REGION_TOO_SMALL = "REGION_TOO_SMALL"
    #: A registration whose deformation was fitted at another placement.
    REGISTRATION_ELSEWHERE = "REGISTRATION_ELSEWHERE"
    #: A picture could not be drawn; the write stands.
    RENDER_FAILED = "RENDER_FAILED"
    #: A background run ended without an answer.
    RUN_LOST = "RUN_LOST"
    #: Background work is running on the section.
    SECTION_BUSY = "SECTION_BUSY"
    #: The folder's job was made with other settings.
    SETTINGS_CHANGED = "SETTINGS_CHANGED"
    #: The placement's midline is too oblique to name sides.
    SIDES_AMBIGUOUS = "SIDES_AMBIGUOUS"
    #: A stack engine asked of more sections than it holds in memory.
    STACK_TOO_LARGE = "STACK_TOO_LARGE"
    #: The section changed while a long call computed.
    STALE_INPUT = "STALE_INPUT"
    #: A landmark endpoint drawn on a plane other than the section's own.
    STALE_LANDMARK_PLANE = "STALE_LANDMARK_PLANE"
    #: A mask made on another orientation of the section.
    STALE_MASK = "STALE_MASK"
    #: Submit gate: a spacing off the strict interval, or a break reported.
    STRICT_INTERVAL = "STRICT_INTERVAL"
    #: More channels asked than one picture draws.
    TOO_MANY_CHANNELS = "TOO_MANY_CHANNELS"
    #: More landmark pictures asked than one call draws.
    TOO_MANY_LANDMARK_VIEWS = "TOO_MANY_LANDMARK_VIEWS"
    #: More sections than one call takes.
    TOO_MANY_SECTIONS = "TOO_MANY_SECTIONS"
    #: The section's last trace failed.
    TRACE_FAILED = "TRACE_FAILED"
    #: The section's trace is still running.
    TRACE_RUNNING = "TRACE_RUNNING"
    #: The section's trace was made at another placement.
    TRACE_STALE = "TRACE_STALE"
    #: The section's trace did not land in time.
    TRACE_TIMEOUT = "TRACE_TIMEOUT"
    #: An argument the verb does not take.
    UNKNOWN_ARGUMENTS = "UNKNOWN_ARGUMENTS"
    #: A section image the section does not hold.
    UNKNOWN_IMAGE = "UNKNOWN_IMAGE"
    #: A landmark pair id the section does not hold.
    UNKNOWN_LANDMARK = "UNKNOWN_LANDMARK"
    #: An atlas layer ``look`` does not draw.
    UNKNOWN_LAYER = "UNKNOWN_LAYER"
    #: A mask the section does not hold.
    UNKNOWN_MASK = "UNKNOWN_MASK"
    #: A ``look`` mode that does not exist.
    UNKNOWN_MODE = "UNKNOWN_MODE"
    #: A positioning part past the call's last picture.
    UNKNOWN_PART = "UNKNOWN_PART"
    #: A picture number the job never saved.
    UNKNOWN_PICTURE = "UNKNOWN_PICTURE"
    #: A plugin layer no section shows.
    UNKNOWN_PLUGIN_LAYER = "UNKNOWN_PLUGIN_LAYER"
    #: A region the atlas does not have.
    UNKNOWN_REGIONS = "UNKNOWN_REGIONS"
    #: A registration number the section does not hold.
    UNKNOWN_REGISTRATION = "UNKNOWN_REGISTRATION"
    #: A background run id the job does not have.
    UNKNOWN_RUN = "UNKNOWN_RUN"
    #: A picture's section is not in the stack.
    UNKNOWN_SECTION = "UNKNOWN_SECTION"
    #: A name that is no section's filename.
    UNKNOWN_SLICE_IDS = "UNKNOWN_SLICE_IDS"
    #: A verb that does not exist.
    UNKNOWN_VERB = "UNKNOWN_VERB"
    #: A saved picture's recipe cannot be read to redraw it.
    UNREADABLE_RECIPE = "UNREADABLE_RECIPE"
    #: A verb this job's settings do not include.
    VERB_OFF = "VERB_OFF"
    #: A plugin tool wrote a property it did not declare.
    WRITE_NOT_DECLARED = "WRITE_NOT_DECLARED"
    #: A side the section's placement does not have (left on a top/bottom).
    WRONG_SIDE_PAIR = "WRONG_SIDE_PAIR"


class Refused(ValueError):
    """A call refused, with nothing written.

    Parameters
    ----------
    code
        The reason, a :class:`RefusalCode` member (a plugin may give a code
        of its own).
    message
        What was attempted, what was found and what to do; it stands right
        after the code in the reply. A refusal whose facts come first gives
        ``message=`` among them instead, where it stands in the reply.
    **facts
        The plain facts behind it (the ids, the allowed values), in the
        reply's order. ``status`` among them is the reply's status word:
        ``error`` (the call could not run; the default) or ``refused`` (a
        gate: the call is valid but the job is not ready, as the opening
        pages' gate and the submit gates).
    """

    def __init__(
        self,
        code: str,
        message: str = "",
        /,
        **facts: Any,  # pyright: ignore[reportAny]  # a reply's JSON values
    ) -> None:
        told: dict[str, object] = {"message": message} if message else {}
        told.update(facts)
        status = told.pop("status", ReplyStatus.ERROR)
        self.code = code
        self.status = str(status)
        self.facts = told
        super().__init__(self.message or code)

    @property
    def message(self) -> str:
        """Return the refusal's message, empty when it has none."""
        text = self.facts.get("message")
        return text if isinstance(text, str) else ""

    @classmethod
    def from_exception(cls, error: BaseException) -> Refused:
        """Return the refusal an unexpected exception is answered with.

        A tool, an image-model call or a background job that crashes is
        answered like a refusal whose code is the exception's class name
        (``ValueError``, ``KeyError``): no :class:`RefusalCode` member, since
        the class is whatever was raised.

        Parameters
        ----------
        error
            The exception that ended the call.

        Returns
        -------
        Refused
            The class name as the code, the exception's text as the message.
        """
        return cls(type(error).__name__, message=str(error))

    def payload(self, *, section_id: str | None = None) -> dict[str, object]:
        """Return the reply every door answers with.

        Parameters
        ----------
        section_id
            The filename of the section a per-section answer's row is for;
            None for the reply to a whole call.

        Returns
        -------
        dict
            ``status``, ``error`` (the code), then the facts; a section's
            row has its ``id`` first.
        """
        reply: dict[str, object] = {"status": self.status, "error": self.code}
        reply.update(self.facts)
        if section_id is None:
            return reply
        return {"id": section_id, **reply}

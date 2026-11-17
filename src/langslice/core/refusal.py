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

    OK = "ok"
    ERROR = "error"
    REFUSED = "refused"


FAILED_STATUSES = (ReplyStatus.ERROR, ReplyStatus.REFUSED)


class RefusalCode(TextEnum):
    """Every refusal code, once, with its meaning."""

    ATLAS_ENDPOINT_OUTSIDE = "ATLAS_ENDPOINT_OUTSIDE"
    ATLAS_RENDER_FAILED = "ATLAS_RENDER_FAILED"
    BAD_ARGS = "BAD_ARGS"
    BAD_BOX = "BAD_BOX"
    BAD_CONFIG = "BAD_CONFIG"
    BAD_FIELD = "BAD_FIELD"
    BAD_JSON = "BAD_JSON"
    BAD_LANDMARK_PICTURE = "BAD_LANDMARK_PICTURE"
    BAD_LANDMARK_POINT = "BAD_LANDMARK_POINT"
    BAD_LAYER = "BAD_LAYER"
    BAD_MATRIX = "BAD_MATRIX"
    BAD_PATH = "BAD_PATH"
    BAD_PIVOT = "BAD_PIVOT"
    BAD_PLANE = "BAD_PLANE"
    BAD_REGION_PICTURE = "BAD_REGION_PICTURE"
    BAD_REGISTRATION = "BAD_REGISTRATION"
    BAD_ROTATION = "BAD_ROTATION"
    BAD_SETTINGS = "BAD_SETTINGS"
    BAD_WARP = "BAD_WARP"
    BORDER_TRACE_IO_ERROR = "BORDER_TRACE_IO_ERROR"
    DEFORMATION_UNREADABLE = "DEFORMATION_UNREADABLE"
    DUPLICATE_SLICE_IDS = "DUPLICATE_SLICE_IDS"
    EMPTY_BOX = "EMPTY_BOX"
    ENGINE_MISSING = "ENGINE_MISSING"
    ENV_FAILED = "ENV_FAILED"
    ERROR = "ERROR"
    FIT_FAILED = "FIT_FAILED"
    FROZEN = "FROZEN"
    HAS_DEFORMATION = "HAS_DEFORMATION"
    IMAGE_CALL_IN_PROGRESS = "IMAGE_CALL_IN_PROGRESS"
    INPUTS_CHANGED = "INPUTS_CHANGED"
    INSUFFICIENT_LANDMARKS = "INSUFFICIENT_LANDMARKS"
    INTERNAL = "INTERNAL"
    INTERVAL_BREAKS_UNSUPPORTED = "INTERVAL_BREAKS_UNSUPPORTED"
    INVALID_LINEAR_PLACEMENT = "INVALID_LINEAR_PLACEMENT"
    JOB_BUSY = "JOB_BUSY"
    JOB_UNREADABLE = "JOB_UNREADABLE"
    LANDMARK_TRIANGULATION_FAILED = "LANDMARK_TRIANGULATION_FAILED"
    LANDMARK_WARP_UNAVAILABLE = "LANDMARK_WARP_UNAVAILABLE"
    LAYER_NOT_DRAWABLE = "LAYER_NOT_DRAWABLE"
    MASK_UNSUPPORTED = "MASK_UNSUPPORTED"
    MCP_MISSING = "MCP_MISSING"
    MCP_TIMEOUT = "MCP_TIMEOUT"
    MCP_TOOL_ERROR = "MCP_TOOL_ERROR"
    MCP_UNAVAILABLE = "MCP_UNAVAILABLE"
    MISSING_ARGUMENTS = "MISSING_ARGUMENTS"
    MISSING_DEFORMATIONS = "MISSING_DEFORMATIONS"
    MISSING_POSITIONS = "MISSING_POSITIONS"
    MISSING_TRANSFORMS = "MISSING_TRANSFORMS"
    MIXED_IMAGES = "MIXED_IMAGES"
    NEEDS_STACK = "NEEDS_STACK"
    NO_ATLAS_IN_PICTURE = "NO_ATLAS_IN_PICTURE"
    NO_DEFORMATION = "NO_DEFORMATION"
    NO_FOLDER = "NO_FOLDER"
    NO_IMAGES = "NO_IMAGES"
    NO_JOB = "NO_JOB"
    NO_PICTURE = "NO_PICTURE"
    NO_POSITION = "NO_POSITION"
    NO_REGIONS = "NO_REGIONS"
    NO_SIDES = "NO_SIDES"
    NO_SORTING = "NO_SORTING"
    NO_STRUCTURES = "NO_STRUCTURES"
    NO_TISSUE = "NO_TISSUE"
    NO_TRACE = "NO_TRACE"
    NOT_A_GAP = "NOT_A_GAP"
    NOT_A_STACK = "NOT_A_STACK"
    NOT_FOR_SCRIPTS = "NOT_FOR_SCRIPTS"
    NO_SUCH_PATH = "NOT_FOUND"
    NOT_STARTED = "NOT_STARTED"
    NOT_WRITTEN = "NOT_WRITTEN"
    NOTHING_EXPORTED = "NOTHING_EXPORTED"
    NOTHING_FITTED = "NOTHING_FITTED"
    NOTHING_TO_REDO = "NOTHING_TO_REDO"
    NOTHING_TO_UNDO = "NOTHING_TO_UNDO"
    NOTHING_WRITTEN = "NOTHING_WRITTEN"
    OPENING_NOT_READ = "OPENING_NOT_READ"
    OPTION_OFF = "OPTION_OFF"
    PICTURE_NOT_SAVED = "PICTURE_NOT_SAVED"
    PLUGIN_FAILED = "PLUGIN_FAILED"
    POSITIONING_OFF = "POSITIONING_OFF"
    READ_TOOL_WRITES = "READ_TOOL_WRITES"
    RECORD_WRITE_FAILED = "RECORD_WRITE_FAILED"
    REGION_NOT_IN_PLANE = "REGION_NOT_IN_PLANE"
    REGION_TOO_SMALL = "REGION_TOO_SMALL"
    REGISTRATION_ELSEWHERE = "REGISTRATION_ELSEWHERE"
    RENDER_FAILED = "RENDER_FAILED"
    RUN_LOST = "RUN_LOST"
    SECTION_BUSY = "SECTION_BUSY"
    SECTION_GONE = "SECTION_GONE"
    SETTINGS_CHANGED = "SETTINGS_CHANGED"
    SIDES_AMBIGUOUS = "SIDES_AMBIGUOUS"
    STACK_TOO_LARGE = "STACK_TOO_LARGE"
    STALE_INPUT = "STALE_INPUT"
    STALE_LANDMARK_PLANE = "STALE_LANDMARK_PLANE"
    STALE_MASK = "STALE_MASK"
    STRICT_INTERVAL = "STRICT_INTERVAL"
    TOO_MANY_CHANNELS = "TOO_MANY_CHANNELS"
    TOO_MANY_LANDMARK_VIEWS = "TOO_MANY_LANDMARK_VIEWS"
    TOO_MANY_SECTIONS = "TOO_MANY_SECTIONS"
    TRACE_FAILED = "TRACE_FAILED"
    TRACE_RUNNING = "TRACE_RUNNING"
    TRACE_STALE = "TRACE_STALE"
    TRACE_TIMEOUT = "TRACE_TIMEOUT"
    UNKNOWN_ARGUMENTS = "UNKNOWN_ARGUMENTS"
    UNKNOWN_IMAGE = "UNKNOWN_IMAGE"
    UNKNOWN_LANDMARK = "UNKNOWN_LANDMARK"
    UNKNOWN_LAYER = "UNKNOWN_LAYER"
    UNKNOWN_MASK = "UNKNOWN_MASK"
    UNKNOWN_MODE = "UNKNOWN_MODE"
    UNKNOWN_PART = "UNKNOWN_PART"
    UNKNOWN_PICTURE = "UNKNOWN_PICTURE"
    UNKNOWN_PLUGIN_LAYER = "UNKNOWN_PLUGIN_LAYER"
    UNKNOWN_REGIONS = "UNKNOWN_REGIONS"
    UNKNOWN_REGISTRATION = "UNKNOWN_REGISTRATION"
    UNKNOWN_RUN = "UNKNOWN_RUN"
    UNKNOWN_SLICE_IDS = "UNKNOWN_SLICE_IDS"
    UNKNOWN_VERB = "UNKNOWN_VERB"
    UNREADABLE_RECIPE = "UNREADABLE_RECIPE"
    VERB_OFF = "VERB_OFF"
    WRITE_NOT_DECLARED = "WRITE_NOT_DECLARED"
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
        **facts: Any,  # pyright: ignore[reportAny]
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

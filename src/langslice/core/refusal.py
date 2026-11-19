"""The refusal: what LangSlice answers when it cannot do what it was asked.

A refusal is one exception, :class:`Refused`, carrying a
:class:`RefusalCode` (the machine-readable reason, the same string on every
door), a message saying what was attempted and what to do, and the plain
facts behind it (the ids, the allowed values). Nothing is written for a
refused call. The core, the job and the tools raise it; each door turns it
into its own answer once: the tool reply (:meth:`Refused.payload`), the
agent CLI's envelope, the MCP text block, the library's reply. A per-section
answer carries a refused section as its row, the other sections going on;
a crash is answered as a refusal with the code ``INTERNAL`` and the
exception's class name and text as its message.

The codes are strings the agent reads, so a member's value never changes.
A member's name is its value, so a reader who meets a code in a reply finds
it here with its meaning. The specific reason and what to do next are the
message's, at the place that raises.
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
    """Every refusal code, once, with its meaning.

    ``BAD_ARGUMENT``
        The call's arguments are wrong: a bad value or shape, an unknown mode,
        option or key, too many of something, a value out of range.
    ``NOT_FOUND``
        A named thing does not exist: a section, picture, region, image, mask,
        registration, layer, path, job or run.
    ``FROZEN``
        The job froze the property the call would change.
    ``BUSY``
        Something else holds what the call needs: a busy section, a job lock
        that timed out, background work still running.
    ``STALE_INPUT``
        What the call computed from changed underneath it, or a saved thing no
        longer matches the section.
    ``PRECONDITION``
        The job is not in the state the call needs: no transform yet, no
        deformation, a deformation already held, no trace, an opening not
        read, nothing to export, no sides on this plane.
    ``GATE_REFUSED``
        A submit gate refused: missing positions, transforms or deformations,
        a strict interval, interval breaks that are not supported.
    ``TOOL_OFF``
        The tool or verb is not offered in this job or at this door.
    ``ENGINE_UNAVAILABLE``
        An engine, its weights or the image model cannot be loaded or reached.
    ``FIT_FAILED``
        An engine ran and failed, or produced an unusable result.
    ``IO_ERROR``
        Reading or writing a file failed: a render, a record, a trace.
    ``INTERNAL``
        A fault in LangSlice or in a plugin, or an unexpected exception
        answered by the tool pipeline.
    """

    BAD_ARGUMENT = "BAD_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    FROZEN = "FROZEN"
    BUSY = "BUSY"
    STALE_INPUT = "STALE_INPUT"
    PRECONDITION = "PRECONDITION"
    GATE_REFUSED = "GATE_REFUSED"
    TOOL_OFF = "TOOL_OFF"
    ENGINE_UNAVAILABLE = "ENGINE_UNAVAILABLE"
    FIT_FAILED = "FIT_FAILED"
    IO_ERROR = "IO_ERROR"
    INTERNAL = "INTERNAL"


class Refused(Exception):
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
        **facts: Any,
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
        answered like a refusal with the code ``INTERNAL``: the exception's
        class (``ValueError``, ``KeyError``) is whatever was raised, so it
        leads the message.

        Parameters
        ----------
        error
            The exception that ended the call.

        Returns
        -------
        Refused
            ``INTERNAL``, the class name and the exception's text as the
            message.
        """
        return cls(RefusalCode.INTERNAL, message=f"{type(error).__name__}: {error}")

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

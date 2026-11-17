"""Run notes: free text saved with the results; the ``note`` tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langslice.core.refusal import RefusalCode, Refused

if TYPE_CHECKING:
    from langslice.job.job import Job
    from langslice.tools.run import Run


def add_note(job: Job, text: str) -> list[str]:
    """Append one stripped line to the run notes; the notes as they now stand.

    One undo step. An empty note is refused (``BAD_ARGS``).
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        raise Refused(
            RefusalCode.BAD_ARGS,
            "A note is a line of text; this one is empty, so "
            "nothing was written.",
        )
    before = job.snapshot()
    job.state.notes.append(cleaned)
    job.commit(before)
    return list(job.state.notes)


def note_tool(run: Run, text: str) -> dict[str, Any]:
    """Run the ``note`` tool: add the line, answer with every note.

    Parameters
    ----------
    run
        The run the tool is called in.
    text
        The declared argument as the model sent it
        (``tools/descriptions/bookkeeping.py``).

    Returns
    -------
    dict
        ``status`` and the run notes as they now stand.
    """
    return {"status": "ok", "notes": add_note(run.job, text)}

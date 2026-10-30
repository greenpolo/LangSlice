"""The bookkeeping tools' descriptions: notes, history, submit, exports.

Each tool's description is data, read as
:mod:`~langslice.tools.descriptions.description` says.
"""

from __future__ import annotations

from langslice.tools.arguments import SectionRegistration
from langslice.tools.descriptions.description import (
    REQUIRED,
    Argument,
    Offered,
    ToolDescription,
)

NOTE = ToolDescription(
    summary=(
        "Append one line to the run notes, which are saved with the results."
    ),
    arguments={
        "text": Argument(
            str,
            REQUIRED,
            "The note.",
        ),
    },
)

UNDO = ToolDescription(
    summary=(
        "Undo the last write. A tool call, or a finished piece of background "
        "work, is one step."
    ),
    arguments={
        "steps": Argument(
            int,
            1,
            """\
How many steps to undo at once, newest first (all the history
holds when it holds fewer).""",
        ),
    },
    returns="""\
The sections undone and the fields that changed, and the rows of
those sections.""",
)

REDO = ToolDescription(
    summary="Redo the write that ``undo`` reversed.",
    arguments={
        "steps": Argument(
            int,
            1,
            "How many undone steps to redo at once.",
        ),
    },
    returns="""\
The sections redone and the fields that changed, and the rows of
those sections.""",
)

RESTORE_REGISTRATION = ToolDescription(
    summary=(
        "Put back registrations sections held before: their orientation, "
        "transform and deformation."
    ),
    body=(
        """\
Each registration a section holds gets a number, kept whatever is
written after it: the `registration` of its status row and of the reply
to the write that left it. Undo takes back the job's last write; this
returns sections to any earlier registrations, such as fits made before
another method's, and leaves the other sections as they are. One
undoable write for the call. A deformation fitted with the section at
another position or other cutting angles is refused until the section is
back there.[[?look:section|look:overlay  look draws an earlier registration
without writing it.]]""",
    ),
    arguments={
        "section": Argument(
            str,
            "",
            """\
Filename (the extension may be left off) of a section whose
registrations to list; writes nothing. Give section or sections.""",
        ),
        "sections": Argument(
            list[SectionRegistration],
            [],
            """\
The registrations to put back: {id, registration} per
section, id a filename and registration a number from the list.""",
        ),
        "view": Argument(
            bool,
            True,
            """\
Return a picture of each section put back, under the
registration put back.""",
            offered=Offered.VIEW_CHOSEN,
        ),
    },
    returns="""\
With section, its registrations: per number the tool that made it and
when, its transform (kind, rotation_deg, scale_x, scale_y, shear,
translate_x_mm, translate_y_mm), its deformation (the method that made
it, every step's method, and the settings it was made with), the
numbers of the pictures saved with the write that left it, and which
one the section holds. With sections, per section the registration
it holds now and the one it replaced, or why it was not put back;
the rows of the sections written, and the pictures.""",
)

SUBMIT = ToolDescription(
    summary="End the run. Call this exactly once, last.",
    body=(
        """\
Waits for the background work still running. Refused, with the reason
and nothing changed, while something the run needs is missing.""",
    ),
    arguments={
        "summary": Argument(
            str,
            REQUIRED,
            "One or two sentences on what you did.",
        ),
        "notes": Argument(
            list[str],
            REQUIRED,
            "Short observations worth carrying forward.",
        ),
        "interval_breaks": Argument(
            list[str],
            REQUIRED,
            """\
Filenames (the extension may be left off) of the
sections AFTER a gap you conclude is real. Empty if there are
none.""",
        ),
    },
)

EXPORT_MAPS = ToolDescription(
    summary=(
        "Write each placed section's maps and the stack's exports, from the "
        "job as it stands."
    ),
    body=(
        """\
Per section (sections/<name>/): coords.tif (atlas micrometres per
pixel, float32, NaN outside the tissue or the atlas), labels.tif (atlas
ids, uint32), labels_fiji.tif + labels.csv (a uint16 index Fiji opens
without loss, and its table), residual.tif (the deformation, when one
is applied) and maps.json. Then exports/quicknii.json and
exports/visualign.json, and registration.json again. Only this writes
the per-section maps (submit writes the exports alone); nothing in the
state changes.""",
    ),
    arguments={
        "sections": Argument(
            list[str],
            [],
            """\
Filenames (the extension may be left off); empty is every
section.""",
        ),
        "full_resolution": Argument(
            bool,
            False,
            """\
Write the maps on the image file's own pixels
instead of its working copy (what every picture[[?zoom  but a zoom]] is
drawn from).""",
        ),
    },
    returns="""\
written (the sections), skipped (each with its reason), files
(path and kind), seconds.""",
)

DESCRIPTIONS: dict[str, ToolDescription] = {
    "note": NOTE,
    "undo": UNDO,
    "redo": REDO,
    "restore_registration": RESTORE_REGISTRATION,
    "submit": SUBMIT,
    "export_maps": EXPORT_MAPS,
}

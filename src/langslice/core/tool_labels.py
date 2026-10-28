"""Short plain-language names of the tools, their options and the tasks, for people.

The LangSlice window's prompt generator (``hosts/gui/generator.py``) shows
these beside each tool's own name. They are never model-facing: nothing a
model reads is made from this table (``tests/test_gui_generator.py`` checks
that no label appears in the pinned model-facing text and that only the
window imports it). A tool or option added to :data:`langslice.core.toolset.TOOLS`
needs a row here; the same test refuses a table that names one the toolset
does not have, so a renamed or removed tool cannot linger.
"""

from __future__ import annotations

#: One label per tool and per ``tool:option`` name, in toolset order.
TOOL_LABELS: dict[str, str] = {
    "look": "See sections and the atlas",
    "look:section": "pictures of a section alone",
    "look:atlas": "pictures of the atlas alone",
    "look:overlay": "atlas outlines drawn over a section",
    "look:positioning": "the whole series laid out along the atlas",
    "zoom": "Enlarge part of a picture",
    "make_image": "Make clearer versions of a section (contrast, channel mixes)",
    "grep_atlas": "Look up brain region names",
    "grep_atlas:points": "name the region under a point of a picture",
    "grep_atlas_view": "Show chosen regions' outlines in the atlas",
    "status": "Read the progress table",
    "list_files": "List the job folder's files",
    "search_files": "Search the job folder for words",
    "read_file": "Read one of the job folder's text files",
    "position_sections": "Place sections along the atlas, and tilt the cutting plane",
    "interactive_transform": "Move, turn and stretch a section by hand",
    "interactive_transform:flip": "mirror a section",
    "interactive_transform:rotate": "turn a section by quarter turns",
    "mark_damage": "Mark brain regions a section has lost",
    "make_mask": "Leave dirt, bubbles and folds out of the automatic fits",
    "elastix_affine": "Automatic alignment without bending (Elastix)",
    "fireants_syn": "Automatic warp (FireANTs SyN)",
    "fireants_syn:metric": "choose how the images are compared",
    "fireants_syn:edges": "also match the images' edges",
    "fireants_syn:detail": "choose how fine the warp is",
    "set_landmarks": "Pin matching points on a section and the atlas",
    "landmark_triangulation": "Warp a section by its pinned points",
    "trace_borders": "Image model draws the borders, then a warp to them",
    "note": "Write a line in the run notes",
    "undo": "Take back the last changes",
    "redo": "Put back changes taken back",
    "restore_registration": "Return a section to an earlier result",
    "submit": "Finish the run",
    "export_maps": "Write the per-section coordinate maps",
}

#: One label per task (:data:`langslice.core.spec.ALL_TASKS`).
TASK_LABELS: dict[str, str] = {
    "position": "Place along the atlas",
    "transform": "Align in the plane",
    "nonlinear": "Warp to fit",
}

#: What each task does, in a sentence for people.
TASK_ABOUT: dict[str, str] = {
    "position": "Put the sections in order and find where each one sits in the atlas.",
    "transform": "Turn, move and scale each section onto its atlas plane.",
    "nonlinear": "Bend the atlas onto each aligned section, region by region.",
}

#: The tool groups of :data:`langslice.core.toolset.GROUPS`, for people, and
#: the task each belongs to (None: every run).
GROUP_LABELS: dict[str, tuple[str, str | None]] = {
    "Common": ("Every run: looking, notes and undo", None),
    "Positioning": ("Place along the atlas", "position"),
    "Linear": ("Align in the plane", "transform"),
    "Nonlinear": ("Warp to fit", "nonlinear"),
}


def label(name: str) -> str:
    """The label of a tool or ``tool:option`` name; the name itself for one
    the table does not have (a plugin's tool)."""
    return TOOL_LABELS.get(name, name)

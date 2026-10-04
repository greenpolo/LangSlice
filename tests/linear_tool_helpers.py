"""Convenience adapters for testing one entry of the public batch tool."""

from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.arguments import ViewAuto

#: Keys that go into the call's ``view`` rather than the entry.
VIEW_KEYS = frozenset(ViewAuto.__annotations__)


def legacy_view(tool):
    """*tool* callable the old way: picture options as keywords or entry keys.

    For tests about what a picture shows, not about the argument shape: the
    keywords (and any picture key inside an ``entries`` dict) are moved into
    ``view`` before the real tool runs.
    """
    import functools
    import inspect

    takes_view = "view" in inspect.signature(tool).parameters

    @functools.wraps(tool)
    def call(*args, **kwargs):
        view = dict(kwargs.pop("view", None) or {})
        for key in list(kwargs):
            if key in VIEW_KEYS:
                view[key] = kwargs.pop(key)
        entries = kwargs.get("entries", args[0] if args else None)
        if isinstance(entries, list) and all(isinstance(e, dict) for e in entries):
            moved = [{k: v for k, v in e.items() if k not in VIEW_KEYS} for e in entries]
            for entry in entries:
                view.update({k: v for k, v in entry.items() if k in VIEW_KEYS})
            if "entries" in kwargs:
                kwargs["entries"] = moved
            else:
                args = (moved, *args[1:])
        if takes_view:
            kwargs["view"] = view
        return tool(*args, **kwargs)

    return call


def single_adjust(tool):
    """Call adjust_transforms with one entry and expose its row and images.

    Picture options given as keywords (or as the old positional mode/zoom/
    atlas_opacity/outlines) go into ``view``; the row carries the call's
    ``view`` echo. A refusal of the whole call comes back as it is.
    """
    def adjust(slice_id, *args, **kwargs):
        fields = ("rotation_deg", "scale_x", "scale_y", "translate_x_mm",
                  "translate_y_mm", "mode", "zoom", "atlas_opacity",
                  "pivot", "outlines", "note")
        values = {"id": slice_id, **dict(zip(fields, args, strict=False)), **kwargs}
        # The old positional slots: an opacity of 0 and an empty zoom were
        # "the default", and outlines "none" is now borders left out.
        if values.get("atlas_opacity") == 0.0:
            values.pop("atlas_opacity")
        if values.get("zoom") == []:
            values.pop("zoom")
        if values.get("outlines") == "none":
            values.pop("outlines")
            values.setdefault("atlas_channels", [])
        view = {key: values.pop(key) for key in list(values) if key in VIEW_KEYS}
        result = tool([values], view=view)
        if "results" not in result:
            return result
        row = dict(result["results"][0])
        if "view" in result:
            row["view"] = result["view"]
        media = result.get(TOOL_MEDIA_PARTS_KEY, [])
        row[TOOL_MEDIA_PARTS_KEY] = [media[i] for i in row.get("image_indexes", [])]
        return row
    return adjust

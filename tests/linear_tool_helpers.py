"""Convenience adapters for testing one entry of the public batch tool."""

from langslice.adk import TOOL_MEDIA_PARTS_KEY


def single_adjust(tool):
    """Call adjust_transforms with one entry and expose its row and images."""
    def adjust(slice_id, *args, **kwargs):
        fields = ("rotation_deg", "scale_x", "scale_y", "translate_x_mm",
                  "translate_y_mm", "mode", "zoom", "atlas_opacity",
                  "pivot", "outlines", "note")
        entry = {"id": slice_id, **dict(zip(fields, args, strict=False)), **kwargs}
        result = tool([entry])
        row = dict(result["results"][0])
        media = result.get(TOOL_MEDIA_PARTS_KEY, [])
        row[TOOL_MEDIA_PARTS_KEY] = [media[i] for i in row.get("image_indexes", [])]
        return row
    return adjust

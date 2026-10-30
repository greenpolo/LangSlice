"""The native agent tools: the toolbox and the picture limits of its model.

The tool door (:mod:`.toolbox`, which builds a run's tools, and
:mod:`.view_options`, the largest picture the run's model takes). The ADK
driver that runs them, and its message parts, are
:mod:`langslice.doors.agent`. The media keys (how the tools hand back
pictures) live in :mod:`langslice.core.pictures.media_keys`, so the OAuth
transport can read them too; they are re-exported here.
"""

from __future__ import annotations

from langslice.core.pictures.media_keys import (
    MEDIA_LAYOUT_ATTR,
    TOOL_MEDIA_PARTS_KEY,
)

__all__ = ["MEDIA_LAYOUT_ATTR", "TOOL_MEDIA_PARTS_KEY"]

"""The keys a tool's media travels under: plain strings, no imports.

Shared by the tool door (:mod:`langslice.doors.tools`, which re-exports them),
the ADK driver (:mod:`langslice.doors.agent`), the MCP door and the OAuth
transport (:mod:`langslice.providers.oauth.transport`), so it sits in the
lowest layer every one of them may import.
"""

from __future__ import annotations

TOOL_MEDIA_PARTS_KEY = "images"

MEDIA_LAYOUT_ATTR = "_langslice_media_layout"

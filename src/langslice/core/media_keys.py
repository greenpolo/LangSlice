"""The keys a tool's media travels under: plain strings, no imports.

Shared by the tool door (:mod:`langslice.doors.tools`, which re-exports
them), the ADK driver (:mod:`langslice.agent`), the MCP door and the OAuth
transport (:mod:`langslice.providers.openai_oauth`), so it sits in the
lowest layer every one of them may import.
"""

from __future__ import annotations

# Key under which LangSlice tools hand back media. The tools put their plain
# pictures (PIL images) and lines of text there, in order; each door packages
# them: the ADK agent through ``langslice.doors.tools.media.packaged`` (JPEG
# ``types.Part``s), the MCP server as content blocks.
#
# Since google-adk 2.7.0 a tool result may carry ``types.Part`` objects with
# ``inline_data``; ADK moves them into ``FunctionResponsePart``s on the
# function-response Event (persisted in session history, so the model keeps
# seeing them on later turns) and drops the now-empty key from the JSON the
# model reads. Media is only found one container deep -- a list of Parts
# directly under a dict key works, a list of lists does not -- so tools put
# their parts in a flat list under exactly this key. Text belongs in ordinary
# JSON fields: text Parts are not media and would leak into the JSON result.
TOOL_MEDIA_PARTS_KEY = "images"

# Opaque JSON field tying media to the tool call that produced it. Some ADK
# providers strip generated FunctionResponse ids from replayed history, while
# ordinary response JSON remains intact.
TOOL_MEDIA_DELIVERY_ID_KEY = "media_delivery_id"

# Private FunctionResponse attribute: original attachment count and surviving
# slot indices. Transports preserve their generated text when media is retired.
MEDIA_LAYOUT_ATTR = "_langslice_media_layout"

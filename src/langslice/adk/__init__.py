"""Google ADK integration: plugins, model resolution, and SDK helpers."""

from __future__ import annotations

# Key under which LangSlice tools hand back media to ADK.
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

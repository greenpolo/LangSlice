"""The native agent tools: the toolbox, its arguments, the ``view`` options, packaging.

The tool door (:mod:`.toolbox`, :mod:`.arguments`, :mod:`.view_options`),
the ADK message parts (:mod:`.media`) and the reply budget of a host that
caps one reply (:mod:`.reply`). The ADK driver that runs them is
:mod:`langslice.agent`.
The media keys (how the tools hand back pictures) live in
:mod:`langslice.core.media_keys`, so the OAuth transport can read them too;
they are re-exported here.
"""

from __future__ import annotations

from langslice.core.media_keys import (
    MEDIA_LAYOUT_ATTR,
    TOOL_MEDIA_DELIVERY_ID_KEY,
    TOOL_MEDIA_PARTS_KEY,
)

__all__ = ["MEDIA_LAYOUT_ATTR", "TOOL_MEDIA_DELIVERY_ID_KEY", "TOOL_MEDIA_PARTS_KEY"]

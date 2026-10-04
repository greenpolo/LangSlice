"""The core library's picture builders (layered refactor, phase 3b).

The core layer: plain inputs (a :class:`~langslice.linear.workspace.Workspace`,
the :class:`~langslice.linear.state.StackState`, ids, numbers, the core's
:class:`~langslice.linear.display.DisplayOptions`) in, plain PIL pictures
(captions burned in) and plain metadata out. A door turns the pictures into
what its host reads (ADK message parts, MCP image blocks, files).

The layer rule (``tests/test_core_imports.py``): modules here import other
core modules only (``space``, ``affine``, ``oblique``, ``image_prep``,
``atlas``, ``deformable`` and the core modules under ``linear``: ``render``,
``display``, ``workspace``, ``atlas_fetch``, ``deformation``, ``transform``,
``state``...). Never the job layer (``linear.job``), the operations
(``ops``), a door (``linear.toolbox``, ``linear.view_options``, ``adk``,
``mcp_server``) or ``google.*``, ``litellm``, ``openai``.

Modules: :mod:`.pictures` (the captioned section and atlas pictures the
viewing tools send, and the cached reference pictures), :mod:`.placement`
(a section on its physical canvas at a placement: every placement picture,
the staged interactive transform, and the frame each canvas is drawn in).
"""

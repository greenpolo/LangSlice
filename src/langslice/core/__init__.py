"""The core library: atlas, sections, stack state, renders and fits.

The lowest layer of the layered core (``CLAUDE.md`` in this package):
plain inputs (a :class:`~langslice.core.workspace.Workspace`, the
:class:`~langslice.core.state.StackState`, ids, numbers, the core's
:class:`~langslice.core.display.DisplayOptions`) in, plain PIL pictures
(captions burned in), numpy arrays and plain metadata out. A door turns the
pictures into what its host reads (ADK message parts, MCP image blocks,
files).

The layer rule (``pyproject.toml`` ``[tool.importlinter]``, run by
``tests/test_import_layers.py``; ``tests/test_core_imports.py`` checks what a
fresh interpreter loads): modules here import other core modules only. Never
the job layer (:mod:`langslice.job`), the operations (:mod:`langslice.ops`),
a door (:mod:`langslice.doors`), the agent driver (:mod:`langslice.agent`),
a host, a provider, or ``google.*``, ``litellm``, ``openai``.

Sub-packages: :mod:`.atlas` (BrainGlobe access), :mod:`.deformable` (the
deformable fit), :mod:`.nonlinear` (the image-model border route).
"""

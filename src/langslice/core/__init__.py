"""The core library: atlas, sections, stack state, renders and fits.

The lowest layer of the layered core (``CLAUDE.md`` in this package): plain
inputs (a :class:`~langslice.core.workspace.Workspace`, the
:class:`~langslice.core.stack.stack.Stack`, ids, numbers, the core's
:class:`~langslice.core.pictures.options.DisplayOptions`) in, plain PIL
pictures (captions burned in), numpy arrays and plain metadata out. A door
turns the pictures into what its host reads (ADK message parts, MCP image
blocks, files).

The layer rule (``pyproject.toml`` ``[tool.importlinter]``, run by
``tests/test_import_layers.py``): modules here import other core modules only.
Never the job layer (:mod:`langslice.job`), the tools
(:mod:`langslice.tools`), a door (:mod:`langslice.doors`), a host, a provider,
or ``google.*``, ``litellm``, ``openai``.

Sub-packages: :mod:`.atlas` (BrainGlobe access), :mod:`.sections` (section
files: channels, images, masks, damage, renders), :mod:`.stack` (the stack
state and its JSON), :mod:`.geometry` (affine maps, oblique plane sampling,
triangulation), :mod:`.fits` (the affine fit, deformations, landmarks, the
deformable fit and the image model's border trace), :mod:`.maps` (section
maps and the ABBA, ITK and imported forms), :mod:`.pictures` (every picture
drawn and its options). Single modules: :mod:`.workspace` (one job's sections
and atlas), :mod:`.space`, :mod:`.refusal`, :mod:`.plugins`,
:mod:`.connections`, :mod:`.failures`, :mod:`.hardware`, :mod:`.jsonable`,
:mod:`.outside`, :mod:`.shared_files`, :mod:`.tool_kinds` and
:mod:`.worker_threads`.
"""

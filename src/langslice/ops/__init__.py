"""The verbs: every write to a stack, as library functions on a job.

Each operation takes the :class:`~langslice.job.job.Job` (and, where it
reads the atlas or the section files, the core
:class:`~langslice.core.workspace.Workspace`) plus plain arguments, does
its write as ONE undo step through the job (``snapshot`` before, ``commit``
after: one step and the checkpoint), and returns a plain record of what
changed. No pictures, no wording for a model, no look-before-commit gates:
those belong to the doors (the agent tools, MCP, the CLI), which call these
and add them. A refusal is :class:`~langslice.ops.refusal.Refused`, carrying
its code and the plain facts behind it.

The layer rule (``tests/test_core_imports.py``): this package imports the
core and the job layer only, never a door (``linear.toolbox``,
``linear.view_options``, ``adk``, ``mcp_server``) and never an agent
framework or model client (``google.*``, ``litellm``, ``openai``).

Modules, one verb group each: :mod:`.positions` (positions and the stack's
cutting angles), :mod:`.order` (corrected order), :mod:`.orientation` (flip
and quarter turns), :mod:`.damage` (damage flags), :mod:`.appearance` (how
sections look for viewing and for fits), :mod:`.notes` (run notes),
:mod:`.transforms` (the in-plane transform: knobs to stored numbers, fit
records, the write), :mod:`.deformable` (the deformation on top of the
linear placement: fit, apply, or keep the linear placement).
"""

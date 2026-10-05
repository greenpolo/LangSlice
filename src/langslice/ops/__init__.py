"""The verbs: every write to a stack, and every read a viewing tool makes,
as library functions on a job.

Each operation takes the :class:`~langslice.job.job.Job` (and, where it
reads the atlas or the section files, the core
:class:`~langslice.core.workspace.Workspace`) plus plain arguments, does
its write as ONE undo step through the job (``snapshot`` before, ``commit``
after: one step and the checkpoint), and returns a plain record of what
changed. Pictures are the core's: a verb given display options returns the
pictures the core draws, never drawing its own. No wording for a model and
no look-before-commit gates: those belong to the doors (the agent tools,
MCP, the CLI), which call these and add them. A refusal is
:class:`~langslice.ops.refusal.Refused`, carrying its code and the plain
facts behind it.

The layer rule (``tests/test_core_imports.py``): this package imports the
core and the job layer only, never a door (``doors.tools.toolbox``,
``doors.tools.view_options``, ``doors.tools``, ``doors.mcp``) and never an agent
framework or model client (``google.*``, ``litellm``, ``openai``).

Modules: :mod:`.registry` (every verb and the operation it calls),
:mod:`.views` (the read verbs), :mod:`.atlas` (the region hierarchy),
:mod:`.positions` (positions, the cutting angles, the position search),
:mod:`.order` (corrected order), :mod:`.orientation` (flip and quarter
turns), :mod:`.damage` (damage flags), :mod:`.appearance` (how sections look
for viewing and for fits), :mod:`.notes` (run notes), :mod:`.history` (undo
and redo), :mod:`.transforms` (the in-plane transform: knobs to stored
numbers, fits, the write), :mod:`.deformable` (the deformation on top of
the linear placement: fit, apply, or keep the linear placement),
:mod:`.traces` (the image model's border traces), :mod:`.submit` (the
submit gates and the final write), :mod:`.exports` (the maps and exports),
:mod:`.inputs` (what a long verb re-checks before it applies),
:mod:`.refusal` (the refusal every verb raises).
"""

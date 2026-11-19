"""The tools: every write to a stack, and every read a viewing tool makes,
as library functions on a job.

Each operation takes the :class:`~langslice.job.job.Job` (and, where it reads
the atlas or the section files, the core
:class:`~langslice.core.workspace.Workspace`) plus plain arguments, does its
write as ONE undo step through the job (``snapshot`` before, ``commit`` after:
one step and the checkpoint), and returns a plain record of what changed.
Pictures are the core's: a tool given display options returns the pictures the
core draws, never drawing its own. No wording for a model: that belongs to the
doors (the agent tools, MCP, the CLI), which call these and add them. A refusal
is the core's :class:`~langslice.core.refusal.Refused`, carrying its code (a
:class:`~langslice.core.refusal.RefusalCode`) and the plain facts behind it.

The layer rule (``tests/test_import_layers.py``): this package imports the
core and the job layer only, never a door, a host or a provider, and never an
agent framework or model client (``google.*``, ``litellm``, ``openai``).

Sub-packages, one per kind of tool: :mod:`.looking` (pictures, the atlas, the
status, files, review), :mod:`.changing` (positions, transforms, warps, damage,
masks, section images, registrations), :mod:`.fitting` (the affine fit, the
deformation, the image model's border traces), :mod:`.landmarking` (landmark
pairs and their triangulated warp), :mod:`.bookkeeping` (notes, history,
submit, the exports) and :mod:`.descriptions` (the model-facing text of the
tools and the job statement).

Modules: :mod:`.catalog` (every tool and the operation it calls),
:mod:`.arguments` (the shapes of the arguments, and the check that refuses
strays), :mod:`.pipeline` (the steps every tool call goes through),
:mod:`.events` (each call's ``tool_start`` and ``tool_end``), :mod:`.refusal`
(the refusals the job's rules decide: an unknown, frozen or busy section),
:mod:`.reply` (the reply byte budget of a host that caps one reply),
:mod:`.run` (what a run's tool bodies share), :mod:`.stale` (what a long
operation's result depends on, checked before it applies).
"""

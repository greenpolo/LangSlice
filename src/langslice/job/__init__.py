"""The job layer's files: one job folder per stack, next to its images.

``<images>/langslice/`` holds everything LangSlice writes for a stack
(:mod:`langslice.job.layout`): ``job.json`` (settings and the folder's
format version), the state checkpoint, the undo history
(:mod:`langslice.job.history`), the per-section folders, the pictures the
model was shown with their layers (:mod:`langslice.job.views`), exports and
logs. Old layouts are upgraded on open (:mod:`langslice.job.migrate`); saved
host jobs are found by id through a small index (:mod:`langslice.job.index`).
The :class:`~langslice.job.job.Job` (:mod:`langslice.job.job`) owns the
state, undo, the checkpoint (:mod:`langslice.job.checkpoint`) and the submit
gates; :mod:`langslice.job.formats` and :mod:`langslice.job.quint` write the
public files and exports.

Layer: the job layer. It imports the core only, never an operation
(``langslice.ops``), a door or ``google.*`` (``tests/test_core_imports.py``).
"""

"""The engine service a host starts.

:mod:`.service` (``langslice serve --stdio``: the newline-delimited JSON
service the Fiji connector starts). The protocol models, the runtime
handlers, setup, saved Claude jobs and the linear snapshot worker are
door-level: :mod:`langslice.doors.api`. (The ``nonlinear.abba`` worker, whose
only caller was an unused method of the Fiji connector, was removed on
2026-10-04; ABBA's nonlinear work lands through the same linear worker, as
warp rows on top of the affine step.)
"""

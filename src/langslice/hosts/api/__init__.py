"""The engine service a host starts, and the ABBA-plugin worker it serves.

:mod:`.service` (``langslice serve --stdio``: the newline-delimited JSON
service the Fiji connector starts) and :mod:`.nonlinear_worker`
(``nonlinear.abba``: the image-model refinement of ABBA's placement, through
the ABBA plugin). The protocol models, the runtime handlers, setup, saved
Claude jobs and the linear snapshot worker are door-level:
:mod:`langslice.doors.api`.
"""

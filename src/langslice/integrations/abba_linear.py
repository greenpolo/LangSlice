"""Deprecated shim: the ABBA cutting-angle sign constants.

The Python live mirror that lived here (``hosts/integrations/abba_linear.py``)
was removed 2026-10-04; its sign table is :mod:`langslice.core.abba_angles`.
Kept for SliceBench (``slicebench/ingest/abba.py`` reads the two constants).
"""

from langslice.core.abba_angles import PITCH_TO_ROTATE_X_SIGN, YAW_TO_ROTATE_Y_SIGN

__all__ = ["PITCH_TO_ROTATE_X_SIGN", "YAW_TO_ROTATE_Y_SIGN"]

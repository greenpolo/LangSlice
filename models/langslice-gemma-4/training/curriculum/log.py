"""Backward-compatible shim — curriculum.log is now adaptive.curriculum.log."""
from adaptive.curriculum.log import *  # noqa: F401,F403
from adaptive.curriculum.log import CurriculumLogger  # noqa: F401

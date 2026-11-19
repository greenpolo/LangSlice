# STalign 2.0.0, vendored into LangSlice (GNU GPL v3; LICENSE in this folder).
#
# Upstream: https://github.com/JEFworks-Lab/STalign, the STalign/ package at commit
# 8a84bf1facc5173f7155400867a09c302125904e (main, 2026-10-16), Clifton et al. 2023,
# Nat Commun. Every Python file of the package is here as published, unchanged except for
# this header. Its optional C++ engine (STalign/_cpp, the compiled module _stalign_cpp) is
# not: without it STalign runs its pure-torch engine, which computes the same iterations
# (``backend_info()`` says which engine runs). LangSlice imports this package as
# ``langslice._vendor.stalign`` (``langslice/engines/stalign/engine.py``); its style and type
# checks do not read this folder.

__version__ = '2.0.0'

# The functions of the module STalign.STalign are also those of the package, so that
# `import STalign; STalign.LDDMM(...)` works with an installed STalign as it does when the
# tutorials import STalign.py from the source tree (their "OPTION B"), and as
# `from STalign import STalign; STalign.LDDMM(...)` does.
from .STalign import *  # noqa: E402,F401,F403

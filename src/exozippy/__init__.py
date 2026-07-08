import pytensor

# PyTensor 3 changed the "auto" linker default from a C-based linker to numba.
# None of our custom Ops (mulensing/op.py, orbit's exoplanet-core Kepler Op)
# register a numba implementation, so under the numba linker they still run
# their plain-Python perform() via numba's "object mode" fallback -- same
# execution path as the C-based linker, just with numba's JIT/caching
# overhead added on top for no benefit. Measured ~40% slower end-to-end with
# no speed upside. Restore the old default unless the user explicitly opted
# into numba (e.g. via PYTENSOR_FLAGS=linker=numba), which remains safe: our
# custom Ops are picklable (see VBMDirectMagOp.__getstate__) so numba's
# object-mode caching no longer crashes on them.
if pytensor.config.linker == "auto":
    pytensor.config.linker = "cvm"

from ._version import __version__
from .system import System

__all__ = [
    "__version__",
    "System"
]
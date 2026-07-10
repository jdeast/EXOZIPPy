import os

# Force single-threaded BLAS/OMP before numpy/pytensor/pymc/arviz/jax are
# imported anywhere -- this is the first line of code that runs for any
# entry into the package (CLI, tests, notebooks), since Python imports the
# parent package before any submodule. Threaded BLAS sizes its worker pool
# -- and the memory arena that goes with it -- to the visible core count, so
# on an HPC node this can cost several GB of virtual memory per process
# before a single array is allocated. samplers/ptde.py then forks one
# persistent worker per core, and each inherits that bloated footprint via
# copy-on-write, multiplying the per-process cost by the core count on top.
# Setting these after numpy et al. are already imported is a no-op -- the
# native thread pool is already sized by then -- so this must run first.
for _tvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "BLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_tvar, "1")

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
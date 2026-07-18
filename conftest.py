"""Repo-root conftest: pin BLAS/OpenMP thread counts before numpy imports.

This file is imported at pytest startup -- before any test module (and before
the first ``import numpy`` / ``import pytensor``), and freshly in every xdist
worker subprocess. That timing matters: OpenBLAS / MKL / OpenMP read these
environment variables once, when the native library first loads, so setting
them here is early enough and setting them later would be a no-op.

Why pin to 1: the suite runs ``-n 6`` (six worker processes). With the thread
vars unset, each worker's BLAS grabs *all* cores, so on a 36-core box that is
6 x 36 = 216 threads fighting over 36 cores -- a context-switch storm that,
stacked with six concurrent full-System builds, pushes a loaded machine into
swap and can freeze it for a long time. One BLAS thread per worker keeps the
core count matched to the worker count (6 busy cores, not 216 oversubscribed).
The math here is not BLAS-bound anyway -- the cost is pytensor graph compiles
and Python -- so single-threaded BLAS costs no measurable wall time.

``setdefault`` so an explicit override in the environment still wins (e.g. a
developer profiling BLAS scaling can export their own values).
"""
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

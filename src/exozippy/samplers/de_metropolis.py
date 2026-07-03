"""
Patched DEMetropolis / DEMetropolisZ subclasses.

PyMC bug (observed in 5.25.1): stats_dtypes_shapes declares 'scaling' and
'lambda' as scalar shape [], but np.atleast_1d inside the NDArray backend
always produces a 1-D array, causing a shape mismatch crash.

_fix_de_stats coerces those stats back to plain Python floats after each step.

Not currently wired into run.py — PTDE is the default non-HMC sampler. To
re-enable DEMC, import from here and add a dispatch branch in run.py.
See also: todo.txt — "DEMC is implemented but not well exercised"
"""

import numpy as np
import pymc as pm


def _fix_de_stats(astep_fn):
    def wrapper(self, q0):
        result, stats = astep_fn(self, q0)
        for s in stats:
            for key in ("scaling", "lambda"):
                if key in s and np.ndim(s[key]) > 0:
                    s[key] = float(np.ravel(s[key])[0])
        return result, stats
    return wrapper


class DEMetropolisZ(pm.DEMetropolisZ):
    astep = _fix_de_stats(pm.DEMetropolisZ.astep)


class DEMetropolis(pm.DEMetropolis):
    astep = _fix_de_stats(pm.DEMetropolis.astep)

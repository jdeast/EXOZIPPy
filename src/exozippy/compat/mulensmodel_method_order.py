"""Make MulensModel's magnification-method dispatch order deterministic.

``MagnificationCurve.methods_indices`` (MulensModel <= 3.11,
``magnificationcurve.py``) groups the epochs by magnification method like
this::

    self._methods_indices = {}
    methods = self.methods_for_epochs
    methods_ = np.array(methods)
    for method in set(methods):                 # <-- hash-ordered
        self._methods_indices[method] = (methods_ == method)

``methods`` is a list of method-name *strings* whose hashes are
PYTHONHASHSEED-randomized, so with more than one method in play the resulting
dict's order -- and with it the order in which
``_set_binary_lens_magnification_objects`` constructs the per-method backends
and ``get_binary_lens_magnification`` evaluates them -- flips from process to
process.

That would be harmless if the backends were pure functions of their own
epochs.  They are not: the VBMicrolensing solvers behind both ``VBM`` and the
binary ``point_source`` method carry state across calls, so evaluating the
same epochs *after a different first batch* moves the magnifications by up to
~3 ulp.  Measured on ``examples/ob161003`` (2 sources, 2 lens bodies, a
``VBM`` window inside a ``point_source`` baseline): reversing the dict order
alone -- at a fixed PYTHONHASHSEED, with byte-identical inputs -- changes 238
of 1716 magnifications, in *both* selections.

Downstream that is not cosmetic.  Those magnifications are the NNLS design
matrix of ``MulensInstrument._estimate_flux_components``, so the bootstrapped
``q_source``/``q_flux`` start values, and therefore the model's start-point
logp, took one of exactly two values per run:

    PYTHONHASHSEED=0,2,3 -> set(['point_source', 'VBM']) -> lp ...726189
    PYTHONHASHSEED=1,7   -> set(['VBM', 'point_source']) -> lp ...740741

Sorting the walk by method name fixes the order for good.  Name order, not
first-appearance order, so the answer does not depend on how the epochs
happen to be laid out either.  It changes nothing else: the selections are
disjoint boolean masks, so which one is filled first is not a modeling choice.

This is the same fix shape as the relaxation engine's ``free_symbols`` sort
(PR #57) -- iterate a hash-ordered container in a stated order, because a
float result downstream depends on it.

Retiring this: ``_is_affected`` reads the *source* and looks for the unsorted
set walk, so the patch disables itself the moment upstream sorts it (or
switches to ``dict.fromkeys``).  Delete the module once the MulensModel floor
in pyproject.toml is past that release.
"""

import inspect
import logging

import numpy as np

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_exozippy_method_order_patched"


def _sort_key(method):
    """Total order over method names, tolerating the ``None`` default method.

    ``methods_for_epochs`` fills unbracketed epochs with
    ``MagnificationCurve._default_method``, which is ``None`` unless a caller
    set one -- so a bare ``sorted()`` would raise on ``None`` vs ``str``.
    """
    return (method is None, str(method))


def _methods_indices(self):
    """Deterministic replacement for ``MagnificationCurve.methods_indices``.

    Byte-for-byte the upstream body, except the walk is
    ``sorted(set(methods), key=_sort_key)`` instead of ``set(methods)``.
    """
    if self._methods_indices is None:
        self._methods_indices = {}
        methods = self.methods_for_epochs
        methods_ = np.array(methods)

        # Sorted: see this module's docstring.  The order is load-bearing
        # because the VBMicrolensing backends are not order-independent --
        # do not "simplify" this back to `for method in set(methods)`.
        for method in sorted(set(methods), key=_sort_key):
            selection = methods_ == method
            self._methods_indices[method] = selection

    return self._methods_indices


def _is_affected(prop):
    """True when upstream still walks ``set(methods)`` unsorted.

    Reads the source rather than a version, so an upstream fix stops matching
    without anyone bumping a check here.  Source that cannot be read is
    treated as unaffected: leaving a working library alone beats patching
    blind.
    """
    fget = getattr(prop, "fget", None)
    if fget is None or getattr(fget, _PATCH_FLAG, False):
        return False
    try:
        src = inspect.getsource(fget)
    except (OSError, TypeError):
        return False
    return "set(methods)" in src and "sorted(set(methods)" not in src


def patch_mulensmodel_method_order():
    """Fix MulensModel's method-dispatch order.  Returns True if patched.

    Idempotent, and a no-op when MulensModel cannot be imported or when the
    unsorted walk is already gone.
    """
    try:
        from MulensModel.magnificationcurve import MagnificationCurve
    except Exception:  # pragma: no cover - optional dep
        return False

    prop = MagnificationCurve.__dict__.get("methods_indices")
    if not isinstance(prop, property) or not _is_affected(prop):
        return False

    setattr(_methods_indices, _PATCH_FLAG, True)
    MagnificationCurve.methods_indices = property(
        _methods_indices, doc=prop.__doc__
    )

    logger.debug(
        "Applied the MulensModel determinism patch "
        "(exozippy.compat.mulensmodel_method_order): magnification methods "
        "now dispatch in sorted name order."
    )
    return True

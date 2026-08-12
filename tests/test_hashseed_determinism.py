"""Two runs of identical code must build the identical model, bit for bit.

Python randomizes string hashes per process unless ``PYTHONHASHSEED`` is set,
so any ``set``/``frozenset`` of strings iterates in a different order in every
run.  That is fine right up until the order reaches a float, at which point a
fit's start values -- and its logp -- take a different value in every process.
It has bitten this codebase twice:

* PR #57: ``eq.free_symbols`` walks in the relaxation engine's (since-deleted)
  sympy scale passes scattered ``init_scale`` by orders of magnitude.
* This file: MulensModel's ``MagnificationCurve.methods_indices`` walks
  ``set(methods)``, so the per-method magnification backends are constructed
  and evaluated in hash order -- and VBMicrolensing is not order-independent,
  so the magnifications move by ~3 ulp.  Those magnifications are the NNLS
  design matrix behind ``MulensInstrument``'s bootstrapped ``q_source`` /
  ``q_flux`` start values, so ``examples/ob161003``'s start-point logp took
  one of exactly two values per run:

      PYTHONHASHSEED=0,2,3 -> 79356.587620672726189
      PYTHONHASHSEED=1,7   -> 79356.587620672740741

  Fixed by ``exozippy.compat.mulensmodel_method_order``.

Seeds 0 and 1 are not arbitrary: they are the pair that puts
``set(['point_source', 'VBM'])`` in opposite orders, i.e. the exact pair that
used to produce the two values above.

These run in subprocesses because ``PYTHONHASHSEED`` is read once at
interpreter startup -- ``monkeypatch.setenv`` cannot reproduce this in-process.
They deliberately exercise the magnification kernel rather than a whole
``System``: it is the same defect in ~2 seconds instead of ~2 minutes, and it
fails at the actual leak rather than 300 lines downstream of it.
"""

import os
import subprocess
import sys

import pytest

mm = pytest.importorskip("MulensModel")

# The two hash seeds that order set(['point_source', 'VBM']) oppositely.
SEED_A = "0"
SEED_B = "1"


def _run_under_seed(script, seed):
    """Run ``script`` in a fresh interpreter with PYTHONHASHSEED=seed.

    Returns its last stdout line.  Inherits the environment (so an
    editable/PYTHONPATH install of exozippy is importable) and overrides only
    the hash seed.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    # Thread-count noise would be a second, unrelated source of last-bit
    # differences; pin it so a failure can only mean hash ordering.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"subprocess with PYTHONHASHSEED={seed} failed:\n{proc.stderr[-3000:]}"
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"subprocess with PYTHONHASHSEED={seed} printed nothing"
    return lines[-1].strip()


# A binary lens with a finite-source (VBM) window inside a point-source
# baseline, i.e. TWO magnification methods -- the configuration
# MulensInstrument._binary_magnification_columns builds for every binary-lens
# flux bootstrap.  Geometry is ob161003's (Jung+2017 Table 1), on a synthetic
# time grid so the test needs no data files.
_BINARY_LENS_SETUP = """
import numpy as np
# Importing exozippy's MulensModel wrapper is what applies the determinism
# patch -- the test is as much about that wiring as about the patch itself.
import exozippy.components.mulensing.op  # noqa: F401
import MulensModel as mm

t_0, t_E, window = 2457551.038, 28.931, 3.0 * 28.931
t = np.linspace(t_0 - 5 * t_E, t_0 + 5 * t_E, 400)
inside = np.abs(t - t_0) < window
assert inside.any() and (~inside).any(), "need BOTH methods in play"

model = mm.Model({
    "t_0": t_0, "u_0": 0.059, "t_E": t_E,
    "s": 1.033, "q": 1.188, "alpha": 131.757, "rho": 4.51e-4,
})
model.set_magnification_methods([t_0 - window, "VBM", t_0 + window])
"""

_MAGNIFICATION_SCRIPT = (
    _BINARY_LENS_SETUP
    + """
a = np.asarray(model.get_magnification(t), dtype=float)
print(a.tobytes().hex())
"""
)

_DISPATCH_ORDER_SCRIPT = (
    _BINARY_LENS_SETUP
    + """
curve = model.get_magnification_curve(t, None, None)
curve.set_magnification_methods(
    [t_0 - window, "VBM", t_0 + window], "point_source"
)
methods = list(curve.methods_indices)
assert len(methods) == 2, methods
print(",".join(methods))
"""
)


def test_magnification_method_dispatch_order_is_hashseed_independent():
    """
    Given a binary-lens model with two magnification methods,
    When the per-method epoch selection is built in two processes whose
      PYTHONHASHSEED orders set(['point_source', 'VBM']) oppositely,
    Then the methods dispatch in the same order in both.

    This is the leak itself.  It is separated from the numerical test below
    because it says *why* the numbers moved, and it stays meaningful even if
    a future VBMicrolensing happens to be order-independent.
    """
    order_a = _run_under_seed(_DISPATCH_ORDER_SCRIPT, SEED_A)
    order_b = _run_under_seed(_DISPATCH_ORDER_SCRIPT, SEED_B)

    assert order_a == order_b, (
        "magnification methods dispatch in PYTHONHASHSEED order: "
        f"{order_a!r} vs {order_b!r}.  See "
        "exozippy/compat/mulensmodel_method_order.py."
    )


def test_binary_lens_magnification_is_hashseed_independent():
    """
    Given the binary-lens + finite-source geometry the microlensing flux
      bootstrap evaluates for every such event,
    When the magnifications are computed in two processes with different
      PYTHONHASHSEED values,
    Then every one of them is bit-for-bit identical.

    Compared as raw float64 bytes, not with a tolerance: the whole point is
    that a ~3 ulp drift here reaches the model's start values, and any
    tolerance loose enough to pass a broken build is loose enough to hide the
    next instance of this bug.
    """
    mags_a = _run_under_seed(_MAGNIFICATION_SCRIPT, SEED_A)
    mags_b = _run_under_seed(_MAGNIFICATION_SCRIPT, SEED_B)

    assert mags_a == mags_b, (
        "binary-lens magnifications differ between PYTHONHASHSEED=0 and =1 "
        "for byte-identical inputs; the model built from them (and its logp) "
        "is therefore not reproducible.  See "
        "exozippy/compat/mulensmodel_method_order.py."
    )


def test_compat_patch_is_idempotent_and_self_retiring():
    """
    Given the MulensModel determinism patch,
    When it is applied a second time,
    Then it reports that it did nothing -- and it would have reported the same
      had upstream already sorted the walk.

    ``compat``'s contract: detect the defect, not the version, and be safe to
    call from every site that touches the library.
    """
    from exozippy.compat import patch_mulensmodel_method_order

    # op.py applies it at import; either way, one more call settles it.
    patch_mulensmodel_method_order()

    assert patch_mulensmodel_method_order() is False

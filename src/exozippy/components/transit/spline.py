"""Fixed-knot cubic B-spline basis for a transit file's ``fitspline:`` key.

EXOFASTv2's ``fitspline`` (exofast_chi2v2.pro) re-fits Andrew Vanderburg's
``keplerspline`` to the residuals ``flux - modelflux + 1`` at every chi2 call
and multiplies the model by it, so the spline coefficients are PROFILED.
Under NUTS the same spline is co-fit by SAMPLING its coefficients instead:
the basis below depends only on the file's times, so it is built once here,
in numpy, and its columns join that file's block of the ordinary detrend
design matrix (``Instrument._build_block_detrend``), whose coefficients are
already sampled, whitened and plotted.

Provenance: ``keplerspline`` is Andrew Vanderburg's (Vanderburg & Johnson
2014), MIT licensed -- the IDL original at
https://github.com/avanderburg/keplerspline, which EXOFASTv2 bundles, and
his Python version at https://github.com/avanderburg/keplersplinev2.  Its
B-spline machinery (``bspline_bkpts.pro``, ``bspline_iterfit.pro``) is the
SDSS idlutils library's (pydl.pydlutils.bspline in Python).  This module is
an independent re-implementation, checked against a vendored copy of
keplersplinev2 in ``tests/third_party`` (``tests/test_transit_fitspline.py``:
the fitted curves agree to rounding).

What is ported, from EXOFASTv2's ``keplerspline/keplerspline.pro`` and
``bspline_bkpts.pro``:

* the gap split -- a new segment starts wherever ``diff(t) > splinespace``
  (keplerspline.pro uses ``ndays`` itself as the gap width), and every
  segment gets its own spline;
* the breakpoints -- per segment, time rescaled to [0, 1],
  ``nbkpts = long(range / bkspace) + 1`` (at least 2) evenly spaced
  breakpoints, plus ``nord - 1 = 3`` padding knots on each side at the same
  spacing; ``nord = 4``, i.e. a cubic.

What is deliberately NOT ported: the per-call 3-sigma clipping loop and the
BIC search over spacings.  The clip only decides which points the spline is
fit to (EXOFASTv2's chi2 still uses every point), it depends on the sampled
model, and a hard mask has no gradient; ``splinespace`` is a fixed input,
exactly as EXOFASTv2's ``ndays`` is.  See ``components/instrument.md``.
"""

import numpy as np
from scipy.interpolate import BSpline

SPLINE_ORDER = 4  # nord in bspline_iterfit: a cubic
DEFAULT_SPLINESPACE = 0.75  # days; mkss.pro's default


def split_segments(t, splinespace):
    """Index arrays of the gap-free segments of sorted times ``t``.

    keplerspline.pro: ``gaps = where(diff(t) gt ndays)`` -- strictly
    greater, so a gap of exactly ``splinespace`` does not split.
    """
    t = np.asarray(t, dtype=float)
    cuts = np.flatnonzero(np.diff(t) > splinespace) + 1
    return np.split(np.arange(t.size), cuts)


def segment_knots(n_bkpts):
    """Full knot vector, in the segment's rescaled [0, 1] time, for
    ``n_bkpts`` evenly spaced breakpoints (bspline_bkpts.pro)."""
    space = 1.0 / (n_bkpts - 1)
    inner = np.arange(n_bkpts) * space
    pad = space * np.arange(1, SPLINE_ORDER)
    return np.concatenate([inner[0] - pad[::-1], inner, inner[-1] + pad])


def segment_n_bkpts(span, splinespace):
    """bspline_bkpts.pro's breakpoint count for a segment spanning ``span``
    days: ``long(range / bkspace) + 1``, never fewer than 2.  Computed in
    the same rescaled units keplerspline.pro uses (``bksp = ndays / span``,
    range 1) so a span that is an exact multiple of the spacing truncates
    the same way.  ``span`` must be positive (``spline_basis`` raises on a
    zero-span segment before calling this)."""
    return max(int(1.0 / (splinespace / span)) + 1, 2)


def spline_basis(t, splinespace, label="spline"):
    """``(len(t), n_coeffs)`` cubic B-spline design matrix for sorted ``t``.

    One block of columns per gap-free segment, zero outside its segment;
    within a segment the columns sum to exactly 1 at every point (a B-spline
    partition of unity), so across the whole file they sum to 1 as well.
    That is the degeneracy with the file's ``baseline`` the caller must
    resolve -- see ``Transit._spline_columns``.

    A segment with no time span (a single point isolated by gaps, or
    repeated timestamps) RAISES, naming ``label``: keplerspline.pro divides
    by that zero span, and there is no spline to fit through one instant.
    """
    t = np.asarray(t, dtype=float)
    blocks = []
    for idx in split_segments(t, splinespace):
        seg = t[idx]
        span = float(seg[-1] - seg[0])
        if span <= 0.0:
            raise ValueError(
                f"[{label}] fitspline: the gap-free segment starting at "
                f"t={seg[0]!r} has {idx.size} point(s) and no time span, so "
                f"there is no spline to fit through it.  Gaps wider than "
                f"splinespace={splinespace!r} d start a new segment; widen "
                f"splinespace or mask the isolated point(s)."
            )
        x = (seg - seg[0]) / span
        knots = segment_knots(segment_n_bkpts(span, splinespace))
        blocks.append((idx, BSpline.design_matrix(x, knots, 3).toarray()))
    basis = np.zeros((t.size, sum(b.shape[1] for _, b in blocks)))
    c = 0
    for idx, b in blocks:
        basis[idx, c : c + b.shape[1]] = b
        c += b.shape[1]
    return basis

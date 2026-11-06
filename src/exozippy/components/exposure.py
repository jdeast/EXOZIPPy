"""Exposure smearing: the quadrature rules an exposure is averaged with.

A measurement is an exposure, not an instant, so what a data file records is
the time-mean of the model over each exposure.  ``Instrument.smear``
(components/instrument.py) evaluates a child's model on each row's
sub-exposure nodes and collapses them with these rules' weights; this module
holds only the rules, as pure numpy, so they can be tested and reused
without a model.

Every rule returns ``(offsets, weights)`` for ONE exposure of unit length
centered on 0: ``offsets`` in units of the exposure time (inside [-1/2,
1/2]) and ``weights`` summing to 1.  A row's sub-exposure times are then
``t + offsets * exptime``.

Units: the exposure time is in SECONDS everywhere in EXOZIPPy (the per-file
``exptime_s:`` key; ``components/instrument.md``, "Exposure smearing").
``SECONDS_PER_DAY`` is the one conversion to the model's days.
"""

import numpy as np

#: The one exposure-time conversion: the model's times are days, an
#: exposure time is seconds.
SECONDS_PER_DAY = 86400.0


def midpoint_rule(n):
    """The n-cell midpoint Riemann rule: n equal-width cells across the
    exposure, sampled at each cell's CENTER (offsets ``(j + 0.5)/n - 0.5``,
    never the exposure edges), weights ``1/n``.

    EXOFASTv2's grid (exofast_chi2v2.pro: ``dindgen(ninterp)/ninterp -
    (ninterp-1)/(2*ninterp)``), and the transit component's rule for parity
    with it.  Its error falls as ``n^-2`` for a smooth integrand -- far
    slower than ``gauss_legendre_rule`` -- but a transit light curve is not
    smooth (it has kinks at the contacts), which is where a Gauss rule's
    advantage shrinks.  With uniform ``1/n`` weights, edge samples would
    overweight the exposure boundaries and systematically over-smear, which
    is why the cell centers and not trapezoid/Simpson nodes are used.
    """
    n = int(n)
    j = np.arange(n)
    return (j + 0.5) / n - 0.5, np.full(n, 1.0 / n)


def gauss_legendre_rule(n):
    """The n-node Gauss-Legendre rule over one exposure: exact for every
    polynomial in time of degree ``2n - 1``.

    The same rule the raw-redshift RV path uses for a uniform-flux exposure
    (``rvinstrument/redshift.uniform_exposure_rule``, #565); three nodes
    are converged for a smooth model over an hour-long exposure
    (``rvinstrument.md``, "The exposure average").
    """
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * x, 0.5 * w


#: The rule family a child may declare as its ``EXPOSURE_RULE``.
RULES = {
    "midpoint": midpoint_rule,
    "gauss": gauss_legendre_rule,
}


def partition(exptime_days, ninterp, rule):
    """Partition rows into one group per node count, for ``Instrument.smear``.

    ``exptime_days`` -- ``(N,)`` each row's exposure time in DAYS;
    ``ninterp``      -- ``(N,)`` each row's node count (int >= 1);
    ``rule``         -- a key of ``RULES``.

    Returns a list of ``(rows, offsets, weights)``: ``rows`` the row indices
    in the group, ``offsets`` an ``(len(rows), k)`` array of sub-exposure
    time offsets in days (``None`` for the k = 1 group, whose lone node sits
    at the timestamp itself whatever the exposure time), ``weights`` the
    ``(k,)`` averaging weights.

    Rows are grouped by their OWN node count, never padded to the largest:
    each group's sub-exposure axis is exactly that group's k wide, so a row
    that needs one node is never evaluated at another file's larger count.
    Within a group the offsets scale with each row's own exposure time, so a
    group may mix files (and, per row, exposures) of different lengths.
    Groups come out in ``np.unique(ninterp)`` order, not row order; the
    caller gathers back into row order with the argsort of the concatenated
    group rows (``Instrument.smear`` does).
    """
    exptime_days = np.asarray(exptime_days, dtype=float)
    ninterp = np.asarray(ninterp, dtype=int)
    groups = []
    for kk in np.unique(ninterp):
        rows = np.nonzero(ninterp == kk)[0]
        if kk == 1:
            groups.append((rows, None, np.ones(1)))
        else:
            frac, weights = RULES[rule](kk)
            offsets = frac[None, :] * exptime_days[rows][:, None]
            groups.append((rows, offsets, weights))
    return groups

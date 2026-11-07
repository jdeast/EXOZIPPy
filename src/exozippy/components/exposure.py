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

    The uniform-flux rule of every exposure an ``exptime_s`` describes
    (#565 introduced it for raw redshifts); three nodes are converged for
    a smooth model over an hour-long exposure (``rvinstrument.md``, "The
    exposure average").
    """
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * x, 0.5 * w


#: Sub-exposure nodes per exposure (``ninterp:`` unstated) when the
#: exposure is described PER ROW (an ``exptime_s`` column or an
#: ``exposure_meter:`` series; ``Instrument.exposure_node_table``).  The
#: rule is Gaussian -- Gauss-Legendre for a uniform flux, the Gauss rule of
#: the exposure meter's own photon distribution otherwise -- so n nodes
#: integrate every polynomial of degree 2n - 1 in time exactly.  Three is
#: converged: against a 4001-point Simpson reference, a 60-min exposure at
#: hour angle 5 h is off by 1.3e-4 mm/s with 3 nodes (0.47 mm/s with 2, the
#: whole 1.2 m/s with 1), and a K = 100 km/s, P = 1 d binary on top of it by
#: the float64 time-resolution floor (~0.1 mm/s) at any n >= 3
#: (rvinstrument.md, "The exposure average").  A midpoint grid (the transit
#: component's rule) would need ~40 nodes for the same 60-min case:
#: transit light curves have kinks at contact, the redshift is smooth.
DEFAULT_ROW_NODES = 3


def moment_rule(t_rms, t_skew3=0.0):
    """``(offsets, weights)`` of the 2-node rule whose photon distribution
    has mean 0, RMS ``t_rms`` and third central moment ``t_skew3`` (s,
    s^3; offsets in s about the flux-weighted midpoint): exact for every
    cubic in time, i.e. for the ``v2 Var/2`` curvature term (Tronsgaard
    et al. 2019; ``v2`` the second time derivative of the velocity) and
    the ``v3 mu_3/6`` skew term.  With ``g = t_skew3 / t_rms^3`` the nodes
    are ``t_rms (g/2 +/- sqrt(1 + g^2/4))``.  What it cannot know is the
    4th moment; for a uniform exposure the rule's is 1.8x too small, which
    is 0.47 mm/s at 1 hour, 5 h from the meridian, scaling as T^4."""
    s = float(t_rms)
    g = float(t_skew3) / s**3
    root = np.sqrt(1.0 + 0.25 * g * g)
    xp, xm = s * (0.5 * g + root), s * (0.5 * g - root)
    return np.array([xm, xp]), np.array([xp, -xm]) / (xp - xm)


def weighted_exposure_rule(t, counts, n):
    """The n-node Gauss rule of an exposure meter's photon distribution.

    ``t`` -- (S,) the meter's sample times (days here);
    ``counts`` -- (S,) the photons (or any proportional number) in each
    sample.  The photon-weighted mean of a smooth f over the exposure is
    ``sum_j counts_j f(t_j) / sum_j counts_j``; the rule returned,
    ``(nodes, weights)`` with weights summing to 1, reproduces that sum
    EXACTLY for every polynomial f of degree <= 2n - 1 (Golub & Welsch
    1969 on the discrete measure, its recurrence by the Stieltjes
    procedure), with positive weights and every node inside the samples'
    span -- so a long, finely sampled meter series costs n model
    evaluations, not S.  A meter with no more than n lit samples is used
    as it is (its own samples are then the exact rule).
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(counts, dtype=float)
    keep = p > 0.0
    t, inv = np.unique(t[keep], return_inverse=True)
    p = np.bincount(inv, weights=p[keep])
    p = p / np.sum(p)
    if t.size <= n:
        return t, p
    c = float(np.sum(p * t))
    h = float(np.max(np.abs(t - c)))
    x = (t - c) / h
    alpha, beta = np.zeros(n), np.zeros(n)
    pi_prev, pi_k = np.zeros_like(x), np.ones_like(x)
    norm_prev = 1.0
    for k in range(n):
        norm = float(np.sum(p * pi_k**2))
        alpha[k] = float(np.sum(p * x * pi_k**2)) / norm
        beta[k] = norm / norm_prev if k else 0.0
        pi_prev, pi_k = pi_k, (x - alpha[k]) * pi_k - beta[k] * pi_prev
        norm_prev = norm
    off = np.sqrt(beta[1:])
    jac = np.diag(alpha) + np.diag(off, 1) + np.diag(off, -1)
    vals, vecs = np.linalg.eigh(jac)
    return c + h * vals, vecs[0] ** 2


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

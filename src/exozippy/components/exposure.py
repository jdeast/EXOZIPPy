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


# ---------------------------------------------------------------------------
# The automatic node count (review 8.5.11; instrument.md, "Choosing ninterp")
# ---------------------------------------------------------------------------

#: The per-file smearing tolerance: the aggregate chi2 the node-count error
#: may cost, ``sum_i (delta_i / sigma_i)^2`` over the file's rows, with
#: ``delta_i`` row i's smearing error and ``sigma_i`` its error bar at the
#: point checked (err_scale / jitter included).  Aggregate, not per point:
#: the bias is coherent (every in-transit point is biased the same way), so
#: 1000 points at 0.1 sigma each cost Delta chi2 ~ 10 (JDE ruling
#: 2026-11-05).
AUTO_DCHI2 = 0.1

#: The largest node count the automatic choice will pick.  The cost of a
#: smeared file's model is linear in n; a file still above ``AUTO_DCHI2``
#: here is given the cap and WARNED about, naming it.
AUTO_NINTERP_CAP = 32

#: The reference an n-node average is compared with: the same rule on
#: ``REFERENCE_FACTOR * n`` nodes.  For the midpoint rule's ``n^-2`` error
#: the difference recovers 15/16 of the n-node error; for a Gauss rule all
#: of it.  Relative to n rather than one fixed fine grid, so a file that
#: needs one node costs one plus four evaluations per row, not hundreds.
REFERENCE_FACTOR = 4

#: A user ``ninterp`` is called too FINE (a warning, never a change) only
#: when it is at least this many times the smallest n that meets
#: ``AUTO_DCHI2``: the cost is ~linear in n, and a 1.1x saving is noise.
FINE_WARN_FACTOR = 2


def smallest_ninterp(evaluate, lo, hi=None, cap=AUTO_NINTERP_CAP):
    """The smallest node count meeting the tolerance, per file.

    ``evaluate({file: n})`` -> ``{file: dchi2}`` evaluates every file of
    one trial at once (one compiled model evaluation per call, so the
    files' searches are batched).  ``lo[file]`` is a count known to FAIL
    (0 when none is), ``hi[file]`` one known to pass (absent when none
    is).  The search doubles from ``lo`` until a count passes (or ``cap``
    fails), then bisects, so on return the chosen n passed and n - 1 (when
    >= 1) was evaluated and failed.

    Returns ``{file: (n, dchi2_at_n, capped, n_needed)}``; ``capped`` is
    True when even ``cap`` fails (n is then ``cap``), and ``n_needed`` is
    then the count the file would need, ESTIMATED from the convergence of
    the two largest counts evaluated (``estimate_needed``; None when they
    do not converge) -- never built.  ``n_needed`` is n itself otherwise.
    A non-finite Delta chi2 counts as a failure.
    """
    lo = {f: int(v) for f, v in lo.items()}
    hi = {f: int(v) for f, v in (hi or {}).items() if f in lo}
    seen = {f: {} for f in lo}

    def run(trial):
        got = evaluate(dict(trial))
        for f, n in trial.items():
            seen[f][n] = float(got[f])

    def passes(f, n):
        return np.isfinite(seen[f][n]) and seen[f][n] < AUTO_DCHI2

    capped = set()
    # Doubling to bracket.
    while True:
        trial = {}
        for f in lo:
            if f in hi or f in capped:
                continue
            if lo[f] >= cap:
                capped.add(f)
                continue
            trial[f] = min(cap, max(1, 2 * lo[f]))
        if not trial:
            break
        run(trial)
        for f, n in trial.items():
            if passes(f, n):
                hi[f] = n
            else:
                lo[f] = n
    # Bisection inside each bracket.
    while True:
        trial = {f: (lo[f] + hi[f]) // 2 for f in hi if hi[f] - lo[f] > 1}
        if not trial:
            break
        run(trial)
        for f, n in trial.items():
            if passes(f, n):
                hi[f] = n
            else:
                lo[f] = n
    # A capped file needs two evaluated counts for its estimate: the top
    # one and half of it (one more evaluation when the search had none).
    for f in capped:
        top = max(list(seen[f]) + [lo[f]])
        for n in sorted({top, max(1, top // 2)} - set(seen[f])):
            run({f: n})
    out = {}
    for f in lo:
        if f in hi:
            n = hi[f]
            if n not in seen[f]:
                run({f: n})
            out[f] = (n, seen[f][n], False, n)
        else:
            top = max(seen[f])
            out[f] = (top, seen[f][top], True, estimate_needed(seen[f]))
    return out


def estimate_needed(seen, thr=None):
    """The node count a file would need, extrapolated from the two largest
    counts evaluated, ``{n: dchi2}``: with ``dchi2 ~ n^-p`` between them
    (p fitted, 4 for the midpoint rule's ``n^-2`` error on smooth data),
    the smallest n with ``dchi2 < thr``.  None when the two do not show
    convergence (p <= 0, or a non-finite or zero Delta chi2) -- an
    estimate for a warning, never a count anything is built at."""
    thr = AUTO_DCHI2 if thr is None else thr
    pts = sorted((n, d) for n, d in seen.items() if np.isfinite(d) and d > 0.0)
    if len(pts) < 2:
        return None
    (n1, d1), (n2, d2) = pts[-2], pts[-1]
    if n2 <= n1 or d2 >= d1:
        return None
    p = np.log(d1 / d2) / np.log(n2 / n1)
    return int(np.ceil(n2 * (d2 / thr) ** (1.0 / p)))

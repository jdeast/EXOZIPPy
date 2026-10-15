"""How much each observation tells us about an orbit's conjunction time.

Pure numpy, stage 3: these feed the choice of WHICH conjunction is sampled
(``Orbit._sampling_epochs``; orbit.md, "tc is SAMPLED near the data"), which
has to be made before the relaxation engine runs, from the data and the
user's seeds alone.

The quantity every function returns is the per-point FISHER INFORMATION on
the conjunction time, ``(d model / d tc)^2 / sigma^2``, in 1/day^2 -- the
same unit whatever the observable, which is what lets an RV, a transit and
an astrometric epoch be weighed against each other.  Why that is the right
weight: with the period and the conjunction time the only free parameters,
every point's model depends on them through the phase ``(t - tc) / P``, so
its gradient is ``g_i (1, E_i)`` with ``E_i = (t_i - tc) / P`` and the
Fisher matrix of ``(tc, P)`` is ``sum w_i [[1, E_i], [E_i, E_i^2]]`` with
``w_i = g_i^2 / sigma_i^2``.  Re-referencing the conjunction to epoch
``E0`` (``T = tc + E0 P``) shifts every ``E_i`` by ``-E0`` and zeroes the
off-diagonal term at ``E0 = sum w_i E_i / sum w_i``: the information-weighted
mean epoch is exactly the epoch at which the conjunction and the period are
UNCORRELATED -- EXOFASTv2's optimal epoch, before sampling.  What this
leaves out (correlations with the amplitude, the eccentricity, the transit
shape, a detrending baseline) is second order for the integer epoch.

Each estimate is deliberately cheap and each states its approximation: the
weights only have to be right to within a factor of a few to pick the right
integer, and the post-sampling optimum (``orbit.t0``) is measured exactly
regardless.
"""

import numpy as np

from .physics import mean_anomaly_at_conjunction

# Days per year and the AU in solar radii, for Kepler's third law in the
# units the seeds come in (solMass, days).
DAYS_PER_YEAR = 365.25
AU_PER_SOLRAD = 1.0 / 215.03215567054764


def true_anomaly(t, tc, period, ecc, omega):
    """True anomaly at times ``t`` of the orbit seeded by ``(tc, period, ecc,
    omega)`` (days, radians), by Newton iteration on Kepler's equation.

    ``tc`` is the time of primary conjunction, so the time of periastron is
    ``tc - M_c P / 2 pi`` with ``M_c`` the mean anomaly there (the same
    algebra ``physics.tc_from_tp`` inverts).
    """
    t = np.asarray(t, dtype=float)
    ecc = float(ecc)
    m_c = float(mean_anomaly_at_conjunction(ecc, omega))
    M = 2.0 * np.pi * (t - tc) / period + m_c
    M = np.mod(M, 2.0 * np.pi)
    E = M + ecc * np.sin(M)
    for _ in range(50):
        dE = (E - ecc * np.sin(E) - M) / (1.0 - ecc * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < 1e-12):
            break
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(0.5 * E),
        np.sqrt(1.0 - ecc) * np.cos(0.5 * E),
    )


def true_anomaly_rate(f, period, ecc):
    """``df/dt`` (rad/day) at true anomaly ``f``: ``(2 pi / P) (1 + e cos
    f)^2 / (1 - e^2)^(3/2)``.  With the eccentricity and omega held fixed, a
    shift of the conjunction time moves every phase by ``-df/dt dtc``."""
    return (
        2.0
        * np.pi
        / period
        * (1.0 + ecc * np.cos(f)) ** 2
        / (1.0 - ecc**2) ** 1.5
    )


def rv_shape(t, tc, period, ecc, omega):
    """The unit-amplitude Keplerian RV curve, ``cos(f + omega) + e cos
    omega``."""
    f = true_anomaly(t, tc, period, ecc, omega)
    return np.cos(f + omega) + ecc * np.cos(omega)


def rv_amplitudes_and_jitter(rvs, errors, shapes, n_iter=5):
    """Per orbit ``|K|`` and per file jitter, from the RVs of ONE star.

    ``rvs``/``errors`` are per file; ``shapes[file][orbit]`` is that file's
    unit-amplitude Keplerian (``rv_shape``) for each orbit of the star at the
    seeded elements.  A weighted linear fit of one offset per file plus one
    amplitude per orbit; each file's jitter is then the residual variance
    beyond its quoted errors, ``max(0, mean(r^2) - mean(err^2))``, and the
    fit is repeated with the errors so inflated.  A file with too few
    points to estimate its own scatter (three or fewer) gets no jitter.
    Returns ``(K per orbit, jitter per file)``, in the RVs' unit; where the
    fit is underdetermined (no more points than columns) the amplitudes are
    the zero-offset scatter ``sqrt(2) std`` per orbit and no jitter.
    """
    n_files = len(rvs)
    n_orbits = len(shapes[0])
    sizes = [len(v) for v in rvs]
    n = sum(sizes)
    jitter = np.zeros(n_files)
    rv = np.concatenate(rvs)
    err = np.concatenate(errors)
    X = np.zeros((n, n_files + n_orbits))
    row = 0
    for k in range(n_files):
        X[row : row + sizes[k], k] = 1.0
        for m in range(n_orbits):
            X[row : row + sizes[k], n_files + m] = shapes[k][m]
        row += sizes[k]
    if n <= X.shape[1]:
        resid = np.concatenate([v - np.mean(v) for v in rvs])
        return np.full(n_orbits, np.sqrt(2.0) * np.std(resid)), jitter
    for _ in range(n_iter):
        sig = np.sqrt(err**2 + np.repeat(jitter, sizes) ** 2)
        coef = np.linalg.lstsq(X / sig[:, None], rv / sig, rcond=None)[0]
        r = rv - X @ coef
        row = 0
        for k in range(n_files):
            rk = r[row : row + sizes[k]]
            ek = err[row : row + sizes[k]]
            if sizes[k] > 3:
                jitter[k] = np.sqrt(max(0.0, np.mean(rk**2) - np.mean(ek**2)))
            row += sizes[k]
    return np.abs(coef[n_files:]), jitter


def rv_epoch_information(times, sigmas, orbits, fit_ecc):
    """Per orbit ``(E*, information)``: the RV data's own optimal epoch and
    the information on the conjunction there, from the full linearized
    Fisher matrix of ONE star's RVs.

    ``times``/``sigmas`` are per file (``sigmas`` already including the
    jitter); ``orbits`` is one ``(tc, period, ecc, omega, K)`` per orbit of
    the star; ``fit_ecc`` says, per orbit, whether its eccentricity is free.
    The parameters are, per orbit, ``tc``, ``P``, ``K`` and -- where free --
    the sampled ``sqrt(e) cos(omega)``, ``sqrt(e) sin(omega)``, plus one
    offset per file.  The per-point ``rv_information`` is the ``(tc, P)``
    block alone, and its weighted mean epoch is the optimum only when
    nothing else correlates with them; on an eccentric orbit the
    eccentricity does.  Measured on the simulated RV-only HD 80606 example
    shipped before #430 (e = 0.93; numpyro 4 x (1000 + 1000)): the
    posterior's optimum is 28.3 periods from the seed, the per-point center
    2.3 and this one 30.9.
    Marginalizing the rest (inverting the full matrix) gives the ``(tc,
    P)`` covariance ``C``; the optimum is ``E* = -C_tP / C_PP`` periods from
    ``tc`` and the information there ``1 / (C_tt - C_tP^2 / C_PP)``.

    Its limit is a near-circular orbit with a free eccentricity, whose
    posterior in ``e`` piles against zero in a way no Fisher matrix
    represents: on `examples/kelt4`'s RVs (e ~ 0) it puts the optimum at
    -1184 periods where the posterior's is -1163 (corr(tc, P) -0.18 there),
    with the eccentricity held fixed it would be -1136 (+0.22).
    Derivatives are central differences of ``rv_shape``.
    """
    t = np.concatenate([np.asarray(x, float) for x in times])
    sig = np.concatenate([np.asarray(x, float) for x in sigmas])
    n_files = len(times)
    sizes = [len(x) for x in times]

    def model(params):
        v = np.zeros_like(t)
        for m, (tc, P, K, h, k) in enumerate(params):
            if fit_ecc[m]:
                e, w = h**2 + k**2, np.arctan2(k, h)
            else:
                # held at the seed: omega is meaningless at e = 0, so it
                # is not recovered from (h, k)
                e, w = orbits[m][2], orbits[m][3]
            v = v + K * rv_shape(t, tc, P, e, w)
        return v

    base = [
        [tc, P, K, np.sqrt(ecc) * np.cos(omega), np.sqrt(ecc) * np.sin(omega)]
        for tc, P, ecc, omega, K in orbits
    ]
    cols = []
    for m, (tc, P, ecc, omega, K) in enumerate(orbits):
        steps = [1e-4 * P, 1e-7 * P, 1e-3 * max(abs(K), 1e-12)]
        names = [0, 1, 2]
        if fit_ecc[m]:
            steps += [1e-4, 1e-4]
            names += [3, 4]
        for j, step in zip(names, steps):
            up = [list(b) for b in base]
            dn = [list(b) for b in base]
            up[m][j] += step
            dn[m][j] -= step
            cols.append((model(up) - model(dn)) / (2.0 * step))
    row = 0
    for k in range(n_files):
        c = np.zeros_like(t)
        c[row : row + sizes[k]] = 1.0
        cols.append(c)
        row += sizes[k]
    G = np.column_stack(cols) / sig[:, None]
    # Columns differ by many orders of magnitude (a period derivative is
    # ~(t - tc)/P times a conjunction one): equilibrate before inverting.
    scale = np.sqrt(np.sum(G**2, axis=0))
    scale[scale == 0] = 1.0
    Gs = G / scale
    C = np.linalg.pinv(Gs.T @ Gs) / np.outer(scale, scale)
    out = []
    col = 0
    for m in range(len(orbits)):
        ctt, ctp, cpp = C[col, col], C[col, col + 1], C[col + 1, col + 1]
        E = -ctp / cpp
        var = ctt - ctp**2 / cpp
        out.append((float(E), float(1.0 / var) if var > 0 else 0.0))
        col += 5 if fit_ecc[m] else 3
    return out


def rv_information(t, sigma, tc, period, ecc, omega, K):
    """Per-RV Fisher information on the conjunction time (1/day^2).

    ``v = K (cos(f + omega) + e cos omega)``, so ``dv/dtc = K sin(f + omega)
    df/dt``; for a circular orbit that is ``K (2 pi / P) sin(phase)`` -- an RV
    at quadrature, where the curve is flat, says nothing about the phase,
    and one at conjunction, where it is steepest, says the most.  ``K`` and
    ``sigma`` must share a unit.
    """
    f = true_anomaly(t, tc, period, ecc, omega)
    dvdt = K * np.sin(f + omega) * true_anomaly_rate(f, period, ecc)
    return (dvdt / np.asarray(sigma, dtype=float)) ** 2


def sky_speed_sq(t, tc, period, ecc, omega, cosi, amplitude):
    """Squared sky-plane speed (amplitude units per day, squared) of an
    orbit of angular semimajor axis ``amplitude``.

    In the orbital plane the velocity is ``(2 pi a / P) / sqrt(1 - e^2)``
    times ``(-(sin(f + w) + e sin w), cos(f + w) + e cos w)``; projecting
    onto the sky foreshortens the second component by ``cos i`` and then
    rotates by the node, which leaves the speed unchanged -- so the node is
    not needed.
    """
    f = true_anomaly(t, tc, period, ecc, omega)
    scale = 2.0 * np.pi * amplitude / period / np.sqrt(1.0 - ecc**2)
    vx = -scale * (np.sin(f + omega) + ecc * np.sin(omega))
    vy = scale * (np.cos(f + omega) + ecc * np.cos(omega)) * cosi
    return vx**2 + vy**2


def transit_shape(period, ar, p, cosi, ecc, omega):
    """Carter et al. (2008, ApJ 689, 499) eqs. (6), (7), (9), (10): the
    duration ``T`` between the midpoints of ingress and egress and the
    ingress duration ``tau``, both in days, for an orbit of scaled
    semimajor axis ``ar = a / R_*``, radius ratio ``p``, ``cos i`` and ``(e,
    omega)``:

        T0  = 2 tau_0 = (P / pi) (1 / ar) sqrt(1 - e^2) / (1 + e sin w)
        b   = ar cos i (1 - e^2) / (1 + e sin w)
        T   = T0 sqrt(1 - b^2),   tau = T0 p / sqrt(1 - b^2)

    Their small-planet, non-grazing approximation.  A seed that is grazing
    or does not transit at all (``b > 1 - p``) is evaluated at ``b = 1 -
    p``: the data either show a transit, which a slightly wrong seed must
    not be allowed to switch off, or they show none and time nothing --
    which ``transit_information`` then sees as points outside every
    window.
    """
    fac = np.sqrt(1.0 - ecc**2) / (1.0 + ecc * np.sin(omega))
    T0 = period / (np.pi * ar) * fac
    b = abs(ar * cosi * (1.0 - ecc**2) / (1.0 + ecc * np.sin(omega)))
    b = min(b, 1.0 - p)
    root = np.sqrt(1.0 - b**2)
    return T0 * root, T0 * p / root


def transit_information(t, sigma, tc, period, depth, T, tau, exptime=0.0):
    """Per-point Fisher information on the conjunction time of a transit
    (1/day^2): Carter et al. (2008) eq. (23), ``sigma_tc = Q^-1 T
    sqrt(theta / 2)`` with ``Q = sqrt(Gamma T) delta / sigma`` and ``theta
    = tau / T`` (their eq. 19), spread over the transit.

    A trapezoid of depth ``delta`` changes only during ingress and egress,
    with slope ``delta / tau``, so a point there carries ``(delta / tau)^2 /
    sigma^2`` and every other point none.  Summed over a uniformly sampled
    transit at cadence ``Gamma`` (``2 tau Gamma`` points in ingress and
    egress) that is ``2 Gamma delta^2 / (tau sigma^2)`` -- exactly Carter's
    ``sigma_tc^-2``.  Each point is given the AVERAGE of that over the
    transit, ``(delta / tau)^2 / sigma^2 * 2 tau / (T + tau)`` if it lies
    between first and fourth contact (``|t - t_n| < (T + tau) / 2`` from the
    nearest seeded conjunction ``t_n``) and zero otherwise, because a seed
    propagated across many periods is not accurate to the minutes that would
    say which points fell IN the ingress; it is accurate to the hours that
    say which fell in the transit.  A partial transit therefore counts in
    proportion to its in-transit points.

    An exposure longer than the ingress smears the ingress over it (Price &
    Rogers 2014, ApJ 794, 92), so ``max(tau, exptime)`` stands in for
    ``tau``.  ``sigma`` is in units of the fractional depth.
    """
    t = np.asarray(t, dtype=float)
    tau_eff = max(float(tau), float(exptime))
    phase = (t - tc) / period
    dt = (phase - np.round(phase)) * period
    inside = np.abs(dt) < 0.5 * (T + tau_eff)
    per_point = (depth / tau_eff) ** 2 * 2.0 * tau_eff / (T + tau_eff)
    return np.where(inside, per_point / np.asarray(sigma, float) ** 2, 0.0)


def error_scale(values, errors):
    """``s / median(errors)``, where ``s`` is the robust point-to-point
    scatter of a densely sampled series (``1.4826 MAD`` of the successive
    differences, over ``sqrt 2``): the factor by which the quoted errors
    misstate the noise, in either direction -- the fit's jitter term finds
    the same noise, and may shrink the errors as well as grow them
    (``Instrument._jitter_floor``).  A smooth signal (a transit, a trend)
    barely enters successive differences, so ``s`` measures the noise
    alone.  Not for sparse data (RVs), whose successive points are not
    close in phase.  Noiseless data, with no scatter to measure, keep their
    errors."""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return 1.0
    d = np.diff(values)
    s = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    if not s > 0:
        # noiseless (simulated) data: no scatter to measure the errors by
        return 1.0
    return float(s / np.median(errors))


def semimajor_axis_au(period, m_total):
    """Kepler's third law: ``a`` in AU for ``period`` in days and the total
    mass in solMass."""
    return (m_total * (np.asarray(period, float) / DAYS_PER_YEAR) ** 2) ** (
        1.0 / 3.0
    )

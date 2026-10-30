"""
Galactic prior: a mixture of the thin disk, thick disk and bulge/bar,
enforcing the density and kinematics of the Galaxy.

Provenance: this component derives from genulens (Koshimoto, Baba & Bennett
2021, ApJ 917, 78, "K21"; https://github.com/nkoshimoto/genulens).  Cite
K21 when you use it.  The model below is a simplified version of K21's;
the fidelity notes say where it departs.

Fidelity notes (vs genulens, reviewed 2026-08): the thin disk collapses
genulens's 7 age bins into one exponential layer with its B14disk-mode
(Bennett et al. 2014) kinematics instead of the Shu DF; the bar keeps a
plain-ellipsoid exp(-r_s/2) profile (not the K21 super-ellipsoid + X-shape)
but carries genulens's outer cylindrical cutoff and VVV-box-budget
normalization; NSD, stellar halo, and the bar's spatial sigma gradients /
streaming motion are omitted.  All three branch weights are genulens's
NUMBER-density channel on one absolute scale, so the logsumexp mixture
weight is physical.  The event-rate selection factor (theta_E * mu_rel) is
deliberately NOT here -- it lives in the lens component, so this prior
stays microlensing-agnostic.
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component

# One owner for "read a Parameter's hard support" -- star.py's volume prior
# normalizes over star.distance's bounds exactly the way the IMF branches
# below normalize over star.logmass's.
from exozippy.components.parameter import sampled_bounds as _sampled_bounds
from exozippy.constants import (
    BULGE_BAR_ANGLE,
    BULGE_CENTRAL_NUMBER_DENSITY,
    BULGE_DENSITY_X_0,
    BULGE_DENSITY_Y_0,
    BULGE_DENSITY_Z_0,
    BULGE_RC,
    BULGE_RC_WIDTH,
    BULGE_ROTATION_ANGULAR_VELOCITY,
    BULGE_VELOCITY_SIGMA_1,
    BULGE_VELOCITY_SIGMA_2,
    BULGE_VELOCITY_SIGMA_3,
    CHABRIER_BLEND_WIDTH_DEX,
    CHABRIER_HIGH_MASS_SLOPE,
    CHABRIER_LOG_MC,
    CHABRIER_MATCH_LOGMASS,
    CHABRIER_SIGMA,
    DISK_LOCAL_NUMBER_DENSITY,
    DISK_RDBREAK,
    DISK_ROTATION_VELOCITY,
    DISK_SCALE_HEIGHT,
    DISK_SCALE_LENGTH,
    DISK_VELOCITY_SIGMA_U,
    DISK_VELOCITY_SIGMA_V,
    DISK_VELOCITY_SIGMA_W,
    K_VEL_CONVERSION,
    SALPETER_IMF_SLOPE,
    SUN_GC_DISTANCE,
    THICK_DISK_LOCAL_NUMBER_DENSITY,
    THICK_DISK_ROTATION_VELOCITY,
    THICK_DISK_SCALE_HEIGHT,
    THICK_DISK_SCALE_LENGTH,
    THICK_DISK_VELOCITY_SIGMA_U,
    THICK_DISK_VELOCITY_SIGMA_V,
    THICK_DISK_VELOCITY_SIGMA_W,
)

logger = logging.getLogger(__name__)

#: defaults.yaml's ra/dec initvals (degrees): a star still sitting exactly here
#: with no user or data provenance was never positioned.
ANCHOR_PLACEHOLDER_RA_DEG = 180.0
ANCHOR_PLACEHOLDER_DEC_DEG = 0.0

# The frame, the line-of-sight basis and the position transform live in
# physics.py (the numpy layer) so the start-value hints in
# mulensing/lens.py use exactly the code this likelihood uses.
from .pdmf import GRID_DEX as PDMF_GRID_DEX  # noqa: E402
from .pdmf import PresentDayMassFunction, interp_table  # noqa: E402
from .physics import (  # noqa: E402
    GALACTOCENTRIC_FRAME,
    line_of_sight_basis,
    polar_velocity,
)
from .physics import (  # noqa: E402
    galactic_xyz as _galactic_xyz_np,
)


def _unnormalized_warning():
    logger.warning(
        "[galacticmodel] Cannot read finite log10-mass bounds; the IMF "
        "prior is left unnormalized (harmless for sampling, but its logp "
        "is offset by an unknown constant and is not comparable across "
        "IMF choices)."
    )
    return 0.0


def _power_law_log_norm(k, param):
    """log of the normalizer of p(x) ~ 10^(k x) over x in [lower, upper].

    ``param`` is the sampled log10-mass Parameter; its hard bounds are the
    support, so

        Z = int_lo^hi 10^(k x) dx = (10^(k hi) - 10^(k lo)) / (k ln10)

    which is finite for any k as long as both bounds are.  Evaluated as
    logdiffexp(., .) - log(|k| ln10) so the exponentials never overflow
    (10^(k*lo) is 1e12 already at the defaults.yaml floor of -9 dex).

    ``k`` may be a scalar (one slope shared by every star, as Salpeter is) or
    a per-star array (the FFP slope is per star and user-settable, and each
    star carries its own bounds).  Both branches of the k == 0 split are
    always evaluated and then selected, which is safe because neither can
    produce a NaN: upper > lower always, and the power-law branch substitutes
    k = 1 where k is zero.
    """
    bounds = _sampled_bounds(param)
    if bounds is None:
        return _unnormalized_warning()
    lower, upper = bounds

    ln10 = np.log(10.0)
    k = np.asarray(k, dtype=float)
    is_flat = k == 0.0  # uniform in log10 M
    k_safe = np.where(is_flat, 1.0, k)

    a, b = k_safe * upper * ln10, k_safe * lower * ln10
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    power_law = hi + np.log1p(-np.exp(lo - hi)) - np.log(np.abs(k_safe) * ln10)
    return np.where(is_flat, np.log(upper - lower), power_law)


# Logistic scale of the Chabrier blend.  A logistic w(t) = sigmoid(-t) with
# t = (x - x_match)/s falls from 0.9 to 0.1 over 2 ln(9) s, so the declared
# 10%-90% width fixes s.
_CHABRIER_BLEND_SCALE = CHABRIER_BLEND_WIDTH_DEX / (2.0 * np.log(9.0))

# Quadrature points for the blended normalizer.  The blend has no closed-form
# integral (see chabrier_logmass_logp), so the constant is a 1-D numerical
# integral over the support -- computed ONCE, in numpy, at build time, which
# is legitimate exactly because star.logmass's bounds are STATIC.  If a
# dynamic (linked) bound is ever wanted on logmass this breaks and has to be
# rethought, the same caveat parameter.py's static-bound normalizers carry.
#
# 1e5 intervals put dx ~ 1.2e-4 dex on the default 11.5 dex support, ~380x
# finer than the 0.045 dex blend scale, and trapezoid error falls as dx^2:
# measured against 4x the points the log-normalizer moves by < 1e-12 nats.
_CHABRIER_QUAD_POINTS = 100001


def _softplus(t, xp):
    """log(1 + exp(t)), stable at both extremes and smooth at t = 0.

    Deliberately NOT hand-rolled as ``maximum(t, 0) + log1p(exp(-abs(t)))``:
    that form is stable but its derivative at exactly t = 0 comes out of
    ``maximum``'s tie-break rather than as the true 0.5, and t = 0 is exactly
    where the Chabrier segments meet -- i.e. a gradient artifact planted at
    the one point this whole change exists to smooth.  ``pt.softplus`` gives
    0.5 there (verified), and ``np.logaddexp(0, t)`` is numpy's own stable
    formulation of the same function.
    """
    if hasattr(xp, "softplus"):
        return xp.softplus(t)
    return xp.logaddexp(0.0, t)


def chabrier_logmass_logp(logmass, param, xp=pt):
    """Chabrier (2003) system IMF as a density in x = log10(M/Msun).

    THE functional form, in one place, for both backends: ``xp=pt`` builds the
    symbolic potential, ``xp=np`` is what the normalizer integrates.  Returns
    the NORMALIZED log density; pass ``param=None`` for the unnormalized one.

    Functional form
    ---------------
    Chabrier's system IMF is PIECEWISE (PASP 115, 763, Table 1): a lognormal
    below 1 Msun and dN/dlog m ~ m^-1.3 above it.  Until 2026-08 only the
    lognormal was implemented, applied over the whole support, under a comment
    saying one "would smoothly match it to a Salpeter tail" -- in the
    conditional, describing what one could do.  Review 3.7.1 is that nobody
    had.  The lognormal's slope in log mass is -(x - log Mc)/sigma^2, which is
    -2.02 nats/dex at 1 Msun and -5.10 at 10, against the tail's constant
    -1.3*ln10 = -2.99, so massive stars were over-penalized and the error grew
    with mass.

    Change of variables
    -------------------
    None.  Both segments are quoted as dN/dlog m, and x = log10(M) is the
    sampled coordinate, so they drop straight in.  Contrast the Salpeter
    branch in ``build_likelihood``, quoted as dN/dM, which picks up
    |dM/dx| = M ln10 and turns its exponent from -alpha into (1 - alpha).

    The match, and why it is SMOOTHED
    ---------------------------------
    The tail's amplitude is fixed by continuity at the match point, so the
    two segments agree in value there -- but the true IMF is only C0: the
    slope jumps by 0.97 nats/dex, which is a logp KINK, the same class of
    thing as the SHO kernel's Q = 1/2 switch.  It is blended out over
    ``CHABRIER_BLEND_WIDTH_DEX``, whose comment carries the arithmetic: the
    blend costs ~0.02 nats against ~0.7 nats/dex of published uncertainty in
    the tail's own exponent, so it cannot be the limiting error.

    The blend is a WEIGHTED SUM of the two DENSITIES,
    ``w(x)*lognormal + (1-w(x))*powerlaw``, never a ``pt.where`` over the two
    branch log-densities.  Both branches are finite everywhere here, so a
    where would not actually produce the documented NaN -- but the weighted
    sum has no branch selection at all, so it stays clear of that trap by
    construction rather than by argument.  It is evaluated as
    ``logaddexp(log w + L, log(1-w) + P)``, which is that sum exactly, in the
    form that does not underflow: at the -9 dex floor the lognormal is
    exp(-107) while the unweighted tail is ~1e11.

    Normalization
    -------------
    The blended density has NO closed-form integral, so the analytic
    truncated-lognormal constant becomes a numerical quadrature over the
    parameter's own hard support.  That is the real implementation cost of
    the tail, and it is cheap only because the bounds are static -- see
    ``_CHABRIER_QUAD_POINTS``.  Normalizing over the same support the
    Salpeter and FFP branches use is what keeps switching between them a
    meaningful move in logp rather than an arbitrary offset.
    """
    x = logmass
    lognormal = -0.5 * ((x - CHABRIER_LOG_MC) / CHABRIER_SIGMA) ** 2
    # Continuity fixes the tail's amplitude: it equals the lognormal AT the
    # match point.  (Chabrier quotes the two normalizations separately; using
    # them would be equivalent up to the additive constant the normalizer
    # removes anyway, and stating it as continuity makes the C0 property
    # manifest instead of a coincidence of two rounded numbers.)
    at_match = (
        -0.5
        * ((CHABRIER_MATCH_LOGMASS - CHABRIER_LOG_MC) / CHABRIER_SIGMA) ** 2
    )
    power_law = at_match - CHABRIER_HIGH_MASS_SLOPE * np.log(10.0) * (
        x - CHABRIER_MATCH_LOGMASS
    )

    # w = sigmoid(-t) is 1 well below the match and 0 well above it, so
    # log w = -softplus(t) and log(1 - w) = -softplus(-t).
    t = (x - CHABRIER_MATCH_LOGMASS) / _CHABRIER_BLEND_SCALE
    blended = xp.logaddexp(
        lognormal - _softplus(t, xp), power_law - _softplus(-t, xp)
    )
    if param is None:
        return blended
    return blended - _chabrier_log_norm(param)


def _chabrier_log_norm(param):
    """log Z of the blended Chabrier density over the Parameter's support.

    Trapezoid on a dense uniform grid, per element (each star carries its own
    bounds), with the peak factored out so the exponentials stay in range --
    the integrand spans exp(-107) at the -9 dex floor.  Deliberately the same
    contract as ``_power_law_log_norm``: an unreadable or degenerate support
    falls back to unnormalized with a warning rather than failing a fit,
    because a constant offset never changes the sampling.

    This REPLACED an analytic truncated-lognormal constant
    (``_lognormal_log_norm``, deleted 2026-08 with the tail), whose hard-won
    detail is worth keeping in mind here: it built the truncated mass from
    erf/erfc with a per-side branch and NEVER as a difference of Phi CDFs,
    because this support straddles the Chabrier mean at -14.6 and +5.5 sigma,
    where ``1 - eps`` minus ``1 - eps'`` throws away nearly every significant
    digit.  Quadrature has no such subtraction anywhere -- it sums positive
    numbers -- so the cancellation cannot recur, but a future author tempted
    to restore a closed form for speed needs that warning.  The analytic form
    survives in ``tests/test_galactic_model.py`` as the independent check on
    a below-the-match support, which is where it is still exact.
    """
    bounds = _sampled_bounds(param)
    if bounds is None:
        return _unnormalized_warning()
    lower, upper = bounds

    n = max(np.size(lower), np.size(upper))
    lower = np.broadcast_to(np.atleast_1d(lower), (n,))
    upper = np.broadcast_to(np.atleast_1d(upper), (n,))

    frac = np.linspace(0.0, 1.0, _CHABRIER_QUAD_POINTS)
    grid = lower[:, None] + (upper - lower)[:, None] * frac[None, :]
    f = chabrier_logmass_logp(grid, param=None, xp=np)
    peak = np.max(f, axis=1, keepdims=True)
    integral = np.trapezoid(np.exp(f - peak), grid, axis=1)

    if not np.all(np.isfinite(integral)) or np.any(integral <= 0.0):
        # Support so far into a tail that its probability mass underflows.
        return _unnormalized_warning()
    return np.log(integral) + peak[:, 0]


def ffp_logmass_logp(logmass, alpha, param):
    """Free-floating-planet mass prior, as a density in x = log10(M/Msun).

    THIS IS THE SEAM.  If you have your own free-floating-planet mass
    function, replace the two lines below the comment banner and nothing else:
    every caller (``GalacticModel.build_likelihood``), the per-star selection
    (``star:``'s ``mass_function: ffp``), the floor, and the tests reach the
    functional form only through here.

    Functional form
    ---------------
    Sumi et al. 2023 (AJ 166, 108; arXiv:2303.08280), MOA-II 9-year survey:

        dN/dlog M = Z * (M / M_norm)^(-alpha)  dex^-1 star^-1

    with alpha = 0.96 (+0.47/-0.27), Z = 2.18 (+0.52/-1.40) dex^-1 star^-1 and
    M_norm = 8 M_Earth, fit over 0.33 < M/M_Earth < 6660 (1e-6 < M/Msun <
    0.02).  ``alpha`` here is that exponent, POSITIVE for the observed
    rising-toward-low-mass slope; it is per star and user-settable (``star:``
    key ``mass_function: {kind: ffp, alpha: ...}``) because this measurement
    is uncertain and Roman will revise it.

    Change of variables
    -------------------
    None -- and that is the whole point of preferring this parameterization.
    The measurement is ALREADY a density in log mass, and x = log10(M) is
    already the sampled coordinate (``star.logmass``), so

        log p(x) = -alpha * ln10 * x + const,

    with no Jacobian.  Contrast the Salpeter branch in ``build_likelihood``,
    which is quoted as dN/dM and therefore picks up |dM/dx| = M ln10, turning
    its exponent from -alpha into (1 - alpha).  If you substitute a mass
    function quoted as dN/dM, you must put that Jacobian back.

    Normalization
    -------------
    From the sampled support, not from the paper.  ``param`` is the
    ``star.logmass`` Parameter; ``_power_law_log_norm`` integrates 10^(k x)
    over its hard [lower, upper] bounds, which makes this a proper density
    directly comparable with the IMF branches over the same support.  Both Z
    and M_norm are therefore unusable here: each contributes only an additive
    constant that the normalizer cancels exactly.  Z is an abundance (a rate
    per star), and would matter only to a model that weighed the FFP and
    stellar lens populations against each other -- this prior conditions on
    the lens being an FFP, so it cannot see a rate.

    The support's lower end is a real modeling choice -- a rising density
    piles prior mass against it, 90% of it within 1/alpha dex -- and it is
    deliberately left to the user, as ``star.<name>.logmass``'s ordinary
    ``lower`` bound.  Nothing clamps it: sub-stellar masses are this
    relation's domain, so a floor imposed here would gate exactly the models
    it exists to express.  ``Star._warn_ffp_logmass_bound`` says so out loud
    when the bound is still the default.
    """
    # ---- substitute your own mass function here -------------------------
    # k is the exponent of the density in the SAMPLED coordinate, p(x) ~
    # 10^(k x); the normalizer is the integral of that over the support.
    k = -np.asarray(alpha, dtype=float)
    return k * np.log(10.0) * logmass - _power_law_log_norm(k, param)
    # ---------------------------------------------------------------------


# ----------------------------------------------------------------------
# The kinematic prior's pieces, one owner each.
#
# The density of the three branches is shared by the PyMC graph and by the
# numpy position normalizer (log_position_norm), so the constant cannot drift
# from the density it normalizes.  The linked-star analysis
# (linked_star_structure) is static -- it reads which star.* elements are
# HARD-linked -- and decides which stars carry a position term of their own.
# The rulings are in components/star/star.md, "Linked stars: one position,
# one density term".
# ----------------------------------------------------------------------

_LOG_2PI = float(np.log(2.0 * np.pi))
#: log of the rv column of the velocity Jacobian: the Gaussians are in km/s,
#: the sampled rv is in m/s.
_LOG_KMS_PER_MS = float(np.log(1e-3))

#: Branch order of every per-branch list below (and of the logsumexp).
BRANCHES = ("thin_disk", "thick_disk", "bulge")

#: (v_r, v_phi, v_z) dispersions in km/s, per branch.
BRANCH_SIGMAS = {
    "thin_disk": (
        DISK_VELOCITY_SIGMA_U,
        DISK_VELOCITY_SIGMA_V,
        DISK_VELOCITY_SIGMA_W,
    ),
    "thick_disk": (
        THICK_DISK_VELOCITY_SIGMA_U,
        THICK_DISK_VELOCITY_SIGMA_V,
        THICK_DISK_VELOCITY_SIGMA_W,
    ),
    "bulge": (
        BULGE_VELOCITY_SIGMA_1,
        BULGE_VELOCITY_SIGMA_2,
        BULGE_VELOCITY_SIGMA_3,
    ),
}

#: The velocity coordinates of a star, in the column order of the
#: (pm_ra, pm_dec, rv) -> galactocentric map.
VELOCITY_COORDS = ("pm_ra", "pm_dec", "rv")

# Log-spaced trapezoid for the position normalizer.  The default distance
# support spans 18.4 e-folds (1 mpc to 100 kpc); 2e5 intervals put the step at
# 9e-5 in ln d, i.e. ~0.7 pc at the bulge, against the 50 pc hinge width and
# the >100 pc scale lengths.  Measured: 4x the points moves log Z by < 1e-9.
_POSITION_QUAD_POINTS = 200001


def _hinge(t, xp):
    # Smooth max(t, 0) (C-infinity), transition width ~50 pc in kpc units --
    # keeps the plateau/cutoff kinks differentiable.
    return 0.5 * (t + xp.sqrt(t * t + 0.0025))


def branch_log_number_densities(x, y, z, r, xp=pt):
    """log number density (stars/pc^3) of each branch, in BRANCHES order.

    ``x, y, z`` galactocentric kpc (physics.galactic_xyz), ``r`` the
    SMOOTHED cylindrical radius ``sqrt(x^2 + y^2 + 1e-6)`` the velocity
    frame also uses.  ``xp`` is ``pt`` for the potential and ``np`` for the
    position normalizer -- one form, two backends, as
    ``chabrier_logmass_logp`` does for the IMF.
    """
    z_smooth = xp.sqrt(z**2 + 1e-6)

    # 1. Thin disk.  Radial profile: exponential with Rd = 2.6 kpc, held
    # FLAT inside R = DISK_RDBREAK (genulens's DISK=2 inner plateau) -- a
    # plain exponential over-weights near-GC disk lenses by ~3x at R = 1
    # kpc.  Anchored to the local (R0, midplane) number density.
    r_beyond_sun = SUN_GC_DISTANCE - DISK_RDBREAK
    log_dens_thin = (
        np.log(DISK_LOCAL_NUMBER_DENSITY)
        - (_hinge(r - DISK_RDBREAK, xp) - r_beyond_sun) / DISK_SCALE_LENGTH
        - z_smooth / DISK_SCALE_HEIGHT
    )

    # 2. Thick disk -- same plateau, its own scale length/height.
    log_dens_thick = (
        np.log(THICK_DISK_LOCAL_NUMBER_DENSITY)
        - (_hinge(r - DISK_RDBREAK, xp) - r_beyond_sun)
        / THICK_DISK_SCALE_LENGTH
        - z_smooth / THICK_DISK_SCALE_HEIGHT
    )

    # 3. Bulge/bar, anchored at its center.  The Gaussian cylindrical cutoff
    # beyond BULGE_RC (genulens Rc/srob) is what confines the bar: without
    # it the shallow exp(-r_s/2) profile is still 9% of central at a 4-kpc
    # lens distance (vs ~4e-6 in genulens), flooding the disk range with
    # spurious bulge stars.
    cos_bar = np.cos(BULGE_BAR_ANGLE)
    sin_bar = np.sin(BULGE_BAR_ANGLE)
    x_bar = x * cos_bar + y * sin_bar
    y_bar = -x * sin_bar + y * cos_bar
    r_bulge_coord = xp.sqrt(
        (x_bar / BULGE_DENSITY_X_0) ** 2
        + (y_bar / BULGE_DENSITY_Y_0) ** 2
        + (z / BULGE_DENSITY_Z_0) ** 2
    )
    log_dens_bulge = (
        np.log(BULGE_CENTRAL_NUMBER_DENSITY)
        - 0.5 * r_bulge_coord
        - 0.5 * (_hinge(r - BULGE_RC, xp) / BULGE_RC_WIDTH) ** 2
    )
    return log_dens_thin, log_dens_thick, log_dens_bulge


def _log_total_number_density(d_pc, sun_xyz, los_hat, offsets=None):
    """numpy log sum_b n_b (stars/pc^3) at distance(s) ``d_pc`` along the
    sight line (``line_of_sight_basis(...)[2:]``), with the graph's own
    1 mpc clamp.  ``offsets`` (one per branch, present_day's object-density
    offsets) are added to the branch log densities."""
    d_kpc = np.maximum(np.asarray(d_pc, dtype=float), 1e-3) / 1e3
    x, y, z = _galactic_xyz_np(d_kpc, sun_xyz, los_hat)
    r = np.sqrt(x**2 + y**2 + 1e-6)
    dens = branch_log_number_densities(x, y, z, r, xp=np)
    if offsets is not None:
        dens = [d + o for d, o in zip(dens, offsets)]
    return np.logaddexp.reduce(np.stack(dens), axis=0)


def log_position_norm(lower_pc, upper_pc, sun_xyz, los_hat, offsets=None):
    """log of int_lower^upper sum_b n_b(d) d^2 dd along the sight line.

    The number of stars per steradian in the cone out to ``upper`` -- the
    normalizer of the position half of the kinematic prior, whose density
    in the sampled ``star.distance`` (pc) is ``sum_b n_b(d) d^2``.  The
    sight line is the anchor's and the bounds are static, so this is one
    constant, computed once at build time (the same legitimacy argument as
    ``_chabrier_log_norm``).  Trapezoid in ``ln d`` (integrand ``n d^3``)
    with the peak factored out.
    """
    lnd = np.linspace(
        np.log(float(lower_pc)), np.log(float(upper_pc)), _POSITION_QUAD_POINTS
    )
    f = (
        _log_total_number_density(np.exp(lnd), sun_xyz, los_hat, offsets)
        + 3.0 * lnd
    )
    peak = float(np.max(f))
    return float(np.log(np.trapezoid(np.exp(f - peak), lnd)) + peak)


def _hard_links(param):
    """{element: link spec} of a Parameter's HARD links (sigma: 0 initval
    links), or {} -- a mock or an unlinked parameter has none."""
    links = getattr(param, "element_links", None) or {}
    return links.get("hard", {}) or {}


def linked_star_structure(stars, n_star, prefix="star"):
    """Which stars carry a position of their own, and which velocity
    coordinates of each star are hard-linked.

    Returns ``(root_of, vel_linked, refused)``:

    - ``root_of[k]`` is the star whose position star ``k`` shares.  A star
      whose ``distance`` is HARD-linked to exactly one other star's
      distance, and to nothing else, joins that star's group (chains are
      followed to the end); every other star is its own root.
    - ``vel_linked[k]`` is the sorted tuple of column indices (into
      ``VELOCITY_COORDS``) of star ``k``'s hard-linked velocity
      coordinates, for every star that has any.
    - ``refused`` lists ``(k, dep_paths)`` for a hard-linked distance that
      reads anything other than ONE other star's distance: that star keeps
      its own position term (and the caller warns), because there is no
      single partner whose position it is.
    """
    parent = list(range(n_star))
    refused = []
    for k, spec in sorted(_hard_links(stars.distance).items()):
        k = int(k)
        intra = set(spec.get("intra_deps", ()) or ())
        paths = tuple(spec.get("dep_paths", ()) or ())
        foreign = [
            p
            for p in paths
            if not (
                len(p.split(".")) == 3
                and p.split(".")[0] == prefix
                and p.split(".")[2] == "distance"
            )
        ]
        if len(intra) == 1 and not foreign and k not in intra:
            parent[k] = int(next(iter(intra)))
        else:
            refused.append((k, paths))

    root_of = np.arange(n_star)
    for k in range(n_star):
        j, seen = k, set()
        while parent[j] != j:
            if j in seen:  # a cycle the link engine should have refused
                raise ValueError(
                    f"galacticmodel: the hard links on {prefix}.distance "
                    f"form a cycle through element {k}; a cycle of hard "
                    f"links has no value."
                )
            seen.add(j)
            j = parent[j]
        root_of[k] = j

    vel_linked = {}
    for c, name in enumerate(VELOCITY_COORDS):
        param = getattr(stars, name, None)
        if param is None:
            continue
        for k in _hard_links(param):
            vel_linked.setdefault(int(k), set()).add(c)
    vel_linked = {k: tuple(sorted(v)) for k, v in vel_linked.items()}
    return root_of, vel_linked, refused


def _log_velocity_marginal(cols, c, s, xprime, vphi_mean, sigmas, m_rot, v0):
    """log density of the hard-linked velocity coordinates ``cols`` of ONE
    star under ONE branch, in the km/s coordinates ``xprime``.

    The branch is a Gaussian in ``u = (v_r, v_phi, v_z)`` with mean
    ``(0, vphi_mean, 0)`` and covariance ``diag(sigmas^2)``, and
    ``u = Q (v0 + M_rot x')`` with ``Q`` physics.polar_velocity's map
    ``[[c, s, 0], [-s, c, 0], [0, 0, 1]]``.  ``c, s`` are built from the
    SMOOTHED radius (``r^2 = x^2 + y^2 + 1e-6``), so ``Q`` is a rotation
    times ``rho2 = c^2 + s^2`` on its first two rows, 1 - 1e-4 for a source
    0.1 kpc from the Galactic center -- not negligible against an exact
    normalization, so it is carried exactly: ``x' = B (u - Q v0)`` with
    ``B = M_rot^T Q^-1`` (``M_rot`` is orthogonal, so its inverse is its
    transpose), hence ``x' ~ N(B (m - Q v0), B S B^T)`` and the marginal of any subset is
    the matching sub-block.  The caller adds ``log |det Q| = log rho2``,
    which the full density (written in u) leaves out.  Written element by
    element for |cols| in {1, 2} -- the only sizes a CONDITIONAL needs (0
    linked is the full density, 3 linked leaves nothing to condition).
    """
    m = np.asarray(m_rot, dtype=float)
    rho2 = c * c + s * s
    # Q^-1 = [[c, -s, 0], [s, c, 0], [0, 0, rho2]] / rho2, as columns n.
    qinv = [
        [c / rho2, -s / rho2, 0.0],
        [s / rho2, c / rho2, 0.0],
        [0.0, 0.0, 1.0],
    ]
    # B[i][n] = sum_k M[k, i] Qinv[k][n]
    b_mat = [
        [sum(m[k, i] * qinv[k][n] for k in range(3)) for n in range(3)]
        for i in range(3)
    ]
    qv0 = [c * v0[0] + s * v0[1], -s * v0[0] + c * v0[1], v0[2]]
    du = [0.0 - qv0[0], vphi_mean - qv0[1], 0.0 - qv0[2]]
    var = [float(sg) ** 2 for sg in sigmas]

    mean = {i: sum(b_mat[i][n] * du[n] for n in range(3)) for i in cols}

    def cov(i, j):
        return sum(b_mat[i][n] * b_mat[j][n] * var[n] for n in range(3))

    if len(cols) == 1:
        (i,) = cols
        cii = cov(i, i)
        dx = xprime[i] - mean[i]
        return -0.5 * dx**2 / cii - 0.5 * (_LOG_2PI + pt.log(cii))
    if len(cols) == 2:
        i, j = cols
        cii, cjj, cij = cov(i, i), cov(j, j), cov(i, j)
        det = cii * cjj - cij**2
        dxi = xprime[i] - mean[i]
        dxj = xprime[j] - mean[j]
        quad = (cjj * dxi**2 - 2.0 * cij * dxi * dxj + cii * dxj**2) / det
        return -0.5 * quad - _LOG_2PI - 0.5 * pt.log(det)
    raise ValueError(f"a velocity marginal over {len(cols)} coordinates")


class GalacticModel(Component):
    """The Galactic prior on a line of sight, derived from genulens
    (Koshimoto, Baba & Bennett 2021, ApJ 917, 78).  Cite K21 when you use
    it; see the module docstring for where it departs from genulens."""

    # The only mass function implemented below.  ``IMF:`` is validated
    # against this in __init__ rather than silently ignored: the key selected
    # KROUPA_IMF_SLOPE / SALPETER_IMF_SLOPE for a power-law prior that had
    # not been applied since the Chabrier lognormal replaced it, so every
    # `IMF: Salpeter` in a config was a no-op.  Kroupa is still unsupported:
    # it is a BROKEN power law (alpha = 1.3 below 0.5 Msun, 2.3 above) and
    # needs a piecewise, continuity-matched density, not a single slope --
    # KROUPA_IMF_SLOPE is only its low-mass segment.
    SUPPORTED_IMFS = ("chabrier", "salpeter")

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Galactic Prior"
        if self.n_elements != 1:
            # Only config[0] is ever read (imf, anchor_idx), and the extra
            # blocks used to leak into the likelihood as a broadcast: with
            # 2 blocks and 1 star the whole kinematic prior was counted
            # TWICE, and with 2 blocks and 3 stars it raised a bare shape
            # error.  One galactic prior describes one line of sight.
            raise ValueError(
                f"galacticmodel takes exactly one config block, got "
                f"{self.n_elements}.  It is a single prior on the line of "
                f"sight shared by every star; use 'anchor_idx' to choose "
                f"which star's (ra, dec) defines it."
            )
        imf = str(self.config[0].get("IMF", "chabrier")).lower()
        if imf not in self.SUPPORTED_IMFS:
            raise ValueError(
                f"galacticmodel IMF '{self.config[0].get('IMF')}' is not "
                f"implemented.  Supported: {', '.join(self.SUPPORTED_IMFS)} "
                f"('chabrier' = Chabrier 2003 system IMF, a lognormal in "
                f"log10 mass; 'salpeter' = Salpeter 1955 power law).  "
                f"'Kroupa' was accepted but silently ignored before 2026-08."
            )
        self.imf = imf
        self.anchor_idx = self.config[0].get("anchor_idx", 0)
        # Present-day mass function (review 1.7.2): an independent toggle,
        # orthogonal to the IMF choice (it evolves whichever IMF is
        # selected).  A boolean, never a string: "false" in YAML quotes
        # would otherwise be truthy.
        present_day = self.config[0].get("present_day", True)
        if not isinstance(present_day, bool):
            raise ValueError(
                f"galacticmodel present_day must be true or false, got "
                f"{present_day!r}."
            )
        self.present_day = present_day

    @property
    def prefix(self):
        return "galacticmodel"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "IMF",
                "kind": "option",
                "accepts": list(cls.SUPPORTED_IMFS),
                "required": False,
                "doc": (
                    "Initial mass function for the stellar mass prior "
                    "(default 'chabrier'): 'chabrier' is the Chabrier 2003 "
                    "system IMF, a lognormal in log10 mass; 'salpeter' is "
                    "the Salpeter 1955 power law dN/dM ~ M^-2.35.  Both are "
                    "normalized over the sampled logmass bounds, so their "
                    "logp values are directly comparable.  Anything else "
                    "raises.  This is the STELLAR mass prior: a star may opt "
                    "out of it individually with the star block's "
                    "'mass_function: ffp' (the free-floating-planet mass "
                    "function), which is how an FFP lens and a stellar source "
                    "get different mass priors in the same fit."
                ),
            },
            {
                "key": "present_day",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Evolve the IMF to a PRESENT-DAY mass function per "
                    "Galactic population (default true; false keeps the "
                    "IMF as the present-day mass function, for regression "
                    "comparison): stars above "
                    "their population's MIST turnoff become white dwarfs "
                    "(Kalirai+2008), neutron stars or black holes "
                    "(Raithel+2018), with the age marginalized over each "
                    "population's age distribution (Koshimoto+2021: thin "
                    "disk 0-10 Gyr, thick disk 12 Gyr, bulge 9 +/- 1 Gyr), "
                    "and the mass prior moves inside the thin/thick/bulge "
                    "mixture.  Applies to every star on the stellar IMF "
                    "(not to 'mass_function: ffp' stars).  Remnants are "
                    "DARK in the mass prior only: a lens's light still "
                    "comes from its stellar relation (review 8.10.7)."
                ),
            },
            {
                "key": "anchor_idx",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Index of the star whose (ra, dec) defines the line of "
                    "sight for the density/kinematic prior (default 0). One "
                    "galacticmodel block describes one sight line."
                ),
            },
        ]

    def register_parameters(self, system):
        """No parameters to sample! Just an empty manifest."""
        self.manifest = {}

    # ------------------------------------------------------------------
    # Reporting: tell the tables what these potentials are.
    #
    # Both potentials below act on parameters this component does not own
    # and does not sample -- they are `pm.Potential`s over star.logmass and
    # over (star.distance, pm_ra, pm_dec, rv).  `Parameter.get_prior_str`
    # can only see a parameter's OWN fields, so without these declarations
    # every one of them is reported as "Uniform" (a bounded element with no
    # sigma), which is exactly the prior these terms replace.  See
    # parameter.PriorContribution.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Present-day mass function (review 1.7.2; pdmf.py holds the physics).
    # ------------------------------------------------------------------

    def _pdmf_mask(self, stars, ffp_mask):
        """Which stars take the present-day mass function: every star on
        the stellar IMF when ``present_day: true``, none otherwise."""
        n_star = int(np.atleast_1d(stars.logmass.initval).size)
        if not self.present_day:
            return np.zeros(n_star, dtype=bool)
        if ffp_mask.size:
            return ~ffp_mask
        return np.ones(n_star, dtype=bool)

    def _log_imf_np(self):
        """The selected IMF's numpy log density in x = log10 M (the PDMF
        evolves whichever IMF ``IMF:`` chose)."""
        if self.imf == "salpeter":
            k = SALPETER_IMF_SLOPE + 1.0
            return lambda x: k * np.log(10.0) * x
        return lambda x: chabrier_logmass_logp(x, param=None, xp=np)

    def _build_pdmf(self, stars):
        bounds = _sampled_bounds(stars.logmass)
        if bounds is None:
            raise ValueError(
                f"[{self.prefix}] present_day: true needs finite "
                f"star.logmass lower/upper bounds to normalize the "
                f"present-day mass function; none could be read."
            )
        n_star = int(np.atleast_1d(stars.logmass.initval).size)
        lower = np.broadcast_to(np.atleast_1d(bounds[0]), (n_star,))
        upper = np.broadcast_to(np.atleast_1d(bounds[1]), (n_star,))
        return PresentDayMassFunction(self._log_imf_np(), lower, upper)

    def _pdmf_offsets(self, pdmf_mask):
        """Per branch, per star: -log F_lum,b on PDMF stars, 0 elsewhere."""
        w = pdmf_mask.astype(float)
        return [-w * self._pdmf.log_lum_frac[b] for b in range(len(BRANCHES))]

    def _pdmf_branch_terms(self, birth_logmass, pdmf_mask):
        """(mass_terms, dens_offsets), one per-star vector per branch.

        ``mass_terms[b]`` is log phi_b(logmass) normalized over each star's
        own support (a numpy table interpolated linearly in the graph),
        times the 0/1 PDMF mask -- arithmetic, so a non-PDMF star's entry
        is exactly 0 with a zero gradient (the table is finite everywhere,
        so 0 * finite cannot poison anything).
        """
        pd = self._pdmf
        w = pdmf_mask.astype(float)
        # The BIRTH mass (_birth_logmass): a star whose mass loss a component
        # models is placed in the mass function at its initial mass.
        x = birth_logmass
        mass_terms = []
        for b in range(len(BRANCHES)):
            log_phi = interp_table(pd.log_table[b], pd.x0, PDMF_GRID_DEX, x)
            mass_terms.append(w * (log_phi - pd.log_norm[b]))
        return mass_terms, self._pdmf_offsets(pdmf_mask)

    def _add_remnant_probability(self, birth_logmass, branch_logp, pdmf_mask):
        """``galacticmodel.p_remnant``: per PDMF star (in star order), the
        probability that it is a dark remnant (WD/NS/BH) rather than a
        luminous star, given its mass, position and velocity: the branch
        responsibilities times each branch's dark fraction at its mass.  A
        Deterministic for reporting; it does not enter the logp.  For a
        star in a linked group the responsibilities are its own terms', an
        approximation to the group's shared population."""
        idx = np.nonzero(pdmf_mask)[0]
        pd = self._pdmf
        stacked = pt.stack(branch_logp)[:, idx]
        resp = pt.exp(stacked - pm.math.logsumexp(stacked, axis=0))
        x = birth_logmass[idx]
        dark = pt.stack(
            [
                interp_table(pd.dark_frac[b], pd.x0, PDMF_GRID_DEX, x)
                for b in range(len(BRANCHES))
            ]
        )
        pm.Deterministic(
            f"{self.prefix}.p_remnant", pt.sum(resp * dark, axis=0)
        )

    def _birth_logmass(self, system, stars):
        """The ``stars.logmass`` vector with each star's BIRTH mass in place.

        A star whose present-day mass differs from its initial one has a
        component that says so: ``Component.initial_logmass`` returns
        ``{star index: log10 initial mass}`` (the evolutionary model, which
        also adds the Jacobian of that change of coordinate).  Two
        components claiming one star would be two different birth masses,
        and raise.  Assembled with ``set_subtensor``, never ``pt.where``.
        """
        claims = {}
        for comp in system.active_components.values():
            for si, value in comp.initial_logmass(system).items():
                if si in claims:
                    raise ValueError(
                        f"[{self.prefix}] star {si} has an initial mass from "
                        f"both '{claims[si][0]}' and '{comp.prefix}'; a star "
                        f"has one birth mass."
                    )
                claims[si] = (comp.prefix, value)
        logmass = stars.logmass.value
        if not claims:
            return logmass
        idx = sorted(claims)
        return pt.set_subtensor(
            logmass[np.asarray(idx, dtype=int)],
            pt.stack([claims[i][1] for i in idx]),
        )

    def _declare_mass_prior(self, stars, ffp_mask, pdmf_mask=None):
        """Describe the imf_prior potential, per star.

        Two descriptions, because the potential really is two densities: the
        stellar IMF selected by ``IMF:``, and the FFP mass function on the
        stars whose ``mass_function:`` opted out of it.  Both are normalized
        over star.logmass's own support, so both supersede the uniform.
        """
        logmass = getattr(stars, "logmass", None)
        if logmass is None:
            return

        if self.imf == "salpeter":
            imf_latex = r"Salpeter (1955) IMF in $\log_{10} M$"
            imf_text = "Salpeter (1955) IMF in log10 M"
        else:
            # "lognormal + $m^{-1.3}$" rather than a bare "Chabrier (2003)
            # IMF": the label used to name the paper while the code applied
            # only its low-mass segment (review 3.7.1), so the table said
            # nothing that would have caught it.  Naming both segments is
            # what makes the claim checkable at a glance.
            imf_latex = (
                r"Chabrier (2003) IMF in $\log_{10} M$ "
                r"(lognormal $+\ m^{-1.3}$ tail)"
            )
            imf_text = "Chabrier (2003) IMF in log10 M (lognormal + m^-1.3)"

        if pdmf_mask is not None and pdmf_mask.any():
            imf_latex = (
                imf_latex.replace(" in $\\log_{10} M$", "")
                + r" evolved to a present-day mass function per Galactic "
                r"population (MIST turnoff; WD/NS/BH remnants)"
            )
            imf_text = (
                imf_text.replace(" in log10 M", "")
                + " evolved to a present-day mass function per Galactic "
                "population (MIST turnoff; WD/NS/BH remnants)"
            )
        imf_elements = np.nonzero(~ffp_mask)[0] if ffp_mask.size else None
        if imf_elements is None or len(imf_elements):
            logmass.add_prior_contribution(
                latex=imf_latex,
                text=imf_text,
                elements=imf_elements,
                supersedes_bounds=True,
            )

        # The FFP slope is per star, so a system with two differently-sloped
        # FFPs gets two contributions, each naming its own elements.
        alphas = np.atleast_1d(getattr(stars, "ffp_alpha", []))
        for i in np.nonzero(ffp_mask)[0]:
            alpha = float(alphas[i]) if i < alphas.size else float("nan")
            logmass.add_prior_contribution(
                latex=(
                    r"Sumi et al. (2023) FFP mass function "
                    rf"($\alpha = {alpha:g}$)"
                ),
                text=f"Sumi+2023 FFP mass function (alpha = {alpha:g})",
                elements=[int(i)],
                supersedes_bounds=True,
            )

    def _add_mass_prior_prose(self, system, stars, ffp_mask, pdmf_mask=None):
        """Modeling-draft sentences for the imf_prior potential.

        Mirrors ``_declare_mass_prior``'s per-star split: the stellar IMF
        sentence names its stars only when some star opted onto the FFP
        mass function, and each FFP star gets the Sumi slope actually
        applied.  Declared next to the potential (declare-at-site rule).
        """
        from ...outputs.prose import get_collector, join_names
        from ...outputs.texutils import latex_escape

        prose = get_collector(system)

        if getattr(stars, "logmass", None) is None:
            return
        # Read the star inventory defensively: unit tests drive this
        # potential through minimal mock star components.
        star_names = [str(nm) for nm in (getattr(stars, "names", None) or [])]
        n = max(
            int(getattr(stars, "n_elements", len(star_names))),
            len(star_names),
            int(ffp_mask.size),
        )
        if len(star_names) < n:
            star_names += [str(i) for i in range(len(star_names), n)]
        mask = ffp_mask if ffp_mask.size else np.zeros(n, dtype=bool)
        # Says "system" and names both segments: the sentence used to read
        # "\citet{Chabrier:2003} lognormal", which was an accurate
        # description of what the code did and an inaccurate description of
        # the cited IMF (review 3.7.1).  A modeling draft has to be able to
        # stand as written.
        imf_cite = (
            r"\citet{Salpeter:1955} power-law"
            if self.imf == "salpeter"
            else r"\citet{Chabrier:2003} system"
        )
        imf_names = [nm for nm, f in zip(star_names, mask) if not f]
        if imf_names:
            target = (
                "each modeled star"
                if len(imf_names) == n
                else (
                    ("star " if len(imf_names) == 1 else "stars ")
                    + join_names(latex_escape(nm) for nm in imf_names)
                )
            )
            prose.add(
                f"We applied the {imf_cite} initial mass function as a "
                f"prior on the mass of {target}, normalized over the "
                f"sampled log-mass support.",
                section="priors",
                key=f"{self.prefix}.imf",
                rank=10,
            )
        if imf_names and pdmf_mask is not None and pdmf_mask.any():
            prose.add(
                r"We evolved it to a present-day mass function separately "
                r"for each Galactic population, following "
                r"\citet{Koshimoto:2021} and \citet{Lam:2020}: stars above "
                r"the main-sequence turnoff, taken from the MIST lifetimes "
                r"\citep{Dotter:2016,Choi:2016}, became white dwarfs with "
                r"the initial-final mass relation of \citet{Kalirai:2008} "
                r"or neutron stars and black holes following "
                r"\citet{Raithel:2018}, with the age marginalized over "
                r"each population's age distribution (thin disk 0--10\,Gyr, "
                r"thick disk 12\,Gyr, bulge $9 \pm 1$\,Gyr).  The mass "
                r"prior therefore enters inside the thin-disk, thick-disk "
                r"and bulge mixture of the Galactic model.  Remnants are "
                r"dark in the mass prior only; the light of each star still "
                r"follows its stellar relation.",
                section="priors",
                key=f"{self.prefix}.pdmf",
                rank=10,
            )
        if imf_names and self.imf != "salpeter":
            # A config fact, not a fitted value: the smoothing width is a
            # constant of the implementation, so it is interpolatable and
            # belongs in the draft rather than left for a reader to discover
            # that the published piecewise form was not applied verbatim.
            prose.add(
                r"The \citet{Chabrier:2003} system IMF is piecewise -- a "
                r"lognormal below $1\,M_\odot$ matched to "
                r"$dN/d\log m \propto m^{-1.3}$ above it -- and is only "
                r"$C^0$ at the join.  We blended the two segments over a "
                rf"{CHABRIER_BLEND_WIDTH_DEX:g} dex ramp so the prior is "
                r"differentiable there; the blend departs from the exact "
                r"piecewise form by less than $0.02$ nats, far below the "
                r"published uncertainty on the high-mass exponent "
                r"($x = 1.3 \pm 0.3$).",
                section="priors",
                key=f"{self.prefix}.imf.blend",
                rank=10,
            )
        alphas = np.atleast_1d(getattr(stars, "ffp_alpha", []))
        for i in np.nonzero(mask)[0]:
            alpha = float(alphas[i]) if i < alphas.size else float("nan")
            prose.add(
                f"For {latex_escape(star_names[i])}, we instead adopted "
                r"the free-floating-planet mass function of "
                r"\citet{Sumi:2023}, "
                r"$dN/d\log M \propto M^{-\alpha}$ with "
                rf"$\alpha = {alpha:g}$.",
                section="priors",
                key=f"{self.prefix}.ffp.{i}",
                rank=11,
            )

    def _declare_kinematic_prior(self, stars):
        """Describe the kinematic_prior potential.

        One joint density over distance and the velocity coordinates, so the
        same term is declared against each of them; distance additionally
        carries the ``2 log d`` volume element, which is why
        ``Star.build_likelihood`` stands down where this component exists.
        """
        distance = getattr(stars, "distance", None)
        if distance is not None:
            distance.add_prior_contribution(
                latex=(
                    r"Galactic density $\times$ kinematics "
                    r"(incl. $d^{2}$ volume element)"
                ),
                text="Galactic density x kinematics (incl. d^2 volume)",
                supersedes_bounds=True,
            )
        for name in ("pm_ra", "pm_dec", "rv"):
            param = getattr(stars, name, None)
            if param is not None:
                param.add_prior_contribution(
                    latex="Galactic kinematic model",
                    supersedes_bounds=True,
                )

    def _require_anchor_position(self, stars, anchor):
        """Refuse to build the prior on an anchor star nobody positioned.

        The whole prior -- density, d^2 volume, velocity frame, the three
        branch Gaussians, for EVERY star -- is a function of one line of
        sight, the anchor's.  A star left at defaults.yaml's ra/dec (180, 0)
        is Galactic (l 276, b +60), and a sight line leaving the plane at
        60 deg makes the density fall exponentially with distance: the
        2026-09 DC2018 sweep ran every event that way (the configs pinned the
        Lens's coordinates and anchored on the Source) and reported source
        distances and radii at half their true values for a week before the
        cause was found (notes 2026-09-24).  Nobody wants a Galactic prior
        without saying where it is, so this is an error, not a warning.

        "Set" means the provenance ledger says the start came from the user
        or from data (a hint pushed by an instrument).  A Parameter with no
        ledger at all (tests, scripts) is left alone: the ledger is the only
        witness that distinguishes "left at the default" from "unknown".
        """
        # A coordinate is "unset" when the provenance ledger does not credit it
        # to the user or to data AND its start value is still defaults.yaml's
        # placeholder.  Both halves matter: tests position stars by writing
        # `star.ra.initval = ...` straight past the ledger (provenance says
        # "default", value says otherwise -- positioned), and a mock Parameter
        # has no ledger at all (getattr, not a method call).  The placeholders
        # are pinned by tests/test_galactic_anchor_position.py.
        placeholder = {
            "ra": np.radians(ANCHOR_PLACEHOLDER_RA_DEG),
            "dec": np.radians(ANCHOR_PLACEHOLDER_DEC_DEG),
        }
        unset = []
        for name, param in (("ra", stars.ra), ("dec", stars.dec)):
            src = getattr(param, "element_initval_source", lambda i: None)(
                anchor
            )
            if src in ("user", "data", "solved"):
                continue
            if np.isclose(
                float(param.element_start(anchor)),
                placeholder[name],
                rtol=0.0,
                atol=1e-12,
            ):
                unset.append(name)
        if not unset:
            return
        names = getattr(stars, "names", None)
        label = names[anchor] if names and anchor < len(names) else str(anchor)
        missing = " and ".join(f"star.{label}.{n}" for n in unset)
        raise ValueError(
            f"[{self.prefix}] the galactic model's anchor star '{label}' "
            f"(star index {anchor}) has no sky position: {missing} "
            f"still at the defaults.yaml value. The whole density and "
            f"kinematic prior is evaluated along this star's line of sight, "
            f"so set star.{label}.ra and star.{label}.dec in the params file "
            f"(or point 'anchor_idx' at a star that has them)."
        )

    def _warn_if_anchor_coords_sampled(self, stars, ra_rad, dec_rad, anchor):
        """Warn once if the anchor star's ra/dec are SAMPLED, since the
        line-of-sight basis below is built from their start values and frozen.

        Same policy as ``Lens._frozen_op_coords_deg``: keep the freeze, say so
        -- but only when it could conceivably matter, i.e. when the sampler
        actually moves the coordinates (galacticmodel + gaia/abs astrometry).

        The freeze is safe here too, and separately measured rather than
        assumed by analogy with the microlensing Op: shifting the sight line
        by 1 arcsec moves the anchor's galactocentric velocity by 1.3e-3 km/s
        (4e-5 of a 30 km/s thin-disk dispersion) and its galactic position by
        5e-5 kpc, against kpc-scale density gradients.  It stays negligible
        out to ~1 arcmin (0.08 km/s, 3 pc) and only reaches 0.16 sigma at a
        full degree -- five orders of magnitude beyond the mas-scale posterior
        width astrometry gives ra/dec.
        """
        moving = [
            name
            for name, param in (("ra", stars.ra), ("dec", stars.dec))
            if param.element_is_sampled(anchor)
        ]
        if not moving:
            return
        logger.warning(
            f"[{self.prefix}] star.{'/'.join(moving)} of the anchor star "
            f"{anchor} is sampled, but the galactic model's line of "
            f"sight is FROZEN at the start value (ra="
            f"{np.degrees(ra_rad):.6f} deg, dec={np.degrees(dec_rad):.6f} "
            f"deg) for the whole fit. This is safe -- 1 arcsec of coordinate "
            f"error moves the galactocentric velocity by ~1e-3 km/s and the "
            f"galactic position by ~0.05 pc -- but the sampled ra/dec do NOT "
            f"feed the density or kinematic prior."
        )

    def build_likelihood(self, model, system):
        stars = system.star

        # 1. Pre-compute the transformation matrix using Astropy, ONCE, from
        # the anchor star's initial RA/Dec.  This is a property of the line
        # of sight, not of a component element: the loop this replaced ran
        # over self.n_elements (the number of galacticmodel BLOCKS) while
        # indexing self.anchor_idx every time, so it stacked n_elements
        # identical copies and then relied on those copies broadcasting
        # against the (n_star,) sampled vectors -- which silently doubled
        # the whole prior for 2 blocks and 1 star, and raised a bare shape
        # error whenever n_blocks and n_stars disagreed and neither was 1.
        # Keeping the matrix 2-D (and the direction cosines scalar) lets it
        # broadcast over any number of stars by construction.
        # A lens whose position is DERIVED from its source's (review 8.10.4)
        # has no start of its own -- its resolved initval is a placeholder --
        # so the sight line is read off the star it derives from, which it
        # matches to the lens-source separation (mas; the freeze below is
        # safe out to ~1 arcmin).
        anchor = stars.position_root(self.anchor_idx)
        if anchor != self.anchor_idx:
            logger.info(
                f"[{self.prefix}] anchor star {self.anchor_idx}'s position is "
                f"derived from star {anchor}'s (the lens track); the line of "
                f"sight is read from star {anchor}."
            )
        self._require_anchor_position(stars, anchor)
        ra_rad = stars.ra.element_start(anchor)
        dec_rad = stars.dec.element_start(anchor)
        self._warn_if_anchor_coords_sampled(stars, ra_rad, dec_rad, anchor)

        # Shared with the start-value hints (physics.line_of_sight_basis):
        # the affine (pm_ra_cosdec, pm_dec, rv) -> galactocentric velocity map,
        # as one offset plus three unit-response columns.
        # Positions come from the SAME transform (sun_xyz + d * los_hat), so
        # the two can never be in different frames (review 1.7.1; the frame
        # is documented in physics.py's module docstring).
        m_rot_np, v0_arr, sun_xyz_np, los_hat_np = line_of_sight_basis(
            ra_rad, dec_rad
        )

        # Convert to tensors for graph injection
        M_rot = pt.as_tensor_variable(m_rot_np)  # (3, 3)
        v0 = pt.as_tensor_variable(v0_arr)  # (3,)

        # 2. PyTensor Math Helpers
        def get_galactocentric_velocity(dist_kpc, pm_ra, pm_dec, rv_ms):
            v_alpha_kms = K_VEL_CONVERSION * pm_ra * dist_kpc
            v_delta_kms = K_VEL_CONVERSION * pm_dec * dist_kpc
            v_rad_kms = rv_ms / 1e3
            v_icrs = pt.stack(
                [v_alpha_kms, v_delta_kms, v_rad_kms], axis=1
            )  # (N, 3)
            v_gal_offset = (M_rot @ v_icrs[:, :, None]).squeeze(-1)
            return v0 + v_gal_offset

        def get_galactic_xyz(dist):
            # physics.galactic_xyz is pure arithmetic, so the same function
            # serves numpy scalars (the hints) and tensors (here).
            return _galactic_xyz_np(dist, sun_xyz_np, los_hat_np)

        # match the IMF.  The key is validated in __init__, so every branch
        # here is a supported option.  BOTH are densities in the SAMPLED
        # coordinate x = log10(M), not in M, and both are summed over every
        # modeled star (lens and source alike).
        #
        # A mass function is a density in the BIRTH mass.  For most stars
        # that is star.logmass itself; a star whose mass loss is modeled
        # (a MIST track) samples its PRESENT-DAY mass, and the component
        # that models it supplies the initial mass along with the Jacobian
        # that makes this a density in it (Component.initial_logmass).
        birth_logmass = self._birth_logmass(system, stars)
        if self.imf == "salpeter":
            # Salpeter (1955): dN/dM ~ M^-alpha, alpha = 2.35.  Change of
            # variables to x = log10(M)  (M = 10^x, dM/dx = M ln10):
            #   dN/dx = (dN/dM)(dM/dx) = M^-alpha * M ln10
            #         = ln10 * M^(1-alpha) = ln10 * 10^((1-alpha)x)
            #   log p(x) = (1 - alpha) * ln10 * x + const
            # SALPETER_IMF_SLOPE is the SIGNED MASS-SPACE exponent (-alpha),
            # NOT an already-converted log-space slope, so the slope in the
            # sampled coordinate is SALPETER_IMF_SLOPE + 1 = -1.35, i.e.
            # -1.35*ln10 = -3.108 nats per dex of mass.  A linear tilt is
            # fine for NUTS now that the bounded-coordinate transform is
            # sound, and the support IS bounded (star.logmass carries hard
            # lower/upper), so the density is proper and normalizable.
            k = SALPETER_IMF_SLOPE + 1.0
            imf_logp = k * np.log(10.0) * birth_logmass - (
                _power_law_log_norm(k, stars.logmass)
            )
        else:
            # Chabrier (2003) system IMF -- the PIECEWISE one: a lognormal
            # below 1 Msun (beautiful constant curvature -1/sigma^2 for NUTS,
            # and the low-mass end is where unconstrained particles fall into
            # the abyss) matched to a dN/dlog m ~ m^-1.3 tail above it,
            # smoothed across the slope discontinuity.  The form, the match,
            # the smoothing and the numerical normalizer all live in
            # chabrier_logmass_logp; only the lognormal segment shipped until
            # 2026-08 (review 3.7.1).
            #
            # Normalized over the SAME star.logmass support the power law
            # uses, so both IMFs are proper densities and switching between
            # them moves logp by a meaningful amount rather than by an
            # arbitrary offset.  Two archived-baseline moves have come out
            # of that constant: -0.3568 nats/star when the lognormal was
            # first normalized, and +0.0029 nats/star when the tail landed
            # (the tail adds a little mass above 1 Msun, so log Z drops from
            # 0.3568196 to 0.3539152).  The second is small at the default
            # start ONLY because these stars are near the Chabrier peak; the
            # density itself moves by +0.57 nats at 10 Msun and more above.
            # Both re-pinned in tests/test_runaway_logp_regression.py.
            imf_logp = chabrier_logmass_logp(birth_logmass, stars.logmass)

        # Per-star opt-out of the stellar IMF.  A microlensing lens that is a
        # free-floating planet MUST be declared as a star (Lens._validate_
        # bodies rejects a non-star primary), and everything else that makes
        # it a star is right -- the galactic density and kinematic priors
        # below are mass-independent.  Only the mass prior is wrong: under
        # Chabrier a 3 Mjup body sits ~3.3 sigma below the peak, ~5.4 nats of
        # penalty for being what the user said it is.  The choice has to be
        # PER STAR (a stellar source with an FFP lens is the realistic
        # system), which is why it lives on the star block and not here; see
        # Star._parse_mass_functions.
        ffp_mask = getattr(stars, "ffp_mask", None)
        ffp_mask = (
            np.zeros(0, dtype=bool)
            if ffp_mask is None
            else np.asarray(ffp_mask, dtype=bool)
        )
        if ffp_mask.any():
            # Both branches are finite everywhere on the support (one linear
            # in x, one quadratic), so a plain weighted sum is safe and, being
            # arithmetic rather than pt.switch, keeps a clean gradient on
            # every backend.
            w = ffp_mask.astype(float)
            imf_logp = (
                w
                * ffp_logmass_logp(
                    birth_logmass, stars.ffp_alpha, stars.logmass
                )
                + (1.0 - w) * imf_logp
            )

        # Present-day mass function (review 1.7.2): the stars it covers take
        # their mass prior INSIDE the population mixture below (it depends
        # on the population), so they leave this potential.  With
        # present_day: false this block is skipped and the potential is the
        # pre-1.7.2 expression (bit-identical start logp on every shipped
        # example).
        pdmf_mask = self._pdmf_mask(stars, ffp_mask)
        self._pdmf = None
        if pdmf_mask.any():
            self._pdmf = self._build_pdmf(stars)
            imf_logp = (1.0 - pdmf_mask.astype(float)) * imf_logp

        pm.Potential(f"{self.prefix}.imf_prior", pt.sum(imf_logp))

        self._declare_mass_prior(stars, ffp_mask, pdmf_mask)
        self._add_mass_prior_prose(system, stars, ffp_mask, pdmf_mask)
        ######

        # even though non-physical values will be rejected
        # we still need to be able to calculate a likelihood,
        distance = pt.maximum(stars.distance.value, 1e-3) / 1e3  # kpc
        x, y, z = get_galactic_xyz(distance)
        r = pt.sqrt(x**2 + y**2 + 1e-6)

        vel = get_galactocentric_velocity(
            distance, stars.pm_ra.value, stars.pm_dec.value, stars.rv.value
        )  # km/s
        v_x, v_y, v_z = vel[:, 0], vel[:, 1], vel[:, 2]
        v_r, v_phi = polar_velocity(x, y, r, v_x, v_y)

        # Each mixture branch must carry its own normalization: constants
        # cancel within a single Potential but NOT across the logsumexp.
        # Velocity Gaussians contribute -log(sigma1*sigma2*sigma3) (the
        # shared (2*pi)^(3/2) is identical in all branches and dropped
        # here, and reported by evidence_log_norm); the density anchors
        # (stars/pc^3, genulens's number-density channel, see constants.py)
        # set the physical population ratios.  The densities live in
        # branch_log_number_densities, shared with the position normalizer.
        #
        # Galactic rotation is AZIMUTHAL: the circular-speed offset belongs
        # on v_phi (paired with the azimuthal dispersion) and the radial
        # component v_r is zero-centered. These centers were swapped until
        # 2026-08: the prior penalized an EXACTLY circular co-rotating orbit
        # by ~49 nats and was maximized by a star plunging radially at the
        # rotation speed (pinned by tests/test_galactic_model.py). On
        # examples/ob140939 the swap gave the anti-rotation parallax
        # solution +5 nats and flipped the mode weights to 0.98/0.02 AGAINST
        # the Yee et al. 2015 proper-motion-preferred solution. Sign
        # convention: v_phi as computed by polar_velocity is positive
        # for co-rotation at every position, in the model frame (astropy's
        # Galactocentric mirrored in x, positions and velocities alike;
        # pinned to 1e-6 km/s against astropy by tests/
        # test_galactic_frame.py).  The thick disk's mean rotation (170 km/s) is the
        # asymmetric drift; the bulge rotates cylindrically, Omega * r.
        log_dens = branch_log_number_densities(x, y, z, r)
        vphi_means = (
            DISK_ROTATION_VELOCITY,
            THICK_DISK_ROTATION_VELOCITY,
            BULGE_ROTATION_ANGULAR_VELOCITY * r,
        )
        log_vel = []
        log_sig = []
        for name, vphi_mean in zip(BRANCHES, vphi_means):
            s1, s2, s3 = BRANCH_SIGMAS[name]
            log_vel.append(
                (-0.5 / s1**2) * v_r**2
                + (-0.5 / s2**2) * (v_phi - vphi_mean) ** 2
                + (-0.5 / s3**2) * v_z**2
            )
            log_sig.append(np.log(s1 * s2 * s3))
        mass_terms = None
        if self._pdmf is not None:
            # Per branch: the star's PDMF term (normalized over its own
            # logmass support) and the object-density offset -log F_lum,b
            # (the density anchors count LUMINOUS stars, genulens's n0MS;
            # with remnants in the mass function the branch's objects are
            # n_b / F_lum,b).  Zero on stars the PDMF does not cover.
            # Every term is a finite table interpolation: no branch
            # selection, no where (CLAUDE.md).
            mass_terms, dens_offsets = self._pdmf_branch_terms(
                birth_logmass, pdmf_mask
            )
            log_dens = [d + off for d, off in zip(log_dens, dens_offsets)]
        branch_logp = [
            dens + lv - ls for dens, lv, ls in zip(log_dens, log_vel, log_sig)
        ]
        if mass_terms is not None:
            branch_logp = [bl + mt for bl, mt in zip(branch_logp, mass_terms)]
            self._add_remnant_probability(
                birth_logmass, branch_logp, pdmf_mask
            )

        volume_element = 2.0 * pt.log(distance * 1000.0)

        # The velocity Gaussians are evaluated in km/s while the sampled
        # coordinates are (pm_ra, pm_dec, rv); the change of variables
        # v = M_rot @ (K*pm_ra*d, K*pm_dec*d, rv/1e3) + v0 needs
        # |det dv/d(pm_ra, pm_dec, rv)| = (K*d)^2 * 1e-3 (M_rot is
        # orthogonal, |det| = 1 -- det = -1 with the frame's x mirror; the constant 1e-3 from m/s -> km/s is
        # dropped here and reported by evidence_log_norm).  Without the
        # +2*log(d) the intended prior rho * d^2 * f(v) * (K*d)^2 was
        # applied as rho * d^2 * f(v), under-weighting large distances by
        # d^2.
        velocity_jacobian = 2.0 * pt.log(K_VEL_CONVERSION * distance)

        # Linked stars (star.md, "Linked stars: one position, one density
        # term").  A star whose distance is HARD-linked to another star's
        # has no position of its own: the system it belongs to has one
        # position along the line of sight, so it gets one density and one
        # d^2 volume element, and the linked star contributes only the
        # density of its own still-free velocity coordinates, inside the
        # same branch (one population for the whole system).
        n_star = int(
            getattr(stars, "n_elements", 0)
            or np.atleast_1d(stars.distance.initval).size
        )
        root_of, vel_linked, refused = linked_star_structure(
            stars, n_star, prefix=getattr(stars, "prefix", "star")
        )
        self._linked_star_report(stars, root_of, vel_linked, refused)
        roots = sorted(set(int(j) for j in root_of))
        members = [k for k in range(n_star) if root_of[k] != k]

        if not members and not vel_linked:
            # Every star independent: exactly the expression this component
            # has always built (bit-identical start logp on every shipped
            # example).
            kinematic_penalty = pt.sum(
                pm.math.logsumexp(pt.stack(branch_logp), axis=0)
                + volume_element
                + velocity_jacobian
            )
        else:
            kinematic_penalty = self._grouped_kinematic_logp(
                stars,
                distance,
                x,
                y,
                r,
                log_dens,
                log_vel,
                log_sig,
                vphi_means,
                volume_element,
                velocity_jacobian,
                root_of,
                vel_linked,
                m_rot_np,
                v0_arr,
                mass_terms=mass_terms,
            )

        self._evidence_log_norm = self._kinematic_evidence_log_norm(
            stars,
            roots,
            vel_linked,
            (sun_xyz_np, los_hat_np),
            dens_offsets=(
                None if self._pdmf is None else self._pdmf_offsets(pdmf_mask)
            ),
        )
        pm.Potential(f"{self.prefix}.kinematic_prior", kinematic_penalty)

        self._declare_kinematic_prior(stars)
        from ...outputs.prose import get_collector

        get_collector(system).add(
            r"We imposed Galactic density and kinematic priors on the "
            r"distance, proper motion, and radial velocity of each modeled "
            r"star, following a simplified version of the parametric "
            r"Galactic model of \citet{Koshimoto:2021}.",
            section="priors",
            key=f"{self.prefix}.kinematic",
            rank=20,
        )

    # ------------------------------------------------------------------
    # Linked stars and the normalization (star.md, "Linked stars: one
    # position, one density term").
    # ------------------------------------------------------------------

    def _linked_star_report(self, stars, root_of, vel_linked, refused):
        """Say, once per build, which stars share a position and what each
        linked star still contributes -- and warn for a hard-linked
        distance the rule cannot place."""
        names = list(getattr(stars, "names", None) or [])

        def label(k):
            return names[k] if k < len(names) else str(k)

        for k, j in enumerate(root_of):
            if j == k:
                continue
            linked = vel_linked.get(k, ())
            free = [
                VELOCITY_COORDS[c] for c in range(3) if c not in set(linked)
            ]
            what = (
                "nothing further (its velocity is hard-linked too)"
                if not free
                else "the density of its free velocity coordinate(s) "
                + ", ".join(free)
                + (
                    " conditional on the linked "
                    + ", ".join(VELOCITY_COORDS[c] for c in linked)
                    if linked
                    else ""
                )
                + ", in the same Galactic population"
            )
            logger.info(
                f"[{self.prefix}] star '{label(k)}' shares the position of "
                f"star '{label(int(j))}' (hard-linked distance): one "
                f"position, one density and d^2 volume term; '{label(k)}' "
                f"contributes {what}."
            )
        for k, paths in refused:
            logger.warning(
                f"[{self.prefix}] star '{label(k)}' has a hard-linked "
                f"distance that does not name exactly one other star's "
                f"distance (it reads {list(paths) or 'nothing'}), so there "
                f"is no single partner whose position it shares. It keeps "
                f"its own Galactic density term, which is then a factor on "
                f"the link's sources rather than a density of a coordinate "
                f"of its own. Link it as 'initval: star.<other>.distance' "
                f"with 'sigma: 0' to share that star's position instead."
            )

    def _grouped_kinematic_logp(
        self,
        stars,
        distance,
        x,
        y,
        r,
        log_dens,
        log_vel,
        log_sig,
        vphi_means,
        volume_element,
        velocity_jacobian,
        root_of,
        vel_linked,
        m_rot_np,
        v0_arr,
        mass_terms=None,
    ):
        """The kinematic prior with linked stars grouped onto one position.

        Per group g with root r and members k (root included):

            log p_g = logsumexp_b [ log n_b(d_r)
                                    + sum_k log f_b(x_free,k | x_linked,k) ]
                      + 2 log d_r + (branch-independent Jacobians)

        so the group has ONE density and ONE volume element, and ONE
        population (the branch is shared by the whole system).  A member
        with no hard-linked velocity coordinate contributes the full,
        exactly normalized velocity density; one with some contributes the
        exactly normalized CONDITIONAL density of the rest; one with all
        three contributes nothing.  Every per-star vector is assembled with
        pt.set_subtensor (never pt.where -- CLAUDE.md).
        """
        n_star = len(root_of)
        roots = sorted(set(int(j) for j in root_of))
        members = [k for k in range(n_star) if root_of[k] != k]

        # Per-branch per-star velocity term V_b and branch-independent J.
        # Default (an independent star): the unnormalized Gaussian and the
        # (K d)^2 Jacobian, exactly the fast path's terms.
        V = [lv - ls for lv, ls in zip(log_vel, log_sig)]
        J = velocity_jacobian

        # A member with no linked velocity: the FULL velocity density,
        # exactly normalized -- the constants the root's term leaves to
        # evidence_log_norm are owed here in the model, because a member's
        # term must integrate to exactly 1 over its free coordinates for a
        # linked config to compare exactly against an unlinked one.
        for k in members:
            if k in vel_linked:
                continue
            # |det Q| of the smoothed polar map (_log_velocity_marginal) is
            # owed too: 1 - 1e-4 for a star 0.1 kpc from the Galactic center.
            log_det_q = pt.log((x[k] ** 2 + y[k] ** 2) / r[k] ** 2)
            J = pt.set_subtensor(
                J[k], J[k] + _LOG_KMS_PER_MS - 1.5 * _LOG_2PI + log_det_q
            )

        # A star (root or member) with hard-linked velocity coordinates: the
        # conditional density of its free coordinates given the linked ones,
        # log f(x) - log f_L(x_L), in the SAMPLED coordinates (each column's
        # Jacobian: K d for a proper motion, 1e-3 for rv).
        for k, cols in sorted(vel_linked.items()):
            if len(cols) == 3:
                for b in range(len(V)):
                    V[b] = pt.set_subtensor(V[b][k], 0.0)
                J = pt.set_subtensor(J[k], 0.0)
                continue
            d_k = distance[k]
            log_jac_col = (
                pt.log(K_VEL_CONVERSION * d_k),
                pt.log(K_VEL_CONVERSION * d_k),
                _LOG_KMS_PER_MS,
            )
            xprime = (
                K_VEL_CONVERSION * stars.pm_ra.value[k] * d_k,
                K_VEL_CONVERSION * stars.pm_dec.value[k] * d_k,
                stars.rv.value[k] / 1e3,
            )
            c_k = x[k] / r[k]
            s_k = y[k] / r[k]
            full_jac = J[k] + _LOG_KMS_PER_MS  # (K d)^2 * 1e-3
            # |det Q| of the smoothed polar map (see _log_velocity_marginal).
            log_det_q = pt.log(c_k * c_k + s_k * s_k)
            linked_jac = sum(log_jac_col[c] for c in cols)
            for b, name in enumerate(BRANCHES):
                vphi_mean = vphi_means[b]
                if not np.isscalar(vphi_mean):
                    vphi_mean = vphi_mean[k]
                log_marg = _log_velocity_marginal(
                    cols,
                    c_k,
                    s_k,
                    xprime,
                    vphi_mean,
                    BRANCH_SIGMAS[name],
                    m_rot_np,
                    v0_arr,
                )
                cond = (
                    V[b][k]
                    - 1.5 * _LOG_2PI
                    + full_jac
                    + log_det_q
                    - (log_marg + linked_jac)
                )
                V[b] = pt.set_subtensor(V[b][k], cond)
            J = pt.set_subtensor(J[k], 0.0)

        # The present-day mass term is each star's own (a member has a mass
        # of its own even when its whole velocity is linked), added after
        # the velocity terms are set.  The object-density offset is already
        # in log_dens, which only the ROOT's row reads below.
        if mass_terms is not None:
            V = [v + mt for v, mt in zip(V, mass_terms)]

        # W[g, k] = 1 when star k belongs to group g (its root's row).
        W = np.zeros((len(roots), n_star))
        for k in range(n_star):
            W[roots.index(int(root_of[k])), k] = 1.0
        W = pt.as_tensor_variable(W)
        roots_idx = np.asarray(roots, dtype=np.int64)
        group_branch = [
            dens[roots_idx] + pt.dot(W, v) for dens, v in zip(log_dens, V)
        ]
        return (
            pt.sum(pm.math.logsumexp(pt.stack(group_branch), axis=0))
            + pt.sum(volume_element[roots_idx])
            + pt.sum(J)
        )

    def _kinematic_evidence_log_norm(
        self, stars, roots, vel_linked, los, dens_offsets=None
    ):
        """The constant to ADD to a log evidence for the kinematic prior to
        be a normalized density (outputs/evidence's sign convention: the
        bridge integrates exp(model logp), and lnZ_norm = lnZ + this).

        Per ROOT (a star with a position of its own), the term in the model
        is the normalized density over (distance, pm_ra, pm_dec, rv) plus

            log Z_pos + (3/2) log(2 pi) - log(1e-3)

        where Z_pos = int sum_b n_b(d) d^2 dd over the root's static
        distance bounds (log_position_norm) -- the stars per steradian in
        the cone -- and the last two are the Gaussians' (2 pi)^(3/2) and
        the rv column's m/s -> km/s Jacobian.  A root whose own velocity is
        partly hard-linked is already exactly conditional (no Gaussian
        constant owed).  A root whose distance is PINNED owes log of its
        density times d^2 at the pin instead of Z_pos (the remaining
        velocity mixture is then normalized).  A member owes nothing: its
        term is exact in the model.  Velocity bounds are treated as
        infinite (pm +/- 1e5 mas/yr is > 400 km/s beyond 1 pc).

        ``dens_offsets`` (present_day only): per branch, per star, the
        object-density offset ``-log F_lum,b`` the model adds to a PDMF
        root's branch density; the position normalizer integrates the same
        offset densities, so the mixture over (position, mass) stays a
        normalized density.
        """
        sun_xyz, los_hat = los
        bounds = _sampled_bounds(stars.distance)
        is_derived = getattr(stars.distance, "element_is_derived", None)
        links = getattr(stars.distance, "element_links", None) or {}
        if links.get("lower") or links.get("upper"):
            logger.warning(
                f"[{self.prefix}] star.distance carries a linked lower/upper "
                f"bound; the kinematic prior's position normalizer uses the "
                f"STATIC bounds, so evidence_log_norm is approximate there."
            )
        total = 0.0
        cache = {}
        for k in roots:
            offs = (
                None
                if dens_offsets is None
                else tuple(float(o[k]) for o in dens_offsets)
            )
            sampled = stars.distance.element_is_sampled(k) or bool(
                is_derived(k) if callable(is_derived) else False
            )
            if sampled:
                if bounds is None:
                    logger.warning(
                        f"[{self.prefix}] cannot read finite star.distance "
                        f"bounds; the kinematic prior's position normalizer "
                        f"is omitted from evidence_log_norm."
                    )
                    c = 0.0
                else:
                    lo = float(
                        np.atleast_1d(bounds[0])[
                            min(k, np.size(bounds[0]) - 1)
                        ]
                    )
                    hi = float(
                        np.atleast_1d(bounds[1])[
                            min(k, np.size(bounds[1]) - 1)
                        ]
                    )
                    if (lo, hi, offs) not in cache:
                        cache[(lo, hi, offs)] = log_position_norm(
                            lo, hi, sun_xyz, los_hat, offsets=offs
                        )
                    c = cache[(lo, hi, offs)]
            else:
                d0 = float(stars.distance.element_start(k))
                c = float(
                    _log_total_number_density(
                        d0, sun_xyz, los_hat, offsets=offs
                    )
                ) + 2.0 * np.log(max(d0, 1e-3))
            if k not in vel_linked:
                c += 1.5 * _LOG_2PI - _LOG_KMS_PER_MS
            total -= c
        return float(total)

    def evidence_log_norm(self):
        """Constant that makes this component's potentials normalized priors
        in a log evidence (ADD it to lnZ; see _kinematic_evidence_log_norm).

        Covers the kinematic prior only -- the IMF/FFP potential is already
        normalized over star.logmass's support.  Available after
        build_likelihood; 0.0 before.
        """
        return float(getattr(self, "_evidence_log_norm", 0.0))

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass

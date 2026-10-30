"""Age and metallicity distributions of the Galactic populations (review 1.7.5).

THE one definition of each population's age distribution g_b(T) and
metallicity distribution function (MDF) p_b([Fe/H]), for the three branches
of the galacticmodel mixture (thin disk, thick disk, bulge/bar).  Two
consumers read it, and they must agree (review 1.7.5's consistency rule):

  * ``pdmf.py`` -- the present-day mass function marginalizes each
    population's age distribution at build time (``age_cdf``), and takes its
    one turnoff metallicity per population from the MDF's mean
    (``POPULATION_FEH``);
  * ``galacticmodel.build_likelihood`` -- a star under the galactic model
    whose ``age`` / ``initfeh`` is SAMPLED (in practice a star on an
    evolutionary track) gets ``log g_b(age)`` and ``log p_b(initfeh)`` inside
    the branch mixture, each normalized over the star's own support, so the
    branch weights stay the physical number densities.

Each density is written ONCE for both backends (``xp=np`` for the
normalizers and the PDMF, ``xp=pt`` for the graph), as
``branch_log_number_densities`` and ``chabrier_logmass_logp`` are.

Ages (Gyr), on [0, AGE_MAX_GYR]:

  * thin disk: star formation ~ exp(-(10 Gyr - T) / 7 Gyr) over 0-10 Gyr --
    genulens's wtSFR (Koshimoto, Baba & Bennett 2021, ApJ 917, 78 = K21,
    after Bovy 2017's tSFR = 7 Gyr), which is what #526's PDMF used.  K21
    splits it into 7 age bins with their own scale heights; our one-layer
    thin disk has one age distribution everywhere.  The 10 Gyr edge is a
    LOGISTIC ramp of scale ``THIN_AGE_EDGE_GYR`` (0.1 Gyr; 10-90% over 0.44
    Gyr) rather than a step, so the per-star density is differentiable;
    K21's step is the zero-width limit.
  * thick disk: Gaussian 12 +/- 1 Gyr.  K21 (after Robin et al. 2014) uses
    a single 12 Gyr age; a delta cannot be a per-star density, so we give it
    the same 1 Gyr width K21 gives the bulge.  This is OUR choice (stated
    here and in star.md); it smears the thick disk's turnoff by ~0.05 Msun.
  * bulge/bar: Gaussian 9 +/- 1 Gyr (K21, after Koshimoto et al. 2020).

  All three are truncated to [0, AGE_MAX_GYR], the age of the Universe
  (star/defaults.yaml's ``age`` upper bound).

Metallicity distribution functions ([Fe/H], dex):

  * thin disk: Gaussian, sigma ``THIN_FEH_SIGMA`` = 0.20 dex (the width of
    the solar-neighborhood MDF, whose peak is at solar; Casagrande et al.
    2011, A&A 530, A138), centered on solar at R0 with the Cepheid radial
    gradient -0.060 dex/kpc (Genovali et al. 2014, A&A 566, A37).  The
    gradient acts between R = 5.3 kpc (``DISK_RDBREAK``, where the density
    model's inner plateau begins) and R = 15 kpc and is held flat outside
    (smooth hinges, 50 pc wide): +0.17 dex in the inner disk, -0.41 beyond
    15 kpc.
  * thick disk: Gaussian -0.78 +/- 0.30 dex (Robin et al. 2003, A&A 409,
    523, the Besancon thick disk; genulens's own source-population prior
    centers its thick disk at -0.8).
  * bulge: Hill et al. 2011 (A&A 534, A80, Table 5, the 1-D [Fe/H] split of
    219 red-clump giants in Baade's window), a two-Gaussian mixture: 45% at
    -0.30 +/- 0.25 and 55% at +0.32 +/- 0.11 dex.  Zoccali et al. 2017
    (A&A 599, A12, GIBS) find the same two peaks (~ -0.4 and +0.3) at every
    latitude, with a metal-poor FRACTION that varies with latitude (0.29 at
    b = -3.5 deg, 0.77 at b = -8.5); one fixed Baade's-window mix is our
    simplification.

The upper end of any star's [Fe/H] support on a track is the MIST grid's
(+0.5 dex), a MODEL-DOMAIN bound (the one place rope-not-gates allows a hard
edge: the model does not exist beyond it).  It truncates ~3% of the bulge
MDF and ~5% of the inner thin disk's; each density is renormalized over the
star's own support, so the truncated tail's weight is redistributed, not
lost from the branch weight.  The MDF prior applies to ``initfeh`` (the
initial metallicity the track is indexed by); the surveys measure the
present-day surface [Fe/H], which differs by diffusion (~0.1 dex for old
dwarfs) -- below these MDFs' widths.
"""

import numpy as np
import pytensor.tensor as pt

from ...constants import DISK_RDBREAK, SUN_GC_DISTANCE
from .physics import hinge

#: Branch order (galacticmodel.BRANCHES, pdmf.POPULATIONS).
POPULATIONS = ("thin_disk", "thick_disk", "bulge")

#: Age of the Universe (Gyr): the support of every age distribution and
#: star/defaults.yaml's age upper bound.
AGE_MAX_GYR = 13.77

# ---- ages ------------------------------------------------------------
THIN_AGE_MAX_GYR = 10.0
THIN_SFR_TIMESCALE_GYR = 7.0
#: Logistic scale of the thin disk's 10 Gyr edge (not K21's; see above).
THIN_AGE_EDGE_GYR = 0.1
THICK_AGE_MEAN_GYR = 12.0
#: Our width for K21's single-age thick disk (see the module docstring).
THICK_AGE_SIGMA_GYR = 1.0
BULGE_AGE_MEAN_GYR = 9.0
BULGE_AGE_SIGMA_GYR = 1.0

# ---- metallicities ---------------------------------------------------
THIN_FEH_AT_R0 = 0.0
THIN_FEH_GRADIENT = -0.060  # dex/kpc, Genovali et al. 2014
THIN_FEH_SIGMA = 0.20
THIN_FEH_R_INNER = DISK_RDBREAK  # kpc: gradient flat inside
THIN_FEH_R_OUTER = 15.0  # kpc: gradient flat outside
THICK_FEH_MEAN = -0.78  # Robin et al. 2003
THICK_FEH_SIGMA = 0.30
#: Hill et al. 2011 Table 5 (1-D [Fe/H]): (weight, mean, sigma).
BULGE_FEH_COMPONENTS = ((0.45, -0.30, 0.25), (0.55, 0.32, 0.11))

#: The one turnoff metallicity per population the PDMF's MIST lifetimes use:
#: each MDF's mean (the thin disk's at R0).  The thick disk's -0.78 puts its
#: 12 Gyr dying mass at 0.856 Msun, genulens's Minidie.dat value (0.857).
POPULATION_FEH = {
    "thin_disk": THIN_FEH_AT_R0,
    "thick_disk": THICK_FEH_MEAN,
    "bulge": float(sum(w * m for w, m, _ in BULGE_FEH_COMPONENTS)),
}

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))

#: Quadrature for the per-star normalizers and the PDMF's age CDF: the
#: sharpest feature is the 0.1 Gyr logistic edge, ~150 steps wide here.
_AGE_QUAD_POINTS = 20001
_FEH_QUAD_POINTS = 20001


def _softplus(t, xp):
    if xp is np:
        return np.logaddexp(0.0, t)
    return pt.softplus(t)


def _log_normal(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - np.log(sigma) - _LOG_SQRT_2PI


def log_age_density(population, age_gyr, xp=pt):
    """UNNORMALIZED log g_b(T), T in Gyr, finite for every real T.

    Arithmetic only (a softplus, a square): no branch selection, so the
    gradient is clean on every backend.
    """
    t = age_gyr
    if population == "thin_disk":
        # exp(T / tau) * sigmoid((T_max - T) / w); log sigmoid(u) =
        # -softplus(-u).
        return t / THIN_SFR_TIMESCALE_GYR - _softplus(
            (t - THIN_AGE_MAX_GYR) / THIN_AGE_EDGE_GYR, xp
        )
    if population == "thick_disk":
        return -0.5 * ((t - THICK_AGE_MEAN_GYR) / THICK_AGE_SIGMA_GYR) ** 2
    if population == "bulge":
        return -0.5 * ((t - BULGE_AGE_MEAN_GYR) / BULGE_AGE_SIGMA_GYR) ** 2
    raise ValueError(f"unknown Galactic population {population!r}")


def log_age_norm(population, lower, upper):
    """log int_lower^upper g_b(T) dT (numpy; per-element arrays allowed)."""
    lower = np.atleast_1d(np.asarray(lower, dtype=float))
    upper = np.atleast_1d(np.asarray(upper, dtype=float))
    n = max(lower.size, upper.size)
    lower = np.broadcast_to(lower, (n,))
    upper = np.broadcast_to(upper, (n,))
    out = np.empty(n)
    for k in range(n):
        grid = np.linspace(lower[k], upper[k], _AGE_QUAD_POINTS)
        f = log_age_density(population, grid, xp=np)
        peak = f.max()
        out[k] = np.log(np.trapezoid(np.exp(f - peak), grid)) + peak
    return out


def age_cdf(population, age_gyr):
    """CDF of g_b on [0, AGE_MAX_GYR] at numpy age(s): the PDMF's
    P_b(alive | M_ini) = age_cdf(b, tau(M_ini))."""
    grid = np.linspace(0.0, AGE_MAX_GYR, _AGE_QUAD_POINTS)
    f = np.exp(log_age_density(population, grid, xp=np))
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(grid))]
    )
    return np.interp(age_gyr, grid, cdf / cdf[-1])


def thin_disk_feh_mean(r_kpc, xp=pt):
    """Thin-disk MDF center at cylindrical radius R (kpc): the gradient
    between THIN_FEH_R_INNER and THIN_FEH_R_OUTER, flat outside, with the
    density model's own C-infinity hinge (50 pc) at both ends."""
    r_in = THIN_FEH_R_INNER + hinge(r_kpc - THIN_FEH_R_INNER, xp)
    r_eff = THIN_FEH_R_OUTER - hinge(THIN_FEH_R_OUTER - r_in, xp)
    return THIN_FEH_AT_R0 + THIN_FEH_GRADIENT * (r_eff - SUN_GC_DISTANCE)


#: The range of the thin disk's MDF center over all radii (hinges included).
THIN_FEH_MEAN_RANGE = (
    THIN_FEH_AT_R0
    + THIN_FEH_GRADIENT * (THIN_FEH_R_OUTER - SUN_GC_DISTANCE)
    - 0.01,
    THIN_FEH_AT_R0
    + THIN_FEH_GRADIENT * (THIN_FEH_R_INNER - SUN_GC_DISTANCE)
    + 0.01,
)


def log_feh_density(population, feh, r_kpc=None, xp=pt):
    """NORMALIZED (over the real line) log p_b([Fe/H]).

    The thin disk's center depends on the star's cylindrical radius
    ``r_kpc`` (a tensor in the graph); the other two do not read it.
    """
    if population == "thin_disk":
        if r_kpc is None:
            raise ValueError("the thin-disk MDF needs the star's radius R")
        return _log_normal(feh, thin_disk_feh_mean(r_kpc, xp), THIN_FEH_SIGMA)
    if population == "thick_disk":
        return _log_normal(feh, THICK_FEH_MEAN, THICK_FEH_SIGMA)
    if population == "bulge":
        (w1, m1, s1), (w2, m2, s2) = BULGE_FEH_COMPONENTS
        return xp.logaddexp(
            np.log(w1) + _log_normal(feh, m1, s1),
            np.log(w2) + _log_normal(feh, m2, s2),
        )
    raise ValueError(f"unknown Galactic population {population!r}")


def _erf(x, xp):
    if xp is np:
        from scipy.special import erf

        return erf(x)
    return pt.erf(x)


def thin_disk_log_feh_norm(r_kpc, lower, upper, xp=pt):
    """log of the thin-disk MDF's probability over [lower, upper] at R.

    Symbolic (its center moves with the star's position), as an erf
    difference.  That difference cancels only when BOTH bounds sit many
    sigma on the same side of the center; ``check_thin_disk_feh_support``
    refuses such a support at build time, over the whole range the center
    can take, so the graph never meets it.
    """
    mu = thin_disk_feh_mean(r_kpc, xp)
    s = THIN_FEH_SIGMA * np.sqrt(2.0)
    return xp.log(
        0.5 * (_erf((upper - mu) / s, xp) - _erf((lower - mu) / s, xp))
    )


def check_thin_disk_feh_support(lower, upper, label):
    """Raise if a star's [Fe/H] support could carry < 1e-6 of the thin-disk
    MDF at some radius (the model used outside its domain)."""
    lower = np.atleast_1d(np.asarray(lower, dtype=float))
    upper = np.atleast_1d(np.asarray(upper, dtype=float))
    mus = np.linspace(*THIN_FEH_MEAN_RANGE, 201)
    s = THIN_FEH_SIGMA * np.sqrt(2.0)
    from scipy.special import erf

    for lo, hi in zip(*np.broadcast_arrays(lower, upper)):
        z = 0.5 * (erf((hi - mus) / s) - erf((lo - mus) / s))
        if np.min(z) < 1e-6:
            raise ValueError(
                f"{label}: the [Fe/H] support [{lo:g}, {hi:g}] holds "
                f"{np.min(z):.2g} of the thin-disk metallicity distribution "
                f"at some Galactocentric radius (center "
                f"{THIN_FEH_MEAN_RANGE[0]:+.2f}..{THIN_FEH_MEAN_RANGE[1]:+.2f}"
                f", sigma {THIN_FEH_SIGMA}); widen it or set "
                f"'population_feh: false' on the galacticmodel block."
            )


def log_feh_norm(population, lower, upper):
    """log int p_b over [lower, upper] for the R-independent MDFs (numpy,
    per-element arrays)."""
    if population == "thin_disk":
        raise ValueError("the thin-disk MDF normalizer is symbolic")
    lower = np.atleast_1d(np.asarray(lower, dtype=float))
    upper = np.atleast_1d(np.asarray(upper, dtype=float))
    n = max(lower.size, upper.size)
    lower = np.broadcast_to(lower, (n,))
    upper = np.broadcast_to(upper, (n,))
    out = np.empty(n)
    for k in range(n):
        grid = np.linspace(lower[k], upper[k], _FEH_QUAD_POINTS)
        f = log_feh_density(population, grid, xp=np)
        peak = f.max()
        out[k] = np.log(np.trapezoid(np.exp(f - peak), grid)) + peak
    return out

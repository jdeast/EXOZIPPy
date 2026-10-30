"""Present-day mass function (PDMF) of each Galactic population (review 1.7.2).

The galacticmodel's mass prior used to apply the IMF to every star as if it
were the mass function TODAY: every star ever born still alive and luminous.
A microlensing lens is whatever object is there now -- stars below its
population's turnoff, and the dark remnants (white dwarfs, neutron stars,
black holes) of everything heavier.  This module builds, per population
(thin disk, thick disk, bulge), the normalized present-day density in
x = log10(M / Msun):

    phi_b(x) = phi_b^lum(x) + phi_b^dark(x)
    phi_b^lum(x)  = IMF(x) * P_b(alive | x)
    P_b(alive | M_ini) = int g_b(T) [T < tau(M_ini, [Fe/H]_b)] dT
    phi_b^dark(x) = sum_type int IMF(x_ini) (1 - P_b(alive | x_ini))
                    P(type | M_ini) delta(x - log10 m_type(M_ini)) dx_ini

with the age MARGINALIZED over the population's age distribution g_b(T) at
build time, which is exact for the mass prior and leaves one fixed 1-D table
per population.  The prescriptions are genulens's (Koshimoto, Baba & Bennett
2021, ApJ 917, 78 = K21; and its source, model/mass_function.cpp), which in
turn are PopSyCLE's (Lam et al. 2020, ApJ 889, 31, App. C):

  * ages: thin disk star formation ~ exp(-(10 Gyr - T) / 7 Gyr) over
    0-10 Gyr (genulens's wtSFR, i.e. more old stars); thick disk a single
    age of 12 Gyr; bulge Gaussian 9 +/- 1 Gyr;
  * turnoff: from OUR MIST v2.5 grid's lifetimes (``mist_lifetimes.csv``,
    scripts/make_mist_lifetimes.py), at one [Fe/H] per population
    (``POPULATION_FEH``); genulens uses PARSEC isochrones instead, a few %
    apart in M_die;
  * white dwarfs (M_ini < 9 Msun): Kalirai et al. 2008,
    M_WD = 0.109 M_ini + 0.394;
  * neutron stars and black holes (M_ini >= 9): Raithel, Sukhbold & Ozel
    2018 (ApJ 856, 35), probabilistic P(NS | M_ini) in mass bins, the NS
    and BH mass fits exactly as genulens codes them (including its 1.60 +/-
    0.158 Msun NS scatter for 18.5 <= M_ini < 21.7 and the f_ej = 0.9 BH
    ejection fraction below 42.21 Msun).

Smoothness: everything is built on a 1e-3 dex grid and the LINEAR densities
are convolved with a Gaussian of ``SMOOTH_DEX`` (0.02 dex, the design's
kernel) before taking the log, so the thick disk's single-age turnoff step,
the white-dwarf upper edge (1.375 Msun) and the R18 bin edges become erf
ramps 0.02 dex wide rather than cliffs.  A branch density never reaches
zero: ``FLOOR_REL`` (1e-20 of the branch's peak) keeps the log finite where
a population has no objects at all (e.g. a 3 Msun object in the 12 Gyr thick
disk, between its neutron stars and its black holes).  That floor is for
finiteness only -- the mixture over populations is what keeps heavy masses
reachable (the young thin disk and the remnants): rope, not gates.

The tables are numbers; the graph only interpolates them linearly in x
(``interp_table``), so the gradient is a finite difference of adjacent table
entries everywhere: no branch selection, no pt.where, no radicand.

Out of scope here (review 1.7.2 ruling, JDE 2026-10-29): remnant natal kicks
(the dark branch shares its population's kinematics), remnant LIGHT (8.10.7:
a lens's blend tie still charges a luminous relation's light whatever its
mass), binaries, giant-branch mass loss, a metallicity spread within a
population.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytensor.tensor as pt
from scipy.special import ndtr

#: Branch order (galacticmodel.BRANCHES).
POPULATIONS = ("thin_disk", "thick_disk", "bulge")

# ---- age distributions (K21 / genulens) -------------------------------
#: Thin disk: SFR weight exp(-(THIN_AGE_MAX - T) / THIN_SFR_TIMESCALE) on
#: [0, THIN_AGE_MAX] Gyr (genulens stellar_population_runtime wtSFR,
#: gamma = 1 / tSFR, tSFR = 7 Gyr).
THIN_AGE_MAX_GYR = 10.0
THIN_SFR_TIMESCALE_GYR = 7.0
#: Thick disk: one age (Robin et al. 2014 via K21).
THICK_AGE_GYR = 12.0
#: Bulge/bar: Gaussian (Koshimoto et al. 2020 via K21).
BULGE_AGE_MEAN_GYR = 9.0
BULGE_AGE_SIGMA_GYR = 1.0

#: One initial [Fe/H] per population for the MIST lifetimes.  genulens uses
#: one PARSEC metallicity per population too; its dying masses (Minidie.dat:
#: thin disk 10 Gyr 1.011, thick disk 12 Gyr 0.857 Msun) sit between our
#: [Fe/H] = 0 and -0.5 rows, closest to these choices.
POPULATION_FEH = {"thin_disk": 0.0, "thick_disk": -0.5, "bulge": 0.0}

# ---- initial-final mass relations ------------------------------------
#: Kalirai et al. 2008 white dwarfs below this initial mass (genulens
#: white_dwarf_initial_mass_max_ = 9 Msun), NS/BH above.
WD_INITIAL_MASS_MAX = 9.0
NS_MASS_RANGE = (1.2, 2.1)  # genulens kNeutronStarMin/MaxMass
#: Raithel+2018 P(NS | M_ini): (upper edge of the bin, P_NS).  The last bin
#: (60-120 Msun in R18/genulens) is extended to the top of the logmass
#: support, since our IMF support runs past genulens's 120 Msun cap.
R18_P_NS = (
    (15.0, 1.0),
    (17.8, 0.679),
    (18.5, 0.833),
    (21.7, 0.500),
    (25.2, 0.0),
    (27.5, 0.652),
    (60.0, 0.0),
    (np.inf, 0.4),
)
#: Scatter of the NS mass for 18.5 <= M_ini < 21.7 (R18 via genulens), and
#: above 60 Msun.
NS_SCATTER = {(18.5, 21.7): (1.60, 0.158), (60.0, np.inf): (1.78, 0.02)}

# ---- numerics -------------------------------------------------------
#: Table step in log10 M (dex).  0.001 dex is 20x finer than the smoothing.
GRID_DEX = 1e-3
#: Gaussian smoothing of every density, in dex (the design's 0.02 dex).
SMOOTH_DEX = 0.02
#: Padding beyond the support so the smoothing has no edge effect.
PAD_DEX = 0.3
#: Relative density floor per population (finiteness only; see docstring).
FLOOR_REL = 1e-20

LIFETIMES_CSV = Path(__file__).resolve().parent / "mist_lifetimes.csv"


@lru_cache(maxsize=1)
def _lifetime_table():
    """(log10 M_ini, [Fe/H] values, log10 lifetime [Gyr] (n_mass, n_feh))."""
    with open(LIFETIMES_CSV) as fh:
        header = [ln for ln in fh if ln.startswith("#")][-1]
    cols = header.lstrip("# ").strip().split(",")
    if cols[0] != "mass":
        raise ValueError(f"{LIFETIMES_CSV}: unexpected header {cols}")
    fehs = np.array([float(c.split("feh")[1]) for c in cols[1:]])
    data = np.loadtxt(LIFETIMES_CSV, delimiter=",", comments="#")
    return np.log10(data[:, 0]), fehs, data[:, 1:]


def log10_lifetime_gyr(logm_ini, feh):
    """log10 MIST lifetime (Gyr) at initial log10 mass(es), one [Fe/H].

    Linear in log M between tracks and in [Fe/H] between grid metallicities.
    Below the grid's 0.1 Msun the lifetime is held at 0.1 Msun's (> 1e3
    Gyr: alive at any Galactic age); above its top mass, at the top track's.
    """
    logm, fehs, table = _lifetime_table()
    if not fehs[0] <= feh <= fehs[-1]:
        raise ValueError(
            f"[Fe/H] = {feh} is outside the MIST lifetime table "
            f"[{fehs[0]}, {fehs[-1]}] ({LIFETIMES_CSV.name})"
        )
    j = min(np.searchsorted(fehs, feh, side="right") - 1, fehs.size - 2)
    f = (feh - fehs[j]) / (fehs[j + 1] - fehs[j])
    col = (1.0 - f) * table[:, j] + f * table[:, j + 1]
    return np.interp(logm_ini, logm, col)


def alive_fraction(population, logm_ini):
    """P_b(alive | M_ini): the fraction of population b's stars of this
    initial mass still on their track today, age marginalized."""
    tau = 10.0 ** log10_lifetime_gyr(logm_ini, POPULATION_FEH[population])
    if population == "thin_disk":
        # CDF of g(T) ~ exp(T / t_sfr) on [0, T_max], at T = tau.
        t = np.clip(tau, 0.0, THIN_AGE_MAX_GYR)
        s = THIN_SFR_TIMESCALE_GYR
        return np.expm1(t / s) / np.expm1(THIN_AGE_MAX_GYR / s)
    if population == "thick_disk":
        return (tau > THICK_AGE_GYR).astype(float)
    if population == "bulge":
        lo = ndtr(-BULGE_AGE_MEAN_GYR / BULGE_AGE_SIGMA_GYR)  # T < 0 cut
        cdf = ndtr((tau - BULGE_AGE_MEAN_GYR) / BULGE_AGE_SIGMA_GYR)
        return np.clip((cdf - lo) / (1.0 - lo), 0.0, 1.0)
    raise ValueError(f"unknown Galactic population {population!r}")


def _p_ns(m_ini):
    p = np.zeros_like(m_ini)
    lo = WD_INITIAL_MASS_MAX
    for hi, val in R18_P_NS:
        p[(m_ini >= lo) & (m_ini < hi)] = val
        lo = hi
    return p


def _ns_mass(m_ini):
    """genulens Mini2Mrem's mean NS mass (R18 fits), M_ini >= 9."""
    d = m_ini - 14.75
    e = m_ini - 2.619
    return np.select(
        [
            m_ini < 13.0,
            m_ini < 15.0,
            m_ini < 17.8,
            m_ini < 18.5,
            m_ini < 21.7,
            m_ini < 27.5,
        ],
        [
            2.24 + 0.508 * d + 0.125 * d**2 + 0.011 * d**3,
            0.123 + 0.112 * m_ini,
            0.996 + 0.0384 * m_ini,
            -0.020 + 0.10 * m_ini,
            np.full_like(m_ini, 1.60),
            3232.29 - 409.429 * e + 17.2867 * e**2 - 0.24315 * e**3,
        ],
        default=1.78,
    )


def _bh_mass(m_ini):
    """genulens Mini2Mrem's BH mass (R18 core-mass fits), M_ini >= 9."""
    m_core = np.where(
        m_ini < 42.21,
        -2.049 + 0.4140 * m_ini,
        5.697 + 7.8598e8 * m_ini**-4.858,
    )
    d = m_ini - 25.97
    m_all = 15.52 - 0.3294 * d - 0.02121 * d**2 + 0.003120 * d**3
    f_ej = np.where(m_ini < 42.21, 0.9, 1.0)
    return f_ej * m_core + (1.0 - f_ej) * m_all


def _gaussian_kernel(sigma_dex, dx):
    half = int(np.ceil(6.0 * sigma_dex / dx))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / sigma_dex) ** 2)
    return k / k.sum()


def build_population_tables(log_imf, x_lo, x_hi):
    """Linear present-day densities of every population on one x grid.

    ``log_imf(x)`` is the (unnormalized) log IMF density in x = log10 M,
    numpy; ``[x_lo, x_hi]`` the union of the stars' logmass supports, which
    is also the range of initial masses born.  Returns ``(x, dens)`` with
    ``dens[pop][kind]`` for kind in ("lum", "WD", "NS", "BH"), each a linear
    density in x on the padded grid ``x`` (smoothed, not floored, not
    normalized).
    """
    n = int(round((x_hi - x_lo + 2 * PAD_DEX) / GRID_DEX)) + 1
    x = x_lo - PAD_DEX + GRID_DEX * np.arange(n)
    imf = np.exp(log_imf(x) - np.max(log_imf(x)))
    born = (x >= x_lo - 0.5 * GRID_DEX) & (x <= x_hi + 0.5 * GRID_DEX)
    m_ini = 10.0**x
    kernel = _gaussian_kernel(SMOOTH_DEX, GRID_DEX)

    def deposit(m_final, weight):
        out = np.zeros(n)
        idx = np.rint((np.log10(m_final) - x[0]) / GRID_DEX).astype(int)
        live = weight > 0
        if np.any(live & ((idx < 0) | (idx >= n))):
            raise ValueError(
                "a remnant mass falls outside the PDMF grid "
                f"[{10 ** x[0]:.3g}, {10 ** x[-1]:.3g}] Msun"
            )
        np.add.at(out, idx[live], weight[live] / GRID_DEX)
        return out

    wd = born & (m_ini >= 0.1) & (m_ini < WD_INITIAL_MASS_MAX)
    hv = born & (m_ini >= WD_INITIAL_MASS_MAX)
    p_ns = _p_ns(m_ini)
    m_ns = _ns_mass(m_ini)
    m_bh = _bh_mass(m_ini)
    bad = (
        hv
        & (p_ns > 0)
        & ~((m_ns >= NS_MASS_RANGE[0]) & (m_ns <= NS_MASS_RANGE[1]))
    )
    if np.any(bad):
        raise ValueError(
            f"R18 NS mass outside {NS_MASS_RANGE} at M_ini = "
            f"{m_ini[bad][:3]} Msun"
        )
    if np.any(hv & (p_ns < 1.0) & (m_bh <= NS_MASS_RANGE[1])):
        raise ValueError("an R18 BH mass below the NS maximum")

    dens = {}
    for pop in POPULATIONS:
        alive = alive_fraction(pop, x)
        dead_w = imf * (1.0 - alive) * GRID_DEX  # objects per grid cell
        wd_d = deposit(
            np.where(wd, 0.109 * m_ini + 0.394, 1.0), np.where(wd, dead_w, 0)
        )
        # NS: deterministic mean mass except in the two scattered bins.
        ns_w = np.where(hv, dead_w * p_ns, 0.0)
        scattered = np.zeros(n, dtype=bool)
        ns_d = np.zeros(n)
        for (lo, hi), (mu, sig) in NS_SCATTER.items():
            sel = hv & (m_ini >= lo) & (m_ini < hi)
            scattered |= sel
            w = ns_w[sel].sum()
            if w > 0:
                # Truncated Gaussian in M on [1.2, 2.1] (genulens redraws
                # out-of-range masses), as a density in x.
                m = 10.0**x
                g = np.exp(-0.5 * ((m - mu) / sig) ** 2) * m
                g[(m < NS_MASS_RANGE[0]) | (m > NS_MASS_RANGE[1])] = 0.0
                ns_d += w * g / (g.sum() * GRID_DEX)
        ns_d += deposit(
            np.where(hv & ~scattered, m_ns, 1.0),
            np.where(scattered, 0.0, ns_w),
        )
        bh_d = deposit(
            np.where(hv, m_bh, 1.0), np.where(hv, dead_w * (1 - p_ns), 0)
        )
        dens[pop] = {
            "lum": np.convolve(imf * alive, kernel, mode="same"),
            "WD": np.convolve(wd_d, kernel, mode="same"),
            "NS": np.convolve(ns_d, kernel, mode="same"),
            "BH": np.convolve(bh_d, kernel, mode="same"),
        }
    return x, dens


class PresentDayMassFunction:
    """The per-population log PDMF tables for one set of stars.

    ``log_imf`` is the IMF's numpy log density in x (unnormalized is fine);
    ``bounds`` the (lower, upper) logmass support of each star (arrays).
    Exposes, per population b (POPULATIONS order):

      ``log_table[b]``    log phi_b on the grid (floored, unnormalized)
      ``dark_frac[b]``    phi_b^dark / phi_b on the grid, in [0, 1]
      ``log_norm[b, k]``  log int phi_b over star k's support
      ``log_lum_frac[b, k]`` log of the LUMINOUS fraction of that integral
    """

    def __init__(self, log_imf, lower, upper):
        lower = np.atleast_1d(np.asarray(lower, dtype=float))
        upper = np.atleast_1d(np.asarray(upper, dtype=float))
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ValueError(
                "the present-day mass function needs finite star.logmass "
                f"bounds; got lower={lower}, upper={upper}"
            )
        self.x, dens = build_population_tables(
            log_imf, float(lower.min()), float(upper.max())
        )
        self.x0 = float(self.x[0])
        self.dens = dens
        n_star = max(lower.size, upper.size)
        lower = np.broadcast_to(lower, (n_star,))
        upper = np.broadcast_to(upper, (n_star,))
        self.log_table = []
        self.dark_frac = []
        self.log_norm = np.empty((len(POPULATIONS), n_star))
        self.log_lum_frac = np.empty((len(POPULATIONS), n_star))
        for b, pop in enumerate(POPULATIONS):
            d = dens[pop]
            dark = d["WD"] + d["NS"] + d["BH"]
            total = d["lum"] + dark
            total = total + FLOOR_REL * total.max()
            self.log_table.append(np.log(total))
            self.dark_frac.append(dark / total)
            for k in range(n_star):
                sel = (self.x >= lower[k]) & (self.x <= upper[k])
                z = np.trapezoid(total[sel], self.x[sel])
                z_lum = np.trapezoid(d["lum"][sel], self.x[sel])
                self.log_norm[b, k] = np.log(z)
                self.log_lum_frac[b, k] = np.log(z_lum / z)

    def log_density_np(self, b, x, k=0):
        """numpy log phi_b(x), normalized over star k's support."""
        return np.interp(x, self.x, self.log_table[b]) - self.log_norm[b, k]

    def type_fractions(self, b, x):
        """{type: P(type | x, population b)} on numpy x, type in lum/WD/NS/BH."""
        d = self.dens[POPULATIONS[b]]
        tot = d["lum"] + d["WD"] + d["NS"] + d["BH"]
        tot = tot + FLOOR_REL * tot.max()
        return {
            t: np.interp(x, self.x, d[t] / tot)
            for t in ("lum", "WD", "NS", "BH")
        }


def interp_table(table, x0, dx, x):
    """Linear interpolation of a uniform-grid numpy ``table`` at tensor x.

    Gradient: (table[i+1] - table[i]) / dx, finite everywhere.  The index
    is clipped (an integer, no gradient) and the fraction is NOT, so x on
    the last node interpolates with fraction 1 exactly; x stays inside the
    grid because the grid pads the logmass support.
    """
    t = (x - x0) / dx
    i = pt.clip(pt.floor(t), 0, table.size - 2).astype("int64")
    f = t - i
    tab = pt.as_tensor_variable(np.asarray(table, dtype=float))
    return tab[i] * (1.0 - f) + tab[i + 1] * f

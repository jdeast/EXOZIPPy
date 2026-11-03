"""Present-day mass function building blocks (review 1.7.2), shared with the
genulens port.

A microlensing lens is whatever object is there TODAY: stars below their
population's turnoff, and the dark remnants (white dwarfs, neutron stars,
black holes) of everything heavier.  ``build_population_tables`` turns an
IMF and, per branch, a turnoff P(alive | M_ini) into the linear present-day
densities in x = log10(M / Msun):

    phi_b(x) = phi_b^lum(x) + phi_b^dark(x)
    phi_b^lum(x)  = IMF(x) * P_b(alive | x)
    phi_b^dark(x) = sum_type int IMF(x_ini) (1 - P_b(alive | x_ini))
                    P(type | M_ini) delta(x - log10 m_type(M_ini)) dx_ini

The turnoffs are the caller's: ``koshimoto21.MassFunction`` passes genulens's
PARSEC ``Minidie`` turnoffs per branch.  (Until review 1.7.6 the removed
3-population model also built its own tables here, from MIST lifetimes with
the age marginalized over each population's age distribution.)  The
initial-final mass relations are genulens's (Koshimoto, Baba & Bennett 2021,
ApJ 917, 78 = K21; its model/mass_function.cpp), which in turn are
PopSyCLE's (Lam et al. 2020, ApJ 889, 31, App. C):

  * white dwarfs (M_ini < 9 Msun): Kalirai et al. 2008,
    M_WD = 0.109 M_ini + 0.394;
  * neutron stars and black holes (M_ini >= 9): Raithel, Sukhbold & Ozel
    2018 (ApJ 856, 35), probabilistic P(NS | M_ini) in mass bins, the NS
    and BH mass fits exactly as genulens codes them (including its 1.60 +/-
    0.158 Msun NS scatter for 18.5 <= M_ini < 21.7 and the f_ej = 0.9 BH
    ejection fraction below 42.21 Msun).

Smoothness: everything is built on a 1e-3 dex grid and the LINEAR densities
are convolved with a Gaussian of ``SMOOTH_DEX`` (0.02 dex, the design's
kernel) before taking the log, so a single-age turnoff step, the white-dwarf
upper edge (1.375 Msun) and the R18 bin edges become erf ramps 0.02 dex wide
rather than cliffs.  The caller floors the log (``FLOOR_REL``, 1e-20 of the
branch's peak) so it stays finite where a branch has no objects at all;
that floor is for finiteness only -- the mixture over branches is what
keeps heavy masses reachable (the young thin disk and the remnants): rope,
not gates.

The tables are numbers; the graph only interpolates them linearly in x
(``interp_table``), so the gradient is a finite difference of adjacent table
entries everywhere: no branch selection, no pt.where, no radicand.
"""

import numpy as np
import pytensor.tensor as pt

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


def build_population_tables(log_imf, x_lo, x_hi, alive):
    """Linear present-day densities of every branch on one x grid.

    ``log_imf(x)`` is the (unnormalized) log IMF density in x = log10 M,
    numpy; ``[x_lo, x_hi]`` the union of the stars' logmass supports, which
    is also the range of initial masses born; ``alive`` ({name: f(x) ->
    P(alive | x)}) the turnoff of each branch (koshimoto21.py passes
    genulens's).  Returns ``(x, dens)`` with ``dens[name][kind]`` for kind
    in ("lum", "WD", "NS", "BH"), each a linear density in x on the padded
    grid ``x`` (smoothed, not floored, not normalized).
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

    alive_fns = alive
    dens = {}
    for pop, alive_fn in alive_fns.items():
        alive = alive_fn(x)
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

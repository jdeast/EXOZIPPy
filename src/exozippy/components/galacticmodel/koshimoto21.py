"""The Koshimoto, Baba & Bennett (2021) Galactic model, ported from genulens.

Provenance: everything in this module is a port of genulens (Koshimoto, Baba
& Bennett 2021, ApJ 917, 78, "K21"; https://github.com/nkoshimoto/genulens,
version 2.0.0a4, commit 2cc5870).  Cite K21 when you use it.  The goal is
numerical EQUIVALENCE with genulens, not an improvement on it: each function
names the genulens function it reproduces, and every constant is genulens's
default (``model/parameters.hpp``, ``simulation/run_context.hpp``,
``simulation/internal/runtime.hpp``), quoted to the digits genulens uses.
Where this port must deviate (smoothing for gradients, a piece genulens
draws by Monte Carlo), the deviation is stated where it happens; the design
and the validation plan are in ``galacticmodel.md`` ("The genulens port").

Units follow genulens: parsecs, km/s, solar masses.  Coordinates are
genulens's Galactocentric frame (``Dlb2xyz``): x from the Galactic center
toward the Sun, y toward l = 90 deg, z toward the north Galactic pole,
origin at Sgr A*.

Validated point by point against genulens's own functions:
``tests/test_koshimoto21.py`` reads ``tests/fixtures/genulens_points.json``
(``scripts/make_genulens_points.py``).

``xp`` is ``numpy`` or ``pytensor.tensor``: one expression, two backends.
"""

from pathlib import Path

import numpy as np

# --- the frame (Initializer::read_model_options / finalize_spatial_model) ---
R0 = 8160.0  # pc, Sun - Galactic center
Z_SUN = 25.0  # pc (constants.hpp kSunZPc)
L_SGRA_DEG = -0.056  # SpatialOptions: Sgr A*'s (l, b), CenSgrA = 1
B_SGRA_DEG = -0.046
PI = 3.1415926535897932385  # genulens's PI

# --- the disk (DensityState; DISK = 2, hDISK = 0) ---------------------------
#: scale heights of the 7 thin-disk age bins and the thick disk (pc)
DISK_ZD = (61.47, 141.84, 224.26, 292.36, 372.85, 440.71, 445.37, 903.12)
#: scale lengths (pc): age bin 0, age bins 1-6, thick disk
DISK_RD = (5000, 2600, 2200)
#: inner plateau: rho(R < Rdbreak) = rho(Rdbreak) (DISK = 2)
DISK_RDBREAK = 5300

# --- the bar (model = 5, addX = 5: E+E_X, K21's best fit) -------------------
BAR_THETA_DEG = 27.0
BAR_X0, BAR_Y0, BAR_Z0 = 930.623146993329, 370.784386649364, 239.547516030578
BAR_C1, BAR_C2 = 1.20011972384328, 4.09326795684828
BAR_RC = 2631.78535429573  # pc, outer cylindrical cutoff
BAR_ZB_C = 1e6  # pc, vertical cutoff (off at the default)
X_X0, X_Y0, X_Z0 = 278.027059842233, 176.318528789193, 286.791941602401
X_C1, X_C2 = 1.3087131258784, 2.21745322869032
X_B_ZX = 1.37774815817195
X_B_ZY = 0.0
X_F = 1.43975636704683
X_RC = 1301.63829617294
CUTOFF_SIGMA = 500.0  # runtime.hpp srob: Gaussian width of both cutoffs

# --- the stellar halo (Robin+03 spheroid; SH = 1) --------------------------
HALO_EPS = 0.76
HALO_ALPHA = 2.44
HALO_AC2 = 500.0 * 500.0  # pc^2, inner core

_TINY = 1e-300

#: component order, genulens's index i (calc_rho_each)
COMPONENTS = (
    "thin1",
    "thin2",
    "thin3",
    "thin4",
    "thin5",
    "thin6",
    "thin7",
    "thick",
    "bar",
    "nsd",
    "halo",
)


def _sgra_offset():
    """genulens's xyzSgrA: Sgr A*'s position in the Sun-centered frame
    (``CoordinateTransformer().distance_l_b_to_xyz(R0, l_SgrA, b_SgrA)``
    with a zero offset), which ``Dlb2xyz`` then subtracts."""
    x, y, z = _dlb_to_xyz_raw(R0, L_SGRA_DEG, B_SGRA_DEG, (0.0, 0.0, 0.0), np)
    return (float(x), float(y), float(z))


def _dlb_to_xyz_raw(d_pc, l_deg, b_deg, offset, xp):
    cosbsun = np.cos(Z_SUN / R0)
    sinbsun = np.sin(Z_SUN / R0)
    b = b_deg * PI / 180.0
    l = l_deg * PI / 180.0  # noqa: E741 -- genulens's name
    xtmp = R0 - d_pc * xp.cos(b) * xp.cos(l)
    ytmp = d_pc * xp.cos(b) * xp.sin(l)
    ztmp = d_pc * xp.sin(b)
    # genulens tilts z by the Sun's height but leaves x untilted -- not a
    # rotation; reproduced as is (the effect is 25/8160 rad).
    return (
        xtmp - offset[0],
        ytmp - offset[1],
        ztmp * cosbsun + xtmp * sinbsun - offset[2],
    )


SGRA_OFFSET = _sgra_offset()


def dlb_to_xyz(d_pc, l_deg, b_deg, xp=np):
    """genulens ``Dlb2xyz`` (CenSgrA = 1): Galactocentric (x, y, z) in pc."""
    return _dlb_to_xyz_raw(d_pc, l_deg, b_deg, SGRA_OFFSET, xp)


def _log_cosh_sq_term(z, zd, xp):
    """log of genulens's thin-disk profile 4 / (e^{2z/zd} + e^{-2z/zd} + 2)
    = sech^2(z/zd), written as log 4 - 2 logaddexp(z/zd, -z/zd) so it never
    overflows (the identity is exact)."""
    u = z / zd
    return np.log(4.0) - 2.0 * xp.logaddexp(u, -u)


def _bar_log_shape(xb, yb, zb, R, xp):
    """log of genulens ``calc_rhoB`` (model = 5, addX = 5): the
    exponential super-ellipsoid bar plus the two-arm X-shape, each with its
    outer Gaussian cylindrical cutoff.

    genulens multiplies by exp(-(R - Rc)^2 / 2 srob^2) only for R >= Rc;
    ``max(R - Rc, 0)^2`` is the same function (continuous, with a continuous
    first derivative), so this is exact, not a smoothing.
    """
    xn = xp.abs(xb / BAR_X0)
    yn = xp.abs(yb / BAR_Y0)
    zn = xp.abs(zb / BAR_Z0)
    # _TINY keeps the chain rule finite on the bar's own axes, where a power
    # sum is exactly 0 and 0^(1/C - 1) * 0 would be inf * 0 (a relative
    # change of 1e-300: below float precision of anything genulens computes)
    rs_cyl = (xn**BAR_C1 + yn**BAR_C1 + _TINY) ** (1.0 / BAR_C1)
    rs = (rs_cyl**BAR_C2 + zn**BAR_C2 + _TINY) ** (1.0 / BAR_C2)
    log_main = -rs - 0.5 * (xp.maximum(R - BAR_RC, 0.0) / CUTOFF_SIGMA) ** 2
    # (the zb_c vertical cutoff is 1e6 pc at the default: never active)

    znx = xp.abs(zb / X_Z0)
    log_arms = []
    for sx in (-1.0, 1.0):
        xn_x = xp.abs((xb + sx * X_B_ZX * zb) / X_X0)
        yn_x = xp.abs((yb - X_B_ZY * zb) / X_Y0)
        rs_x = (
            (xn_x**X_C1 + yn_x**X_C1 + _TINY) ** (X_C2 / X_C1)
            + znx**X_C2
            + _TINY
        ) ** (1.0 / X_C2)
        log_arms.append(-rs_x)
    # b_zY = 0 at the default, so genulens's second pair of arms is off.
    log_x = (
        np.log(X_F)
        + xp.logaddexp(log_arms[0], log_arms[1])
        - 0.5 * (xp.maximum(R - X_RC, 0.0) / CUTOFF_SIGMA) ** 2
    )
    return xp.logaddexp(log_main, log_x)


def log_component_shapes(x, y, z, xp=np):
    """log of genulens ``calc_rho_each``'s ``rhos[i]`` for every
    component: the dimensionless shapes the number densities are
    ``n0[i] * rhos[i]`` of.  Returns a list of 11 entries in ``COMPONENTS``
    order.  The NSD's (index 9) is its tabulated mass density (genulens
    ND = 3, rho0ND = 1), returned everywhere; genulens includes the NSD
    only on sight lines inside its box (``nsd_active``).
    """
    # 1e-12 pc^2: a strictly positive radicand, so the gradient stays finite
    # on the z axis (sqrt'(0) * 0 would be NaN); 1e-16 pc at R = 5 kpc
    R = xp.sqrt(x * x + y * y + 1e-12)
    out = []
    # thin disk age bins 0-6 (sech^2) and the thick disk (exponential).
    # DISK = 2: exp(-R / Rd) beyond Rdbreak, flat inside it -- exactly
    # exp(-max(R, Rdbreak) / Rd).  y0d = exp(-R0 / Rd) normalizes to the
    # Sun's radius.  hDISK = 0: no flaring, so zd/zdtmp = 1.
    r_eff = xp.maximum(R, float(DISK_RDBREAK))
    for i, zd in enumerate(DISK_ZD):
        rd = DISK_RD[0] if i == 0 else DISK_RD[1] if i < 7 else DISK_RD[2]
        log_z = _log_cosh_sq_term(z, zd, xp) if i < 7 else -xp.abs(z) / zd
        out.append(log_z - r_eff / rd + R0 / rd)
    # the bar, in its own frame
    ct = np.cos(BAR_THETA_DEG / 180.0 * PI)
    st = np.sin(BAR_THETA_DEG / 180.0 * PI)
    xb = x * ct + y * st
    yb = -x * st + y * ct
    out.append(_bar_log_shape(xb, yb, z, R, xp))
    out.append(nsd_log_rho(R, z, xp))
    # stellar halo, normalized to 1 at the Sun's R (R0 / a)^alpha, a floored
    # at 500 pc
    z_sh = z / HALO_EPS
    a2 = xp.maximum(R * R + z_sh * z_sh, HALO_AC2)
    out.append(HALO_ALPHA * (np.log(R0) - 0.5 * xp.log(a2)))
    return out


# ---------------------------------------------------------------------------
# Number densities (genulens's start-up normalizations, default model)
# ---------------------------------------------------------------------------
# genulens derives these at start-up from the K21 IMF, the PARSEC turnoff
# table and its population recipe (store_IMF_nBs; the bar's from
# finalize_density_normalization's VVV-box integral).  Quoted from the
# genulens 2.0.0a4 run that wrote tests/fixtures/genulens_points.json
# (``constants``), which tests/test_koshimoto21.py pins them to.  Units:
# stars / pc^3 at the shape's reference point (the Sun's R in the disk
# midplane; the bar center; the Sun for the halo; for the NSD, per
# Msun/pc^3 of its tabulated mass density).
#: all objects (main sequence + giants + white dwarfs): genulens's LENS
#: density (n0d, n0b, n0SH)
N0 = (
    0.0140060036574,
    0.0157592166458,
    0.013386097035,
    0.0118448591911,
    0.0230735150358,
    0.0259764105967,
    0.0553413740781,
    0.00941837606125,
    13.4396677614,
    3.18797927873,
    5.72825867147e-05,
)
#: main sequence + giants only: genulens's SOURCE density (n0MSd, ...)
N0_MS = (
    0.0138588976676,
    0.0152922551676,
    0.0127380816914,
    0.0110785273479,
    0.0212082541026,
    0.0234708547804,
    0.0492007560133,
    0.00803877497343,
    11.9712163015,
    2.87786338496,
    4.77914710526e-05,
)


def log_number_densities(x, y, z, xp=np, luminous_only=False):
    """log number density (stars / pc^3) of each component at (x, y, z) pc:
    ``log n0[i] + log rhos[i]`` (genulens's ``rhoD_L``; ``luminous_only``
    gives ``rhoD_S``'s main-sequence count instead)."""
    n0 = N0_MS if luminous_only else N0
    return [
        np.log(n) + s for n, s in zip(n0, log_component_shapes(x, y, z, xp))
    ]


# ---------------------------------------------------------------------------
# Kinematics (kinematics_runtime.cpp get_vxyz_ran and friends)
# ---------------------------------------------------------------------------
#: genulens KS2MY: km/s/pc -> mas/yr; K = 1000 / KS2MY km/s per (mas/yr kpc)
KS2MY = 210.949526569698696
K_KMS = 1000.0 / KS2MY
#: the Sun's Galactocentric velocity (KinematicsState vxsun, vysun, vzsun)
V_SUN = (-10.0, 243.0, 7.0)

# disk dispersions (read_model_options defaults)
SIG_U10D, SIG_W10D = 42.0, 24.4
SIG_U0TD, SIG_W0TD = 75.0, 49.2
BETA_U, BETA_W = 0.32, 0.77
HSIG_UT, HSIG_WT = 14300.0, 5900.0
HSIG_UTHICK, HSIG_WTHICK = 180000.0, 9400.0
#: median ages (Gyr) of the 7 thin-disk bins and the thick disk (medtauds)
MEDTAUDS = (
    0.075273,
    0.586449,
    1.516357,
    2.516884,
    4.068387,
    6.069263,
    8.656024,
    12.0,
)

# bar kinematics (E+E_X fit)
OMEGA_P = 47.4105844018699  # km/s/kpc
VX_STR, Y0_STR = 43.0364707040617, 406.558313420815
VB = dict(
    x0=858.106595717275,
    y0=3217.04987721548,
    z0=950.690583433628,
    c1=4.25236641149869,
    c2=1.02531652066343,
)
VBZ = dict(
    x0=558.430182718529,
    y0=2003.21703656302,
    z0=3823.20855045157,
    c1=3.71001266000693,
    c2=1.07455173734341,
)
SIG_VB = (151.854794853683, 78.0278905748233, 81.9641955092164)
SIG_VB0 = (63.9939241108675, 75.8180486866697, 71.2336430487113)

# stellar halo
SIG_SH = (131.0, 106.0, 85.0)  # (v_R, v_phi, v_z), km/s

#: Bland-Hawthorn & Gerhard (2016) Fig. 16 rotation curve (Portail+15), as
#: genulens reads it from input_files/Rotcurve_BG16.dat (R pc, v_c km/s),
#: linearly interpolated (getx2y; v_c = 0 beyond the last row).
# fmt: off
ROTCURVE_R = (
    0, 94.59, 162.16, 355.8, 384.68, 413.56, 442.44, 478.54, 514.64, 550.74,
    594.07, 637.39, 680.71, 724.03, 774.57, 832.33, 924.34, 1003.77, 1091.34,
    1227.59, 1441.42, 1660.81, 1964.06, 2260.09, 2548.9, 2844.93, 3148.18,
    3451.43, 3754.68, 3978.51, 4175.68, 4418.94, 4659.47, 5025.44, 5359.18,
    5647.54, 5949.63, 6252.88, 6556.13, 6859.38, 7162.63, 7465.88, 7769.13,
    8072.38, 8375.63, 8678.88, 8982.13, 9195.13, 9633.27, 9863, 9891.05,
    15247.7, 20205.2, 25011,
)
ROTCURVE_V = (
    0, 26.16822, 41.1215, 82.2624, 87.50551, 92.87346, 98.28301, 104.62191,
    111.03709, 117.21647, 123.04215, 128.82622, 134.64001, 140.43596,
    146.30325, 151.87448, 157.76319, 163.50981, 169.11703, 175.03895,
    181.37693, 185.10346, 190.26516, 195.00092, 200.44647, 204.25092,
    207.46099, 210.74834, 214.91548, 219.52495, 224.06542, 228.35254,
    231.5047, 234.17276, 235.78553, 237.45772, 238.44409, 239.49033,
    240.09074, 240.54847, 240.34992, 240.19774, 239.93618, 239.34766,
    238.23009, 237.04757, 235.56278, 234.62386, 232.66982, 231.66169,
    231.49119, 215.8747, 212.48535, 210.19041,
)
_FACA = (
    -0.028476, -1.4518, 12.492, -21.842, 19.130, -10.175, 3.5214, -0.81052,
    0.12311, -0.011851, 0.00065476, -1.5809e-05,
)
# fmt: on

_RC_R = np.array(ROTCURVE_R, dtype=float)
_RC_V = np.array(ROTCURVE_V, dtype=float)
_RC_S = np.diff(_RC_V) / np.diff(_RC_R)  # per-segment slope
_RC_A = _RC_V[:-1] - _RC_S * _RC_R[:-1]  # v_c = a + s R on segment k
#: angular momentum L = v_c R at the knots (monotonic: checked at import)
_RC_L = _RC_V * _RC_R
if not np.all(np.diff(_RC_L) > 0):
    raise ValueError("koshimoto21: rotation curve v_c R must increase")
_LOG_2PI = float(np.log(2.0 * np.pi))


def _segment(xp, knots, value):
    """Index k of the segment [knots[k], knots[k+1]) holding ``value``
    (clamped to the table): an integer, carrying no gradient."""
    inner = knots[1:-1]
    if xp is np:
        return np.searchsorted(inner, value, side="right")
    return xp.sum(
        xp.ge(value[..., None], xp.as_tensor_variable(inner)), axis=-1
    )


def _take(xp, table, k):
    return table[k] if xp is np else xp.as_tensor_variable(table)[k]


def rotation_curve(r_pc, xp=np):
    """genulens getx2y(Rcs, Vcs): v_c(R), linear in R inside the table."""
    k = _segment(xp, _RC_R, r_pc)
    return _take(xp, _RC_A, k) + _take(xp, _RC_S, k) * r_pc


def _guiding_radius(l_target, xp):
    """R_g with v_c(R_g) R_g = L, exactly: on segment k, L = a R + s R^2,
    so R = 2 L / (a + sqrt(a^2 + 4 s L)) (stable for s = 0 and s < 0).
    Returns (R_g, dL/dR_g = a + 2 s R_g)."""
    k = _segment(xp, _RC_L, l_target)
    a = _take(xp, _RC_A, k)
    s = _take(xp, _RC_S, k)
    # strictly positive radicand floor (CLAUDE.md); exact inside the table
    rad = xp.maximum(a * a + 4.0 * s * l_target, 1e-12)
    rg = 2.0 * l_target / (a + xp.sqrt(rad))
    return rg, a + 2.0 * s * rg


def _disk_kinematic_params(i):
    """(sigU0, sigW0, hsigU, hsigW, rd) of disk component i (get_vxyz_ran,
    store_cumuP_Shu); the thin bins at their median ages."""
    tau = MEDTAUDS[i]
    if i < 7:
        sig_u0 = SIG_U10D * ((tau + 0.01) / 10.01) ** BETA_U
        sig_w0 = SIG_W10D * ((tau + 0.01) / 10.01) ** BETA_W
        h_u, h_w = HSIG_UT, HSIG_WT
    else:
        sig_u0, sig_w0 = SIG_U0TD, SIG_W0TD
        h_u, h_w = HSIG_UTHICK, HSIG_WTHICK
    rd = DISK_RD[0] if i == 0 else DISK_RD[1] if i < 7 else DISK_RD[2]
    return sig_u0, sig_w0, h_u, h_w, rd


#: the same, for all 8 disk components at once: shape (8, 1) columns, so
#: one graph broadcasts over (component, star) instead of 8 copies
_DISK = np.array([_disk_kinematic_params(i) for i in range(8)], dtype=float)
_DISK_COLS = tuple(_DISK[:, j : j + 1] for j in range(5))


def _gammaln(x, xp):
    if xp is np:
        from scipy.special import gammaln

        return gammaln(x)
    return xp.gammaln(x)


def _log_shu(fg, rg, vc, sig_u0, h_u, rd, xp):
    """calc_PRRg's expression given f_g, R_g = R f_g and v_c(R_g) / facz;
    the component parameters may be (8, 1) columns (broadcast)."""
    a0 = sig_u0 / vc * np.exp(R0 / h_u)
    q = rd / h_u
    xq = rg / h_u
    poly = 0.0
    for coef in reversed(_FACA):
        poly = poly * xq + coef
    faca = 1.0 - 0.25 * a0**2.04 / q**0.49 * poly
    a = sig_u0 / vc * xp.exp(-(rg - R0) / h_u) * faca
    # genulens returns 0 for c <= 0.5; P ~ (c - 0.5) as c -> 0.5+, so
    # flooring dc = c - 0.5 at 1e-300 makes P ~ e^-690 there (continuous)
    dc = xp.maximum(0.5 / (a * a) - 0.5, 1e-300)
    c = dc + 0.5
    # Sigma(R_g): Sharma+14 eq. 20-21 with SBH13's rising-v_c Table 1
    rgmax = 3.822 * rd / (1.0 + q / 0.524)
    xs = rg / rgmax
    s = 31.53 * xp.exp(-xs / 0.2743) * ((xs / 0.6719) ** 2 - 1.0)
    sig_rg = 0.5 * xp.exp(-rg / rd) / PI - 0.00567 * a0**2.13 * s
    log_gc_exact = c + _gammaln(dc, xp) - np.log(2.0) - dc * xp.log(c)
    log_gc_approx = 0.5 * xp.log(0.5 * PI / xp.maximum(c - 0.913, 1e-3))
    log_gc = xp.where(c < 10.0, log_gc_exact, log_gc_approx)
    return (
        xp.log(xp.maximum(sig_rg, 1e-300))
        + c * (2.0 * xp.log(fg) + 1.0 - fg * fg)
        - log_gc
    )


def _facz(z_pc, xp):
    """Sharma+14 eq. 22: v_c's decline with height, 1 + 0.0374 |z/kpc|^1.34."""
    return 1.0 + 0.0374 * (0.001 * xp.abs(z_pc)) ** 1.34


def log_shu_unnormalized(fg, r_pc, z_pc, i, xp=np):
    """log of genulens ``calc_PRRg``: the Shu distribution of f_g = R_g / R
    at (R, z) for disk component i (Sharma+14 eq. 16-21, Schoenrich &
    Binney 12, Sharma & Bland-Hawthorn 13 eq. 32/39), unnormalized.

    Where genulens returns 0 (f_g <= 0, c <= 0.5, a negative surface
    density) the log is floored at a finite, very negative value instead
    (strictly positive floors, CLAUDE.md).  c's two forms (exact below
    c = 10, genulens's large-c approximation above) are both finite, so
    selecting between them carries no NaN; the jump at c = 10 is
    genulens's (7e-4 relative).  genulens evaluates this at integer table
    nodes; here (R, z) are the star's own.
    """
    sig_u0, _, h_u, _, rd = _disk_kinematic_params(i)
    fg = xp.maximum(fg, 1e-12)
    rg = r_pc * fg
    vc = xp.maximum(rotation_curve(rg, xp) / _facz(z_pc, xp), 1e-12)
    return _log_shu(fg, rg, vc, sig_u0, h_u, rd, xp)


# --- the Shu normalization Z(R, |z|) = int P(f_g) df_g, tabulated ----------
SHU_NORM_FILE = Path(__file__).resolve().parent / "koshimoto21_shu_lognorm.npz"
_SHU_NORM = None


def shu_log_norm_grid(r_grid, z_grid, n_fg=4000):
    """numpy: log int_0^{f_max} P(f_g | R, z) df_g for the 8 disk components
    on (r_grid x z_grid), f_max = R_max(rotation curve) / R, by trapezoid
    on ``n_fg`` points (genulens: a trapezoid over its truncated support at
    the table nodes).  scripts/make_koshimoto21_tables.py writes it."""
    out = np.empty((8, len(r_grid), len(z_grid)))
    for ir, r in enumerate(r_grid):
        r = max(float(r), 1.0)
        fmax = min(_RC_R[-1] / r, 50.0)
        fg = np.linspace(0.0, fmax, n_fg)[1:]
        for iz, z in enumerate(z_grid):
            for i in range(8):
                lp = log_shu_unnormalized(fg, r, float(z), i)
                m = lp.max()
                out[i, ir, iz] = m + np.log(np.trapezoid(np.exp(lp - m), fg))
    return out


def _shu_norm_table():
    global _SHU_NORM
    if _SHU_NORM is None:
        with np.load(SHU_NORM_FILE) as f:
            _SHU_NORM = (f["r"], f["z"], f["log_norm"])
    return _SHU_NORM


def shu_log_norm(r_pc, z_pc, xp=np):
    """log Z(R, |z|) of all 8 disk components, shape (8,) + R's shape:
    bilinear in log Z on the shipped 100-pc table, clamped to its edges
    (interpolation error < 0.3%, largest at R < 1 kpc)."""
    r_g, z_g, table = _shu_norm_table()
    nx, ny = table.shape[1:]
    fx = xp.clip((r_pc - r_g[0]) / (r_g[1] - r_g[0]), 0.0, nx - 1 - 1e-9)
    fy = xp.clip(
        (xp.abs(z_pc) - z_g[0]) / (z_g[1] - z_g[0]), 0.0, ny - 1 - 1e-9
    )
    if xp is np:
        ix = np.floor(fx).astype(int)
        iy = np.floor(fy).astype(int)
        t = table
    else:
        ix = xp.floor(fx).astype("int64")
        iy = xp.floor(fy).astype("int64")
        t = xp.as_tensor_variable(table)
    tx = fx - ix
    ty = fy - iy
    return (
        (1 - tx) * (1 - ty) * t[:, ix, iy]
        + tx * (1 - ty) * t[:, ix + 1, iy]
        + (1 - tx) * ty * t[:, ix, iy + 1]
        + tx * ty * t[:, ix + 1, iy + 1]
    )


def _log_normal(v, sig, xp):
    # a constant sigma's log in numpy: pt.log of a Python float is
    # autocast to float32 (a 3e-7 error)
    log_sig = np.log(sig) if np.isscalar(sig) else xp.log(sig)
    return -0.5 * (v / sig) ** 2 - log_sig - 0.5 * _LOG_2PI


def _bar_sigmas(xb, yb, zb, xp):
    """genulens calc_sigvb: the bar's dispersions in its own frame."""

    def facsig(p):
        xn = xp.abs(xb / p["x0"])
        yn = xp.abs(yb / p["y0"])
        zn = xp.abs(zb / p["z0"])
        rs_cyl = (xn ** p["c1"] + yn ** p["c1"] + _TINY) ** (1.0 / p["c1"])
        rs = (rs_cyl ** p["c2"] + zn ** p["c2"] + _TINY) ** (1.0 / p["c2"])
        return xp.exp(-rs)

    f, fz = facsig(VB), facsig(VBZ)
    return (
        SIG_VB[0] * f + SIG_VB0[0],
        SIG_VB[1] * f + SIG_VB0[1],
        SIG_VB[2] * fz + SIG_VB0[2],
    )


def bar_mean_and_sigma(x, y, z, xp=np):
    """The bar's Gaussian in Galactic Cartesian velocity (get_vxyz_ran's
    bar branch): ``((mx, my, 0), (sig_x, sig_y, sig_z))``.  Mean:
    solid-body rotation Omega_p R plus the streaming along the bar,
    -vx_str sign(y_b) (1 - exp(-(y_b/y0)^2)) -- continuous with a
    continuous derivative at y_b = 0 (sign() needs no gradient).
    Dispersions: calc_sigvb's, projected WITHOUT the x-y covariance."""
    ct = np.cos(BAR_THETA_DEG / 180.0 * PI)
    st = np.sin(BAR_THETA_DEG / 180.0 * PI)
    r = xp.sqrt(x * x + y * y + 1.0)
    xb = x * ct + y * st
    yb = -x * st + y * ct
    sx, sy, sz = _bar_sigmas(xb, yb, z, xp)
    sig_x = xp.sqrt(sx * sx * ct * ct + sy * sy * st * st)
    sig_y = xp.sqrt(sx * sx * st * st + sy * sy * ct * ct)
    vrot = 0.001 * OMEGA_P * r
    stream = -VX_STR * xp.sign(yb) * (1.0 - xp.exp(-((yb / Y0_STR) ** 2)))
    mx = -vrot * y / r + stream * ct
    my = vrot * x / r + stream * st
    return (mx, my, 0.0), (sig_x, sig_y, sz)


def disk_dispersions(r_pc, i, xp=np):
    """(sigU(R), sigW(R)) of disk component i (get_vxyz_ran)."""
    sig_u0, sig_w0, h_u, h_w, _ = _disk_kinematic_params(i)
    return (
        sig_u0 * xp.exp(-(r_pc - R0) / h_u),
        sig_w0 * xp.exp(-(r_pc - R0) / h_w),
    )


#: genulens's Shu table starts at R = 500 pc (KinematicsState RstShu)
SHU_R_MIN = 500.0


def log_vphi_densities(v_phi, r_pc, z_pc, xp=np):
    """Normalized log p(v_phi | R, z) of all 8 disk components, shape (8,)
    + v_phi's shape.  v_phi = v_c(R_g) f_g / facz with f_g ~ Shu, so
    p(v_phi) = P(f_g) facz / L'(R_g) / Z(R, z), R_g = L^-1(v_phi R facz),
    L = v_c R (an exact inverse of the piecewise-linear rotation curve, no
    root finding).  genulens draws f_g from its table at R >= RstShu =
    500 pc (node 0 below it) and then puts the star's own R in
    v_c(f_g R): reproduced by evaluating the DF at max(R, 500) (a clamp,
    continuous)."""
    facz = _facz(z_pc, xp)
    l_target = xp.clip(v_phi * r_pc * facz, 1e-6, _RC_L[-1] * (1.0 - 1e-12))
    rg, dl_drg = _guiding_radius(l_target, xp)
    fg = rg / r_pc
    # broadcast R to v_phi's shape (the normalizer table is indexed by it)
    r_df = xp.maximum(r_pc, SHU_R_MIN) + 0.0 * fg
    rg_df = r_df * fg
    vc = xp.maximum(rotation_curve(rg_df, xp) / facz, 1e-12)
    sig_u0, _, h_u, _, rd = _DISK_COLS
    return (
        _log_shu(fg, rg_df, vc, sig_u0, h_u, rd, xp)
        - shu_log_norm(r_df, z_pc + 0.0 * fg, xp)
        + xp.log(facz)
        - xp.log(dl_drg)
    )


def log_vphi_density(v_phi, r_pc, z_pc, i, xp=np):
    """``log_vphi_densities`` for disk component i alone."""
    return log_vphi_densities(v_phi, r_pc, z_pc, xp)[i]


def log_velocity_densities(x, y, z, vx, vy, vz, xp=np):
    """Normalized log density of the Galactocentric velocity (vx, vy, vz)
    km/s per component at (x, y, z) pc: the density of
    what genulens's ``get_vxyz_ran`` draws, without its escape-speed
    rejection (550 / 600 km/s).

    Disk (thin bins, thick): v_R ~ N(0, sigU(R)), v_z ~ N(0, sigW(R)),
    v_phi from the Shu DF (``log_vphi_densities``), all 8 components in
    one broadcast expression.  Bar: Gaussian in Cartesian
    (``bar_mean_and_sigma``).  Halo: zero-mean Gaussian in (v_R, v_phi,
    v_z).  (v_R, v_phi) is a rotation of (vx, vy): no Jacobian.
    """
    r = xp.sqrt(x * x + y * y + 1.0)  # 1 pc^2 floor: finite on the z axis
    v_r = (x * vx + y * vy) / r
    v_phi = (x * vy - y * vx) / r
    sig_u0, sig_w0, h_u, h_w, _ = _DISK_COLS
    sig_u = sig_u0 * xp.exp(-(r - R0) / h_u)
    sig_w = sig_w0 * xp.exp(-(r - R0) / h_w)
    disk = (
        _log_normal(v_r, sig_u, xp)
        + _log_normal(vz, sig_w, xp)
        + log_vphi_densities(v_phi, r, z, xp)
    )
    out = [disk[i] for i in range(8)]
    (mx, my, _), (sig_x, sig_y, sig_z) = bar_mean_and_sigma(x, y, z, xp)
    out.append(
        _log_normal(vx - mx, sig_x, xp)
        + _log_normal(vy - my, sig_y, xp)
        + _log_normal(vz, sig_z, xp)
    )
    out.append(nsd_log_velocity(r, z, v_r, v_phi, vz, xp))
    out.append(
        _log_normal(v_r, SIG_SH[0], xp)
        + _log_normal(v_phi, SIG_SH[1], xp)
        + _log_normal(vz, SIG_SH[2], xp)
    )
    return out


# ---------------------------------------------------------------------------
# The sight line: genulens's position and velocity frames
# ---------------------------------------------------------------------------
def nsd_active(l_deg, b_deg):
    """genulens switches the NSD on for |l| < 5 and |b| < 2 (sampler.cpp)."""
    l_signed = (l_deg + 180.0) % 360.0 - 180.0
    return abs(l_signed) < 5.0 and abs(b_deg) < 2.0


def line_of_sight_basis(ra_rad, dec_rad):
    """``(l_deg, b_deg, M, v0)``: a star's Galactocentric velocity in
    genulens's frame is ``v0 + M @ (K pm_ra d, K pm_dec d, rv)`` (km/s; pm
    in mas/yr, d in kpc, K = ``K_KMS``, rv in km/s), with genulens's own
    (l, b) unit vectors (event_sampler.cpp: mu_l = (vx sin l + vy cos l) /
    D, mu_b = (vx cos l sin b - vy sin l sin b + vz cos b) / D) and v0 the
    Sun's velocity.  The (pm_ra, pm_dec) -> (mu_l, mu_b) rotation is
    astropy's ICRS -> Galactic at the sight line.  |det M| = 1.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    def pm_lb(pmra, pmdec):
        c = SkyCoord(
            ra=ra_rad * u.rad,
            dec=dec_rad * u.rad,
            pm_ra_cosdec=pmra * u.mas / u.yr,
            pm_dec=pmdec * u.mas / u.yr,
            frame="icrs",
        ).galactic
        return (
            c.pm_l_cosb.to_value(u.mas / u.yr),
            c.pm_b.to_value(u.mas / u.yr),
        )

    g = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad, frame="icrs").galactic
    l_deg, b_deg = float(g.l.deg), float(g.b.deg)
    rot = np.array([pm_lb(1.0, 0.0), pm_lb(0.0, 1.0)]).T
    lr, br = l_deg * PI / 180.0, b_deg * PI / 180.0
    e_l = np.array([np.sin(lr), np.cos(lr), 0.0])
    e_b = np.array(
        [np.cos(lr) * np.sin(br), -np.sin(lr) * np.sin(br), np.cos(br)]
    )
    n = np.array(
        [-np.cos(br) * np.cos(lr), np.cos(br) * np.sin(lr), np.sin(br)]
    )
    m = np.column_stack([np.column_stack([e_l, e_b]) @ rot, n])
    return l_deg, b_deg, m, np.array(V_SUN)


# ---------------------------------------------------------------------------
# The mixture as galacticmodel uses it
# ---------------------------------------------------------------------------
#: the components a sight line outside genulens's NSD box carries, in
#: COMPONENTS order (index into the 11)
BRANCHES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
#: ... and one inside it (|l| < 5, |b| < 2: ``nsd_active``)
BRANCHES_NSD = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#: the three-population label of each component (galacticmodel's
#: thin_disk / thick_disk / bulge), for the inputs not yet ported per
#: branch (the age / [Fe/H] priors): the stellar halo borrows the thick
#: disk's (the oldest population there is), the NSD the bulge's
POPULATION = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 2,
    10: 1,
}
POPULATION_OF_BRANCH = tuple(POPULATION[b] for b in BRANCHES)


def branches_for(l_deg, b_deg):
    """The components genulens simulates on this sight line."""
    return BRANCHES_NSD if nsd_active(l_deg, b_deg) else BRANCHES


# ln-spaced trapezoid points for the position normalizer (galacticmodel's
# _POSITION_QUAD_POINTS rationale: ~0.7 pc steps at the bulge)
_POSITION_QUAD_POINTS = 200001


def log_source_weights(branches, d_pc, selection, xp=np):
    """Per branch, the log weight genulens gives a SOURCE at d_pc relative
    to the generic volume element D^2 (los_density_grid.cpp rhoD_S):
    ``selection`` is ("sqrt_distance",) -- sqrt(D / 8 kpc), no D^2 -- or
    ("luminosity_function", i_min, i_max, ai0, h) -- fLF_i(D)."""
    if selection[0] == "sqrt_distance":
        w = log_source_weight_sqrt(d_pc, xp)
        return [w for _ in branches]
    _, i_min, i_max, ai0, h = selection
    return [
        log_detect_fraction(b, d_pc, i_min, i_max, ai0, h, xp)
        for b in branches
    ]


def log_total_number_density(d_pc, l_deg, b_deg, selection=None):
    """numpy: log sum over the sight line's branches of n_b at d_pc; with a
    source ``selection``, genulens's source weight n_MS,b w_S,b instead
    (still per pc^3 of the D^2 volume element, which w_S compensates)."""
    d = np.maximum(np.asarray(d_pc, dtype=float), 1e-3)
    x, y, z = dlb_to_xyz(d, l_deg, b_deg)
    br = branches_for(l_deg, b_deg)
    dens = log_number_densities(x, y, z, luminous_only=selection is not None)
    terms = [dens[i] for i in br]
    if selection is not None:
        terms = [
            t + w for t, w in zip(terms, log_source_weights(br, d, selection))
        ]
    return np.logaddexp.reduce(np.stack(terms), axis=0)


def log_position_norm(lower_pc, upper_pc, l_deg, b_deg, selection=None):
    """numpy: log int_lower^upper sum_b n_b(d) d^2 dd along the sight line
    (trapezoid in ln d, integrand n d^3, peak factored out)."""
    lnd = np.linspace(
        np.log(float(lower_pc)), np.log(float(upper_pc)), _POSITION_QUAD_POINTS
    )
    f = (
        log_total_number_density(np.exp(lnd), l_deg, b_deg, selection)
        + 3.0 * lnd
    )
    peak = float(np.max(f))
    return float(np.log(np.trapezoid(np.exp(f - peak), lnd)) + peak)


# ---------------------------------------------------------------------------
# The mass function: K21's IMF, genulens's turnoffs, remnants and kicks
# ---------------------------------------------------------------------------
#: K21's E+E_X best-fit broken power law dN/dM ~ M^alpha (IMFParameters):
#: break masses (Msun) and slopes, on [IMF_MIN, IMF_MAX]
IMF_M1, IMF_M2 = 0.859770466578045, 0.08
IMF_ALPHA = (-2.32279457078378, -1.13449983242887, -0.175862190587576)
IMF_MIN, IMF_MAX = 0.001, 120.0
_LN10 = float(np.log(10.0))


def log_imf(x, xp=np):
    """log dN/dlog10(M) of K21's IMF (unnormalized), x = log10(M / Msun):
    continuous and piecewise linear in x with DECREASING slopes (alpha + 1
    = 0.824, -0.134, -1.323 per ln-mass), so it is exactly the minimum of
    its three lines -- no branch selection.  genulens has no objects
    outside [0.001, 120] Msun; ``pdmf`` tables put a floor there."""
    a3, a2, a1 = IMF_ALPHA[2] + 1.0, IMF_ALPHA[1] + 1.0, IMF_ALPHA[0] + 1.0
    x2, x1 = np.log10(IMF_M2), np.log10(IMF_M1)
    low = a3 * _LN10 * (x - x2)
    mid = a2 * _LN10 * (x - x2)
    high = a2 * _LN10 * (x1 - x2) + a1 * _LN10 * (x - x1)
    return xp.minimum(xp.minimum(low, mid), high)


def _log_imf_table(x):
    """numpy log IMF with genulens's [0.001, 120] Msun support (a 700-nat
    drop outside it: the PDMF tables floor it)."""
    inside = (x >= np.log10(IMF_MIN)) & (x <= np.log10(IMF_MAX))
    return np.where(inside, log_imf(x), log_imf(x) - 700.0)


#: genulens's turnoff mass (largest initial mass alive, Msun) per branch,
#: from input_files/Minidie.dat (PARSEC v1.2S via CMD 3.3) at the age
#: genulens uses: each thin-disk bin at its median age (medtauds, nearest
#: 50-Myr row), the thick disk at 12 Gyr, the halo at 14 Gyr
#: (event_sampler.cpp).  The bar's age is drawn per event (below).
MINIDIE = {
    0: 5.3151664734,
    1: 2.7425796986,
    2: 1.9923980236,
    3: 1.5829007626,
    4: 1.3345959187,
    5: 1.1716736555,
    6: 1.0536190271,
    7: 0.8567695618,
    10: 0.7761976123,
}
#: the bar's "B" rows of Minidie.dat: age (Gyr) -> turnoff (Msun)
# fmt: off
MINIDIE_BAR = (
    (1.0, 2.3065266609), (1.5, 1.9999334812), (2.0, 1.7264221907),
    (2.5, 1.5922399759), (3.0, 1.4965139627), (3.5, 1.4142400026),
    (4.0, 1.3522768021), (4.5, 1.3033635616), (5.0, 1.2618246078),
    (5.5, 1.2259132862), (6.0, 1.1946581602), (6.5, 1.168975234),
    (7.0, 1.1455239058), (7.5, 1.1236175299), (8.0, 1.1036301851),
    (8.5, 1.0850403309), (9.0, 1.0678446293), (9.5, 1.0517697334),
    (10.0, 1.0366524458), (10.5, 1.0225354433), (11.0, 1.0092254877),
    (11.5, 0.9966011047), (12.0, 0.9846860766), (12.5, 0.9733356237),
    (13.0, 0.9625536203), (13.5, 0.952288866), (14.0, 0.9427683949),
)
# fmt: on
BAR_AGE_MEAN, BAR_AGE_SIGMA = 9.0, 1.0  # StellarPopulationState mageB, sageB


def _bar_age_weights():
    """genulens draws the bar lens's age tau ~ N(9, 1) and reads Minidie at
    the row (int)(2 tau + 0.5) * 50 (10-Myr units): the node a_k gets
    tau in [a_k - 0.25, a_k + 0.25).  Exact weights of that finite mixture
    (renormalized over the table: the mass beyond it is < 1e-15)."""
    from scipy.special import ndtr

    ages = np.array([a for a, _ in MINIDIE_BAR])
    w = ndtr((ages + 0.25 - BAR_AGE_MEAN) / BAR_AGE_SIGMA) - ndtr(
        (ages - 0.25 - BAR_AGE_MEAN) / BAR_AGE_SIGMA
    )
    return w / w.sum()


def alive_fraction(branch, x):
    """P(alive | log10 M_ini = x) in branch ``branch`` (COMPONENTS index):
    genulens's ``M_ini < Minidie`` -- a step for a fixed-age branch, the
    exact age mixture for the bar (the PDMF tables smooth both by
    pdmf.SMOOTH_DEX)."""
    m = 10.0 ** np.asarray(x, dtype=float)
    if branch in (8, 9):
        table = MINIDIE_BAR if branch == 8 else MINIDIE_NSD
        w = _bar_age_weights() if branch == 8 else _nsd_age_weights()
        die = np.array([d for _, d in table])
        return np.sum(w[:, None] * (m[None, :] < die[:, None]), axis=0)
    return (m < MINIDIE[branch]).astype(float)


#: natal kicks (BhKickParameters): fixed speed, isotropic direction, added
#: to the LENS's velocity (event_sampler.cpp)
KICK_NS, KICK_BH = 350.0, 100.0


class MassFunction:
    """K21's present-day mass function per branch, for a set of stars.

    Per branch b (BRANCHES order) and star k: ``log_table[g][b]`` (g in
    ``GROUPS``) the log density in x of the objects that keep the branch's
    velocity distribution ("still": luminous stars and white dwarfs),
    neutron stars and black holes (each kicked) -- floored, unnormalized --
    and ``log_norm[b, k]`` the log of the IMF's integral over star k's
    logmass support (genulens normalizes the IMF and turns each dead star
    into ONE remnant, or with ``remnants=False`` discards it -- REMNANT 0
    -- so the branch weight is n0 * F_alive there, exactly as genulens's
    rejection makes it).  ``dark_frac[b]`` is P(dark | x) on the grid.
    Built with pdmf.build_population_tables (the 0.02 dex smoothing and the
    R18 / K08 initial-final mass relations are shared).
    """

    GROUPS = ("still", "NS", "BH")

    def __init__(self, lower, upper, remnants=True, branches=BRANCHES):
        from . import pdmf

        lower = np.atleast_1d(np.asarray(lower, dtype=float))
        upper = np.atleast_1d(np.asarray(upper, dtype=float))
        self.branches = tuple(branches)
        alive = {b: (lambda xx, b=b: alive_fraction(b, xx)) for b in branches}
        self.x, dens = pdmf.build_population_tables(
            _log_imf_table, float(lower.min()), float(upper.max()), alive=alive
        )
        self.x0 = float(self.x[0])
        n_star = max(lower.size, upper.size)
        lower = np.broadcast_to(lower, (n_star,))
        upper = np.broadcast_to(upper, (n_star,))
        imf = np.exp(_log_imf_table(self.x) - np.max(_log_imf_table(self.x)))
        self.log_imf_peak = float(np.max(_log_imf_table(self.x)))
        self.log_table = {g: [] for g in self.GROUPS}
        self.dark_frac = []
        self.log_norm = np.empty((len(branches), n_star))
        #: luminous objects only (a survey-selected SOURCE: genulens's n0MS
        #: count), normalized over the luminous objects in the support
        self.log_lum = []
        #: white dwarfs alone (with binaries on, the luminous part of
        #: "still" is replaced by the binary-lens density)
        self.log_wd = []
        self.log_norm_lum = np.empty((len(branches), n_star))
        for j, b in enumerate(branches):
            d = dens[b]
            if remnants:
                parts = {
                    "still": d["lum"] + d["WD"],
                    "NS": d["NS"],
                    "BH": d["BH"],
                }
            else:
                parts = {
                    "still": d["lum"],
                    "NS": 0 * d["NS"],
                    "BH": 0 * d["BH"],
                }
            total = parts["still"] + parts["NS"] + parts["BH"]
            floor = pdmf.FLOOR_REL * total.max()
            for g in self.GROUPS:
                self.log_table[g].append(np.log(parts[g] + floor))
            dark = total - d["lum"] if remnants else 0.0 * total
            self.dark_frac.append(dark / (total + floor))
            self.log_lum.append(np.log(d["lum"] + floor))
            self.log_wd.append(np.log(d["WD"] + floor))
            for k in range(n_star):
                sel = (self.x >= lower[k]) & (self.x <= upper[k])
                self.log_norm[j, k] = np.log(
                    np.trapezoid(imf[sel], self.x[sel])
                )
                self.log_norm_lum[j, k] = np.log(
                    np.trapezoid(d["lum"][sel], self.x[sel])
                )


def log_kicked_velocity(vx, vy, vz, mean, sigma_iso, kick, xp=np):
    """log density of v = v_base + kick * n, n isotropic, for an isotropic
    Gaussian base N(mean, sigma_iso^2 I) -- closed form:

        p(v) = (2 pi s^2)^(-3/2) s^2 / (2 k r) [e^{-(r-k)^2/2s^2}
                                               - e^{-(r+k)^2/2s^2}],

    r = |v - mean|, written as -(r-k)^2/2s^2 + log(-expm1(-2rk/s^2)) so it
    is finite for every r > 0 (r has a strictly positive floor).  genulens
    adds the kick to an anisotropic base (the disk's Shu DF, the bar's
    Gaussian); the port replaces the base by an isotropic Gaussian of the
    same mean and det-preserving dispersion (galacticmodel.md)."""
    dx = vx - mean[0]
    dy = vy - mean[1]
    dz = vz - mean[2]
    r = xp.sqrt(dx * dx + dy * dy + dz * dz + 1e-6)
    s2 = sigma_iso * sigma_iso
    # a constant sigma's log in numpy (pt.log of a float autocasts to
    # float32)
    log_s2 = np.log(s2) if np.isscalar(s2) else xp.log(s2)
    return (
        -1.5 * (np.log(2.0 * PI) + log_s2)
        + log_s2
        - np.log(2.0 * kick)
        - xp.log(r)
        - (r - kick) ** 2 / (2.0 * s2)
        + xp.log(-xp.expm1(-2.0 * r * kick / s2))
    )


def kick_bases(x, y, z, xp=np, branches=BRANCHES):
    """(mean, sigma_iso) per branch (BRANCHES order) of the isotropic
    Gaussian a kicked remnant's velocity is convolved from: disk -- mean
    (v_R, v_phi, v_z) = (0, v_c(R) / facz, 0), dispersions (sigU, sigU /
    sqrt 2, sigW) (the epicyclic ratio of a flat rotation curve); bar --
    its Gaussian; halo -- zero mean, its dispersions.  sigma_iso = the
    geometric mean of the three."""
    r = xp.sqrt(x * x + y * y + 1.0)
    vc = rotation_curve(r, xp) / _facz(z, xp)
    disk_mean = (-vc * y / r, vc * x / r, 0.0)
    out = []
    for i in range(8):
        su, sw = disk_dispersions(r, i, xp)
        out.append((disk_mean, (su * su * sw / np.sqrt(2.0)) ** (1.0 / 3.0)))
    (mx, my, _), (sx, sy, sz) = bar_mean_and_sigma(x, y, z, xp)
    out.append(((mx, my, 0.0), (sx * sy * sz) ** (1.0 / 3.0)))
    m_phi, s_phi, s_r, s_z, _ = nsd_moments(r, z, xp)
    out.append(
        (
            (-m_phi * y / r, m_phi * x / r, 0.0),
            (s_phi * s_r * s_z) ** (1.0 / 3.0),
        )
    )
    out.append(((0.0, 0.0, 0.0), float(np.prod(SIG_SH) ** (1.0 / 3.0))))
    return [out[i] for i in branches]


# ---------------------------------------------------------------------------
# The nuclear stellar disk (genulens ND = 3: Sormani et al. 2021 moments)
# ---------------------------------------------------------------------------
NSD_FILE = Path(__file__).resolve().parent / "koshimoto21_nsd.npz"
_NSD = None
#: genulens's NSD support: R <= RenND - 30, |z| <= zenND - 20 (pc)
NSD_R_EDGE, NSD_Z_EDGE = 970.0, 380.0
#: width (pc) of the Gaussian roll-off the port puts beyond that edge, where
#: genulens's density drops to zero (the table there is ~1e-9 of central)
NSD_EDGE_SIGMA = 5.0
#: the NSD's "N" rows of Minidie.dat (age Gyr -> turnoff Msun)
MINIDIE_NSD = (
    (5.0, 1.2837916613),
    (6.0, 1.214597702),
    (7.0, 1.1653949022),
    (8.0, 1.1238201857),
    (9.0, 1.0875971317),
    (10.0, 1.0560716391),
)
NSD_AGE_MEAN, NSD_AGE_SIGMA = 7.0, 1.0  # StellarPopulationState mageND, sageND


def _nsd_table():
    global _NSD
    if _NSD is None:
        with np.load(NSD_FILE) as f:
            _NSD = {k: f[k] for k in f.files}
    return _NSD


def _nsd_age_weights():
    """genulens: tau ~ N(7, 1), row (int)(100 (tau + 0.5)) clamped to the
    table, i.e. row k (age 5 + k) for tau in [4.5 + k, 5.5 + k), the end
    rows taking the tails (event_sampler.cpp)."""
    from scipy.special import ndtr

    ages = np.array([a for a, _ in MINIDIE_NSD])
    lo = np.r_[-np.inf, ages[1:] - 0.5]
    hi = np.r_[ages[:-1] + 0.5, np.inf]
    return ndtr((hi - NSD_AGE_MEAN) / NSD_AGE_SIGMA) - ndtr(
        (lo - NSD_AGE_MEAN) / NSD_AGE_SIGMA
    )


def _nsd_bilinear(name, r, z, xp):
    """genulens interp_xy on the NSD table (|z| rows, R columns), clamped to
    the table."""
    t = _nsd_table()
    rg, zg, tab = t["r"], t["z"], t[name]
    fz = xp.clip(xp.abs(z) / (zg[1] - zg[0]), 0.0, zg.size - 1 - 1e-9)
    fr = xp.clip(r / (rg[1] - rg[0]), 0.0, rg.size - 1 - 1e-9)
    if xp is np:
        iz, ir = np.floor(fz).astype(int), np.floor(fr).astype(int)
        tt = tab
    else:
        iz, ir = xp.floor(fz).astype("int64"), xp.floor(fr).astype("int64")
        tt = xp.as_tensor_variable(tab)
    a, b = fz - iz, fr - ir
    return (
        (1 - a) * (1 - b) * tt[iz, ir]
        + (1 - a) * b * tt[iz, ir + 1]
        + a * (1 - b) * tt[iz + 1, ir]
        + a * b * tt[iz + 1, ir + 1]
    )


def nsd_log_rho(r, z, xp=np):
    """ln of the NSD's tabulated mass density (Msun/pc^3), genulens's
    rhos[9]; beyond genulens's edge, where it is 0, a Gaussian roll-off of
    width NSD_EDGE_SIGMA keeps it finite and differentiable."""
    edge = (xp.maximum(r - NSD_R_EDGE, 0.0) / NSD_EDGE_SIGMA) ** 2 + (
        xp.maximum(xp.abs(z) - NSD_Z_EDGE, 0.0) / NSD_EDGE_SIGMA
    ) ** 2
    return _LN10 * _nsd_bilinear("log10_rho", r, z, xp) - 0.5 * edge


def nsd_moments(r, z, xp=np):
    """(mean v_phi, sig_phi, sig_R, sig_z, corr(R, z)) of the NSD at
    (R, z), bilinear as genulens interpolates them (log sigmas)."""
    return (
        _nsd_bilinear("vphi", r, z, xp),
        10.0 ** _nsd_bilinear("log10_sig_phi", r, z, xp),
        10.0 ** _nsd_bilinear("log10_sig_r", r, z, xp),
        10.0 ** _nsd_bilinear("log10_sig_z", r, z, xp),
        _nsd_bilinear("cor_rz", r, z, xp),
    )


def nsd_log_velocity(r, z, v_r, v_phi, v_z, xp=np):
    """genulens's NSD velocity: v_phi ~ N(m, sig_phi), v_R ~ N(0, sig_R),
    v_z | v_R ~ N(rho sig_z / sig_R v_R, sig_z sqrt(1 - rho^2))."""
    m, s_phi, s_r, s_z, rho = nsd_moments(r, z, xp)
    return (
        _log_normal(v_phi - m, s_phi, xp)
        + _log_normal(v_r, s_r, xp)
        + _log_normal(
            v_z - rho * s_z / s_r * v_r,
            s_z * xp.sqrt(xp.maximum(1.0 - rho * rho, 1e-12)),
            xp,
        )
    )


# ---------------------------------------------------------------------------
# Source weighting (los_density_grid.cpp: rhoD_S)
# ---------------------------------------------------------------------------
LF_FILE = Path(__file__).resolve().parent / "koshimoto21_lf.npz"
_LF = None
#: genulens's default red-clump distance modulus (Nataf+16 as genulens
#: codes it) and dust scale height
DUST_SCALE_HEIGHT = 164.0


def red_clump_distance_modulus(l_deg, b_deg):
    l_signed = (l_deg + 180.0) % 360.0 - 180.0
    return 14.3955 - 0.0239 * l_signed + 0.0122 * abs(b_deg) + 0.128


def log_source_weight_sqrt(d_pc, xp=np):
    """genulens's default (gammaDs = 0.5, no luminosity function): a source
    at distance D weighs n_MS(D) sqrt(D / 8 kpc) -- with NO D^2 volume
    element.  Returned as the log weight relative to the D^2 the
    galactic prior's volume element already carries."""
    return 0.5 * xp.log(d_pc / 8000.0) - 2.0 * xp.log(d_pc)


def _lf_table():
    global _LF
    if _LF is None:
        with np.load(LF_FILE) as f:
            _LF = (f["mi"], f["cumu"])
    return _LF


def _lf_cdf(i, mag, xp):
    """genulens fLF_detect's interpolation of component i's cumulative LF
    at absolute magnitude ``mag`` (clamped to the table)."""
    mi, cumu = _lf_table()
    d = (mi[-1] - mi[0]) / (mi.size - 1)
    f = xp.clip((mag - mi[0]) / d, 0.0, mi.size - 1 - 1e-9)
    if xp is np:
        k = np.floor(f).astype(int)
        t = cumu[i]
    else:
        k = xp.floor(f).astype("int64")
        t = xp.as_tensor_variable(cumu[i])
    frac = f - k
    return t[k + 1] * frac + t[k] * (1.0 - frac)


def log_detect_fraction(i, d_pc, i_min, i_max, ai0, h_pc, xp=np):
    """log fLF_i(D): the fraction of component i's main-sequence stars with
    apparent I in [i_min, i_max] at distance D (genulens fLF_detect), with
    genulens's extinction A_I(D) = ai0 (1 - exp(-D / h)) and its D + 0.1
    pc distance modulus.  Floored strictly positive."""
    ext = ai0 * (1.0 - xp.exp(-d_pc / h_pc)) + 5.0 * xp.log10(
        0.1 * (d_pc + 0.1)
    )
    frac = _lf_cdf(i, i_max - ext, xp) - _lf_cdf(i, i_min - ext, xp)
    return xp.log(xp.maximum(frac, 1e-300))


def extinction_scale(
    l_deg, b_deg, ai_rc, dm_rc=None, h_dust=DUST_SCALE_HEIGHT
):
    """(ai0, h): genulens's ExponentialDustExtinction for the sight line --
    h = h_dust / (|sin b| + 1e-4), A_I(D) = ai0 (1 - exp(-D/h)) with
    A_I(D_rc) = ai_rc at the red clump's distance."""
    if dm_rc is None:
        dm_rc = red_clump_distance_modulus(l_deg, b_deg)
    h = h_dust / (abs(np.sin(b_deg / 180.0 * PI)) + 0.0001)
    d_rc = 10.0 ** (0.2 * dm_rc) * 10.0
    return ai_rc / (1.0 - np.exp(-d_rc / h)), h


# ---------------------------------------------------------------------------
# Binary lenses (genulens BINARY 1: event_sampler.cpp "Binary system",
# model/mass_function.cpp BinaryLensSampler)
# ---------------------------------------------------------------------------
BINARY_FILE = Path(__file__).resolve().parent / "koshimoto21_binary.npz"
_BINARY = None
#: the binary table's grids: x_eff = log10 of the lens mass the event sees,
#: r0 = sqrt(kappa pi_rel) D_L in AU / sqrt(Msun) (the physical Einstein
#: radius of a 1 Msun lens)
BINARY_X = np.arange(-3.0, 1.1 + 1e-9, 0.01)
BINARY_LOG_R0 = np.arange(-1.5, 2.5 + 1e-9, 0.05)
KAPPA = 8.1439  # genulens kKappa, mas / Msun


def _binary_draws(n, rng):
    """Monte Carlo of genulens's binary-lens recipe for a luminous primary
    of mass M (sampled uniformly in log10 M over the alive range, so each
    branch reweights by its IMF x alive): per draw, the case weights (no
    companion; companion drawn below / above the mean separation), the
    companion's q, the projected separation and the blending thresholds,
    for both choices of which body is "lens 1" (genulens picks body 1 with
    probability sqrt(M1) / (sqrt(M1) + sqrt(M2)); weights here)."""
    x = rng.uniform(np.log10(IMF_MIN), np.log10(max(MINIDIE.values())), n)
    m = 10.0**x
    mult = np.minimum(0.196 + 0.255 * m, 1.0)
    out = []
    for coeff in (-1.0, 1.0):
        if coeff < 0:
            gamma = np.clip(1.16 - 2.79 * np.log10(m), 0.0, 4.0)
        else:
            gamma = np.where(
                m >= 0.344, 0.0, np.clip(-3.09 - 6.67 * np.log10(m), 0.0, 4.0)
            )
        t = 0.1 ** (gamma + 1.0)
        q = ((1.0 - t) * rng.uniform(size=n) + t) ** (1.0 / (gamma + 1.0))
        mla = np.clip(0.57 + 1.02 * m, 0.6, 1.7)
        sla = np.clip(1.61 + 1.15 * np.log10(m), 0.3, 1.8)
        log_a = mla + coeff * np.abs(rng.standard_normal(n)) * sla
        a_perp = np.sqrt(1.0 - rng.uniform(size=n) ** 2) * 10.0**log_a
        u0 = rng.uniform(size=n)
        p1 = np.sqrt(m) / (np.sqrt(m) + np.sqrt(m * q))
        for first_is_primary in (True, False):
            m1 = m if first_is_primary else m * q
            m2 = m * q if first_is_primary else m
            qt = m2 / m1
            tmp = qt / u0
            f_l = (np.sqrt(tmp + 1.0) + np.sqrt(tmp)) * np.sqrt(m1)
            tmp2 = np.where(qt > 1.0, 1.0 / qt / u0, tmp)
            f_s = (
                (np.sqrt(tmp2 + 1.0) - np.sqrt(tmp2))
                * np.sqrt(m1 + m2)
                * np.where(
                    qt > 1.0, np.sqrt(1.0 + 1.0 / qt), np.sqrt(1.0 + qt)
                )
            )
            w = 0.5 * mult * (p1 if first_is_primary else 1.0 - p1)
            out.append((w, a_perp, f_l, f_s, np.log10(m1), np.log10(m1 + m2)))
    return x, 1.0 - mult, out


def binary_tables(branches=BRANCHES_NSD, n=2_000_000, seed=20211):
    """numpy: per branch, log of genulens's EFFECTIVE luminous-lens density
    in x_eff on (BINARY_X x BINARY_LOG_R0): IMF x alive primaries, single
    (prob 1 - mult) or with a companion that is either resolved WIDE (the
    event sees body 1 alone: a_perp > f_L r0), blended CLOSE (the total
    mass: a_perp < f_S r0) or REJECTED (a detectable binary; dropped, as
    genulens's retry drops it).  Same unit as MassFunction's tables
    (IMF / its peak), 0.02-dex smoothing.  scripts/make_koshimoto21_tables.py
    writes it (Monte Carlo, ``n`` primaries: a few-% noise in the far
    tails, below the smoothing in the bulk)."""
    from .pdmf import FLOOR_REL, SMOOTH_DEX, _gaussian_kernel

    rng = np.random.default_rng(seed)
    x, w_single, pairs = _binary_draws(n, rng)
    dx = BINARY_X[1] - BINARY_X[0]
    edges = np.r_[BINARY_X - 0.5 * dx, BINARY_X[-1] + 0.5 * dx]
    imf = np.exp(log_imf(x) - log_imf(np.log10(IMF_M2)))
    span = np.log10(max(MINIDIE.values())) - np.log10(IMF_MIN)
    kernel = _gaussian_kernel(SMOOTH_DEX, dx)
    out = np.empty((len(branches), BINARY_X.size, BINARY_LOG_R0.size))
    for j, b in enumerate(branches):
        base = imf * alive_fraction(b, x) * span / n  # density per sample
        single = np.histogram(x, edges, weights=base * w_single)[0] / dx
        for k, lr0 in enumerate(BINARY_LOG_R0):
            r0 = 10.0**lr0
            h = single.copy()
            for w, a_perp, f_l, f_s, x1, x12 in pairs:
                wide = a_perp > f_l * r0
                close = a_perp < f_s * r0
                h += (
                    np.histogram(x1[wide], edges, weights=(base * w)[wide])[0]
                    / dx
                )
                h += (
                    np.histogram(x12[close], edges, weights=(base * w)[close])[
                        0
                    ]
                    / dx
                )
            h = np.convolve(h, kernel, mode="same")
            out[j, :, k] = np.log(h + FLOOR_REL * h.max())
    return out


def _binary_table():
    global _BINARY
    if _BINARY is None:
        with np.load(BINARY_FILE) as f:
            _BINARY = {
                "branches": tuple(int(b) for b in f["branches"]),
                "log_dens": f["log_dens"],
            }
    return _BINARY


def log_r0(d_l_pc, d_s_pc, xp=np):
    """log10 of sqrt(kappa pi_rel) D_L [AU / sqrt(Msun)]: the physical
    Einstein radius of a 1 Msun lens, the scale genulens's blending
    thresholds compare a binary's projected separation to.  pi_rel is
    floored strictly positive (the source-behind-lens barrier is soft)."""
    pirel = xp.maximum(1000.0 / d_l_pc - 1000.0 / d_s_pc, 1e-6)
    return 0.5 * xp.log10(KAPPA * pirel) + xp.log10(d_l_pc / 1000.0)


def log_binary_lens_density(branch, x_eff, lr0, xp=np):
    """Bilinear in (x_eff, log r0) on the shipped table, clamped."""
    tab = _binary_table()
    t = tab["log_dens"][tab["branches"].index(branch)]
    dx = BINARY_X[1] - BINARY_X[0]
    dr = BINARY_LOG_R0[1] - BINARY_LOG_R0[0]
    fx = xp.clip((x_eff - BINARY_X[0]) / dx, 0.0, BINARY_X.size - 1 - 1e-9)
    fr = xp.clip(
        (lr0 - BINARY_LOG_R0[0]) / dr, 0.0, BINARY_LOG_R0.size - 1 - 1e-9
    )
    if xp is np:
        ix, ir = np.floor(fx).astype(int), np.floor(fr).astype(int)
        tt = t
    else:
        ix, ir = xp.floor(fx).astype("int64"), xp.floor(fr).astype("int64")
        tt = xp.as_tensor_variable(t)
    a, b = fx - ix, fr - ir
    return (
        (1 - a) * (1 - b) * tt[ix, ir]
        + a * (1 - b) * tt[ix + 1, ir]
        + (1 - a) * b * tt[ix, ir + 1]
        + a * b * tt[ix + 1, ir + 1]
    )

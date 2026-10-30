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
    """log of genulens ``calc_rho_each``'s ``rhos[i]`` for every component
    but the NSD (index 9, tabulated): the dimensionless shapes the number
    densities are ``n0[i] * rhos[i]`` of.  Returns a list of 11 entries in
    ``COMPONENTS`` order, with ``None`` at index 9.
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
    out.append(None)  # NSD: tabulated (Sormani+21 moments)
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
# midplane; the bar center; the Sun for the halo).  NSD (index 9) is
# tabulated and not yet ported.
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
    None,
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
    None,
    4.77914710526e-05,
)


def log_number_densities(x, y, z, xp=np, luminous_only=False):
    """log number density (stars / pc^3) of each component at (x, y, z) pc:
    ``log n0[i] + log rhos[i]`` (genulens's ``rhoD_L``; ``luminous_only``
    gives ``rhoD_S``'s main-sequence count instead).  NSD entry None."""
    n0 = N0_MS if luminous_only else N0
    return [
        None if s is None else np.log(n) + s
        for n, s in zip(n0, log_component_shapes(x, y, z, xp))
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
    km/s per component (NSD entry None) at (x, y, z) pc: the density of
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
    out.append(None)  # NSD
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
#: the three-population label of each branch (galacticmodel's thin_disk /
#: thick_disk / bulge), for the per-population inputs that are not yet
#: ported per branch (the mass function and the age / [Fe/H] priors): the
#: stellar halo borrows the thick disk's, the oldest population there is
POPULATION_OF_BRANCH = (0, 0, 0, 0, 0, 0, 0, 1, 2, 1)

# ln-spaced trapezoid points for the position normalizer (galacticmodel's
# _POSITION_QUAD_POINTS rationale: ~0.7 pc steps at the bulge)
_POSITION_QUAD_POINTS = 200001


def log_total_number_density(d_pc, l_deg, b_deg):
    """numpy: log sum over BRANCHES of n_b at distance(s) d_pc."""
    d = np.maximum(np.asarray(d_pc, dtype=float), 1e-3)
    x, y, z = dlb_to_xyz(d, l_deg, b_deg)
    dens = log_number_densities(x, y, z)
    return np.logaddexp.reduce(np.stack([dens[i] for i in BRANCHES]), axis=0)


def log_position_norm(lower_pc, upper_pc, l_deg, b_deg):
    """numpy: log int_lower^upper sum_b n_b(d) d^2 dd along the sight line
    (trapezoid in ln d, integrand n d^3, peak factored out)."""
    lnd = np.linspace(
        np.log(float(lower_pc)), np.log(float(upper_pc)), _POSITION_QUAD_POINTS
    )
    f = log_total_number_density(np.exp(lnd), l_deg, b_deg) + 3.0 * lnd
    peak = float(np.max(f))
    return float(np.log(np.trapezoid(np.exp(f - peak), lnd)) + peak)

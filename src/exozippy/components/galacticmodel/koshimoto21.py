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
    rs_cyl = (xn**BAR_C1 + yn**BAR_C1) ** (1.0 / BAR_C1)
    rs = (rs_cyl**BAR_C2 + zn**BAR_C2) ** (1.0 / BAR_C2)
    log_main = -rs - 0.5 * (xp.maximum(R - BAR_RC, 0.0) / CUTOFF_SIGMA) ** 2
    # (the zb_c vertical cutoff is 1e6 pc at the default: never active)

    znx = xp.abs(zb / X_Z0)
    log_arms = []
    for sx in (-1.0, 1.0):
        xn_x = xp.abs((xb + sx * X_B_ZX * zb) / X_X0)
        yn_x = xp.abs((yb - X_B_ZY * zb) / X_Y0)
        rs_x = ((xn_x**X_C1 + yn_x**X_C1) ** (X_C2 / X_C1) + znx**X_C2) ** (
            1.0 / X_C2
        )
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
    R = xp.sqrt(x * x + y * y)
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

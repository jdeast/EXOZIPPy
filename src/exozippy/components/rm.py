"""
Rossiter-McLaughlin (RM) anomaly -- shared physics for the rvinstrument hook.

This module plays the same role for RM that ``components/gp.py`` plays for
Gaussian-process noise: it is NOT an instantiated component.  It owns the
differentiable Hirano+2011 RM kernel and the small helper that assembles the
in-transit RV distortion for one orbit, evaluated at an instrument's own RV
timestamps.  ``rvinstrument`` calls ``compute_rm_rv`` and adds the result to its
RV mean when a data file sets ``rm: <orbit_name>`` (off by default -> the RV
likelihood is byte-for-byte unchanged).

Provenance
----------
Base structure ported from allesfast ``rm_models/Hirano2011.py`` (X.-Y. Wang),
validated against IDL / EXOFASTv2.  Its single non-differentiable piece
(``scipy.special.j0``) is replaced by the symbolic two-branch Bessel J0 from
Jiayin Dong's published Hirano+2011 code (Code Ocean capsule 6180240), and the
scipy quadrature by constant-coefficient symbolic quadrature, so the whole
kernel is differentiable end to end and works with NUTS / numpyro / blackjax.

Geometry (planet sky position) reuses ``orbit.get_true_anomaly``; the limb-
darkened flux weight reuses exoplanet-core, matching the transit component.

Reference: Hirano, Suto, Winn et al. 2011, ApJ 742, 69.

Units: velocities in m/s at the interface (allesfast convention), km/s
internally.  Angles in radians.  Returns the RM anomaly in m/s.
"""

import numpy as np
import pytensor.tensor as pt
from exoplanet_core.pymc import ops


# ==========================================================================
# Differentiable Bessel J0  (Dong capsule; Numerical Recipes bessj0).
# ==========================================================================
def _bessj0_small(x):
    y = x * x
    ans1 = 57568490574.0 + y * (
        -13362590354.0
        + y
        * (
            651619640.7
            + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))
        )
    )
    ans2 = 57568490411.0 + y * (
        1029532985.0
        + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0)))
    )
    return ans1 / ans2


def _bessj0_large(ax):
    # ax is already |x| (>= 0). Guard so this branch has a FINITE, SMOOTH
    # gradient even where it is unused (ax < 8 selects the small branch, but
    # pt.switch still differentiates this one; unguarded 8/ax / sqrt(1/ax) ->
    # NaN grad as ax -> 0). A smooth floor (not pt.maximum) so no derivative
    # kink leaks into the NUTS Hamiltonian.
    ax = pt.sqrt(ax * ax + 1e-8)
    z = 8.0 / ax
    y = z * z
    xx = ax - 0.785398164
    ans1 = 1.0 + y * (
        -0.1098628627e-2
        + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6))
    )
    ans2 = -0.1562499995e-1 + y * (
        0.1430488765e-3
        + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7))
    )
    return pt.sqrt(0.636619772 / ax) * (
        pt.cos(xx) * ans1 - z * pt.sin(xx) * ans2
    )


def _bessj0_abs(ax):
    """J0 for a NON-NEGATIVE argument ax. Avoids pt.abs (whose gradient forms
    sign(x); when x carries the scalar sigma_max division, pytensor's
    local_sign_div rewriter mis-broadcasts scalar/vector and crashes the JAX
    funcify used by numpyro). J0 is even, so callers with a guaranteed-
    non-negative argument use this directly (the RM argument 2*pi*sigma*vsini*t
    is always >= 0)."""
    return pt.switch(pt.lt(ax, 8.0), _bessj0_small(ax), _bessj0_large(ax))


def bessj0(x):
    """Symbolic, differentiable Bessel J0(x) for general (possibly negative) x."""
    return _bessj0_abs(pt.abs(x))


# ==========================================================================
# Quadrature helpers (constant coefficients -> differentiable).
# ==========================================================================
def _gl_nodes_weights(n_gl=64):
    nodes, weights = np.polynomial.legendre.leggauss(n_gl)
    return (0.5 * (nodes + 1.0)).astype("float64"), (0.5 * weights).astype(
        "float64"
    )


def _trapz(y, x):
    """Composite-trapezoid integral of y over x along the last axis."""
    dx = x[1:] - x[:-1]
    return pt.sum(dx * 0.5 * (y[..., 1:] + y[..., :-1]), axis=-1)


def _m_kernel(sigma, vsini_kms, u1, u2, zeta_kms, t_gl, w_gl):
    """Disk-integrated broadening kernel M(sigma) (allesfast _compute_m_array),
    Gauss-Legendre in t; differentiable once bessj0 is symbolic.

    sigma: (n_sigma,) tensor.  Returns (n_sigma,) tensor.
    """
    t = pt.as_tensor_variable(t_gl)
    s = sigma[:, None]
    sqrt_1mt2 = pt.sqrt(pt.maximum(0.0, 1.0 - t * t))
    limb = (1.0 - u1 * (1.0 - sqrt_1mt2) - u2 * (1.0 - sqrt_1mt2) ** 2) / (
        1.0 - u1 / 3.0 - u2 / 6.0
    )
    pz2 = (np.pi * zeta_kms) ** 2 * s * s
    exp_macro = pt.exp(-pz2 * (1.0 - t * t)[None, :]) + pt.exp(
        -pz2 * (t * t)[None, :]
    )
    # argument is always >= 0 (sigma, vsini, t all >= 0) -> use the abs-free J0
    j0 = _bessj0_abs(2.0 * np.pi * s * vsini_kms * t[None, :])
    integrand = limb[None, :] * exp_macro * j0 * t[None, :]
    return pt.dot(integrand, pt.as_tensor_variable(w_gl))


# ==========================================================================
# Core RM kernel: takes geometry + flux, returns delta_v [m/s].
# ==========================================================================
def rm_delta_v_core(
    x,
    y,
    z,
    flux,
    vsini,
    u1,
    u2,
    vzeta=4000.0,
    vbeta=4000.0,
    vgamma=1000.0,
    n_sigma=201,
    n_gl=64,
    sigma_floor_kms=0.1,
):
    """Differentiable Hirano+2011 RM anomaly [m/s].

    x, y, z : planet sky position in stellar radii, with projected obliquity
              lambda already applied to (x, y).  v_sub = vsini * x; z >= 0 is
              in front of the star (transit), z < 0 (occultation) is zeroed.
    flux    : limb-darkened relative flux (1.0 out of transit).  f = 1 - flux.
    vsini, vzeta, vbeta, vgamma : m/s.  vbeta is the full Gaussian broadening
              (Hirano+2011 Eq. 20).
    n_sigma, n_gl : quadrature resolution (static; baked at graph-build time).

    The sigma grid COUNT is static (shape must be static in a symbolic graph),
    but its RANGE tracks the sampled vsini + zeta -- preserving allesfast's
    adaptive sigma_max: sigma = sigma_max * u with a constant grid u in (0, 1].
    """
    vsini_kms = vsini / 1e3
    zeta_kms = vzeta / 1e3
    beta_kms = vbeta / 1e3
    gamma_kms = vgamma / 1e3

    # Adaptive sigma grid (allesfast): the integration RANGE tracks the sampled
    # vsini+zeta (sigma_max = 5/(vsini+zeta+floor), ~5 broadening correlation
    # lengths) while the node COUNT stays static -- so the fixed n_sigma nodes
    # keep the same relative resolution at any vsini (a grid frozen to a constant
    # under-resolves once the sampled vsini exceeds the frozen reference).
    sigma_max = 5.0 / (vsini_kms + zeta_kms + sigma_floor_kms)  # scalar tensor
    u = np.linspace(1e-8, 1.0, n_sigma).astype("float64")
    sigma = sigma_max * pt.as_tensor_variable(u)  # (n_sigma,)

    t_gl, w_gl = _gl_nodes_weights(n_gl)
    m = _m_kernel(sigma, vsini_kms, u1, u2, zeta_kms, t_gl, w_gl)  # (n_sigma,)

    f = 1.0 - flux
    v_sub = vsini_kms * x
    r2 = x * x + y * y
    # theta below uses only cos_th**2 and sin_th**2, so work with the SQUARES
    # directly -- no sqrt, hence no infinite gradient at the disk edge. A hard
    # pt.maximum(0, .) would still leave a derivative KINK at the limb (r2=1),
    # which the planet crosses every transit and which makes the NUTS
    # Hamiltonian integrator diverge. Use the C-infinity smooth relu
    # 0.5*(u + sqrt(u**2 + eps)) instead, so the gradient is continuous
    # everywhere (and it saturates to 0 off-disk, avoiding exp overflow).
    _sm = lambda uu: 0.5 * (uu + pt.sqrt(uu * uu + 1e-6))
    cos_th2 = _sm(1.0 - r2)
    sin_th2 = _sm(r2)

    s = sigma[None, :]
    s2 = s * s
    mrow = m[None, :]
    em = (
        pt.exp(
            -2.0 * np.pi**2 * beta_kms**2 * s2 - 4.0 * np.pi * gamma_kms * s
        )
        * mrow
    )
    pz = (np.pi * zeta_kms) ** 2
    theta = 0.5 * (
        pt.exp(-pz * cos_th2[:, None] * s2)
        + pt.exp(-pz * sin_th2[:, None] * s2)
    )
    tp_vp_s = 2.0 * np.pi * v_sub[:, None] * s
    numer = em * theta * pt.sin(tp_vp_s) * s
    denom = em * (mrow - f[:, None] * theta * pt.cos(tp_vp_s)) * s2

    num_int = _trapz(numer, sigma)
    den_int = _trapz(denom, sigma)
    den_safe = pt.switch(pt.abs(den_int) > 1e-30, den_int, 1e-30)
    delta_v = (
        f / (2.0 * np.pi) * num_int / den_safe
    )  # km/s; outer f zeroes OOT
    delta_v = pt.switch(pt.ge(z, 0.0), delta_v, 0.0)  # occultation -> 0
    return -delta_v * 1e3  # m/s (allesfast sign)


def rm_delta_v_hirano2010(x, flux, vsini, vbeta):
    """Closed-form Hirano+2010 RM anomaly [m/s] -- the fast analytic series
    (rmfit's RMHirano.evaluate). No sigma/t quadrature and no Bessel J0: pure
    algebra on the subplanet velocity vp = vsini*x and the blocked flux
    F = 1 - flux, with Gaussian broadening sigma = vsini/1.31 (Hirano+2010) and
    line dispersion beta. Much faster than the Hirano+2011 disk integral
    (rm_delta_v_core) and accurate for moderate vp/(beta,sigma). flux == 1 out
    of transit / during occultation makes F = 0, so the anomaly is naturally 0
    there -- no explicit z mask needed. vsini, vbeta in m/s; returns m/s."""
    vsini_kms = vsini / 1e3
    beta_kms = vbeta / 1e3
    sigma_kms = vsini_kms / 1.31  # Hirano+2010 approximation
    vp = vsini_kms * x  # subplanet velocity [km/s]
    F = 1.0 - flux  # blocked flux fraction
    den = 2.0 * beta_kms**2 + sigma_kms**2
    pref = ((2.0 * beta_kms**2 + 2.0 * sigma_kms**2) / den) ** 1.5
    series = 1.0 - pt.sqr(vp) / den + pt.sqr(pt.sqr(vp)) / (2.0 * den**2)
    return -1000.0 * vp * F * pref * series  # m/s (rmfit sign)


# ==========================================================================
# Geometry: planet sky x/y/z with projected obliquity lambda (allesfast
# _planet_xy), from the true anomaly.  a_scale = a/Rstar.
# ==========================================================================
def rm_planet_xyz(true_anom, ecc, omega, ar, inc, lam):
    r = ar * (1.0 - ecc**2) / (1.0 + ecc * pt.cos(true_anom))
    wf = true_anom + omega
    x_old = -r * pt.cos(wf)
    y_old = -r * pt.sin(wf) * pt.cos(inc)
    cl, sl = pt.cos(lam), pt.sin(lam)
    x = x_old * cl - y_old * sl
    y = x_old * sl + y_old * cl
    z = r * pt.sin(wf) * pt.sin(inc)
    return x, y, z


def resolve_rm_indices(system, orbit_name, band_name=None):
    """(orbit_idx, planet_idx, band_idx) for an rvinstrument `rm:` request.

    planet_idx is the planet whose ``orbit_map`` points at ``orbit_name``;
    band_idx is the ``rm_band`` if given, else 0 (the first band).  Raises
    ValueError if ``orbit_name`` names no orbit or no planet transits it, so a
    misconfigured `rm:`/`rm_band:` fails loudly instead of silently modelling
    the wrong body.
    """
    orbit_names = list(system.orbit.names)
    if orbit_name not in orbit_names:
        raise ValueError(
            f"[rm] rm: {orbit_name!r} does not name any orbit (have "
            f"{orbit_names})."
        )
    orbit_idx = orbit_names.index(orbit_name)
    omap = list(getattr(system.planet, "orbit_map", []))
    planet_idx = next((i for i, o in enumerate(omap) if o == orbit_idx), None)
    if planet_idx is None:
        raise ValueError(
            f"[rm] rm: {orbit_name!r} has no planet on it "
            f"(planet.orbit_map={omap}); RM needs a transiting planet."
        )
    band_names = list(getattr(system.band, "names", []))
    band_idx = band_names.index(band_name) if band_name is not None else 0
    return orbit_idx, planet_idx, band_idx


def compute_rm_rv(
    system,
    time,
    orbit_idx,
    planet_idx,
    band_idx,
    n_sigma=201,
    model="hirano2011",
):
    """Assemble the RM RV distortion [m/s] for one orbit at ``time``.

    Geometry reuses ``orbit.get_true_anomaly`` (differentiable, exoplanet-core
    kepler) + the projected-obliquity projection ``rm_planet_xyz``.  a/Rstar and
    Rp/Rstar come from the ``planet`` component (``ar``, ``p``); the limb-
    darkening (u1, u2) from the ``rm_band`` Band -- same nodes the transit model
    uses, so the RM and transit LD are consistent.  Broadening terms
    (vzeta/vbeta/vgamma) are the shared stellar nodes; vsini + lambda are on the
    orbit (coupled via the sqrt(vsini)cos/sin(lambda) reparameterization).

    ``model`` selects the kernel: ``hirano2011`` (default; the disk integral,
    rm_delta_v_core) or ``hirano2010`` (the fast closed-form series, uses only
    vsini + vbeta).  Both share the geometry (x, flux) computed here.
    """
    orbit, star, planet, band = (
        system.orbit,
        system.star,
        system.planet,
        system.band,
    )

    ecc = orbit.ecc.value[orbit_idx]
    omega = orbit.omega.value[orbit_idx]
    inc = orbit.inc.value[orbit_idx]
    lam = orbit.lam.value[orbit_idx]
    ar = planet.ar.value[planet_idx]  # a / Rstar
    rprs = planet.p.value[planet_idx]  # Rp / Rstar
    u1 = band.u1.value[band_idx]
    u2 = band.u2.value[band_idx]

    f = orbit.get_true_anomaly(time)[:, orbit_idx]
    x, y, z = rm_planet_xyz(f, ecc, omega, ar, inc, lam)

    # limb-darkened blocked flux at the RV times (same LD basis as transit.py)
    rho = pt.sqrt(x * x + y * y)
    s_vec = ops.quad_solution_vector(rho, rprs + pt.zeros_like(rho))
    c = pt.stack([1.0 - u1 - u2, u1, u2])
    s_off = pt.as_tensor_variable(np.array([np.pi, 2.0 * np.pi / 3.0, 0.0]))
    flux = pt.dot(s_vec, c) / pt.dot(s_off, c)
    flux = pt.switch(pt.ge(z, 0.0), flux, 1.0)

    # Broadening (vmacro/vbeta/vmicro) belongs to the transited star -- the
    # primary of this orbit -- not necessarily star 0 (multi-star systems).
    star_idx = next(
        (
            idx
            for (ctype, idx) in orbit.primary_bodies[orbit_idx]
            if ctype == "star"
        ),
        0,
    )
    vsini = orbit.vsini.value[orbit_idx]
    vzeta = star.vmacro.value[star_idx]
    vbeta = star.vbeta.value[star_idx]
    vgamma = star.vmicro.value[star_idx]

    if model == "hirano2010":
        return rm_delta_v_hirano2010(x, flux, vsini, vbeta)
    return rm_delta_v_core(
        x,
        y,
        z,
        flux,
        vsini,
        u1,
        u2,
        vzeta=vzeta,
        vbeta=vbeta,
        vgamma=vgamma,
        n_sigma=n_sigma,
    )


# ==========================================================================
# Topology helper: which orbit names have RM enabled by some rvinstrument.
# Used by orbit/star to declare the RM parameters only when needed.
# ==========================================================================
def rm_orbits_in_system(system):
    """Return the set of orbit names referenced by any rvinstrument `rm:` key."""
    names = set()
    cfg = getattr(system, "config", None) or {}
    for entry in cfg.get("rvinstrument", []) or []:
        o = entry.get("rm")
        if o:
            names.add(o)
    return names


def rm_enabled(system):
    return len(rm_orbits_in_system(system)) > 0

"""The barycentric correction INSIDE the model: raw measured redshifts.

An ``rvinstrument`` file with ``data_format: raw_redshift`` carries what a
spectrograph measures -- ``c z_meas``, the redshift of the stellar spectrum
against its template BEFORE any barycentric correction -- and this module is
the forward model that predicts it from the fitted star and companions
(``rvinstrument.md``, "Raw redshifts").  It follows Wright & Eastman (2014,
PASP 126, 838; "WE14"), the formalism behind barycorrpy (Kanodia & Wright
2018), but runs it the other way round: instead of correcting the data with
a FIXED catalog star, the prediction is a function of the SAMPLED star
(``star.ra``/``dec``/``pm_ra``/``pm_dec``/``parallax``/``rv``) and of every
orbit's reflex, so the multiplicative cross terms and the consistent proper
motion come out of the one model.

The split is the WE14 one, and it decides what may be precomputed:

* the OBSERVER (``observer_terms``, at stage 1, numpy + astropy): its
  barycentric position and velocity at the reception instant -- the Earth's
  orbit, its rotation (GCRS velocity of the site) or a spacecraft's
  ``.eph`` track -- and the solar-system gravitational potential at it.
  None of it depends on a parameter.
* the TARGET (``star_geometry`` / ``doppler_ratio``, in the graph, written
  against an array module ``xp`` so the same code runs in numpy for the
  tests and the seeds and in PyTensor for the likelihood): the direction
  ``rho_hat(t)`` from the observer to the star, the Roemer delay to the
  barycentric arrival time ``t_B`` at which every orbit is evaluated, the
  star's space velocity, and the Doppler factors.

The prediction (WE14 eq. 28 turned into a forward model) is

    1 + z_meas = (1 + z0_i) * gamma_* (1 + rho_hat . beta_*)
                 / [gamma_o (1 + rho_hat . beta_o)]
                 * (1 + z_grav,o) * (1 + z_Shapiro)

with ``beta_*`` the star's TOTAL velocity (systemic space motion plus every
orbit's reflex) and ``beta_o`` the observer's, both barycentric over c;
``z_grav,o = 1/(1 + Phi_o/c^2) - 1`` the blueshift of light falling into the
solar-system potential at the observer; ``z_Shapiro`` WE14's eq. 27; and
``z0_i`` the instrument's zero point (template, the star's own gravitational
redshift and convective blueshift: constants, absorbed there).

THE STAR'S MOTION is the constant space velocity built from the astrometric
parameters at their reference epoch (the 3-D model whose tangent-plane
linearization is ``AstrometryInstrument._star_track``):

    d0     = 1 / parallax                                   (AU)
    mu     = pm_ra e_hat + pm_dec n_hat                     (rad / yr)
    X(T)   = d0 r0_hat + (d0 mu + rv r0_hat) T              (AU, T in yr)

``pm`` and the position are APPARENT, per year of barycentric arrival time,
as astrometry measures them; the star's TRUE transverse velocity is then
``d0 mu (1 + rv/c)``, because a receding star's emission times are
compressed by ``1 + rv/c`` relative to their arrival times.  That factor is
WE14's / barycorrpy's ``z_lighttravel`` term (4 cm/s for Barnard's star 25
yr from its epoch), here not a separate term but the consistent velocity.
The perspective (secular) acceleration is not a term either: it is the
change of ``rho_hat . beta_*`` as ``rho_hat`` turns along X(T).

Units throughout: positions AU, velocities m/s, times days (JD_TDB), angles
radians, pm mas/yr, parallax mas.  Validated against barycorrpy in
``tests/test_raw_redshift.py`` (numbers in ``rvinstrument.md``).
"""

import astropy.constants as const
import astropy.units as u
import numpy as np

from exozippy.constants import C_MPS

#: AU in metres, a parsec in AU, a mas in radians, a Julian year in s.
AU_M = const.au.to(u.m).value
PC_AU = (1.0 * u.pc).to(u.au).value
MAS_RAD = (1.0 * u.mas).to(u.rad).value
JULIAN_YEAR_D = 365.25
JULIAN_YEAR_S = JULIAN_YEAR_D * 86400.0
#: m/s -> AU per Julian year.
MPS_TO_AU_PER_YR = JULIAN_YEAR_S / AU_M
C_AU_PER_D = C_MPS * 86400.0 / AU_M

#: The bodies whose potential (and Shapiro term) at the observer is kept,
#: and their GM in m^3/s^2 -- barycorrpy's set (WE14 sec. 5: everything else
#: is < 1 mm/s).  The Sun's term is ~3 m/s with a +/-5 cm/s annual swing
#: (the Earth's orbital eccentricity); the Earth's own is ~0.2 m/s and
#: constant at a site; Jupiter and Saturn are sub-mm/s.
GRAVITY_BODIES = (
    ("sun", const.GM_sun.to(u.m**3 / u.s**2).value),
    ("earth", const.GM_earth.to(u.m**3 / u.s**2).value),
    ("jupiter", const.GM_jup.to(u.m**3 / u.s**2).value),
    # DE430's GM of the Saturn system, 37931207.8 km^3/s^2.
    ("saturn", 3.79312078e16),
)


def observer_terms(t_tdb, observer_location, ephemeris):
    """Everything about the observer at each reception instant (stage 1).

    ``t_tdb`` -- (N,) the observer's coordinate time, JD_TDB, at which it
    received the light (``TimeSystem._to_observer_tdb``).

    Returns a dict of numpy arrays:

    ``r_au``  (N, 3) barycentric ICRS position, AU;
    ``beta``  (N, 3) barycentric velocity / c (``ephemeris.get_observer_posvel``:
              the Earth's orbit plus the site's rotation, relativistically
              added, or a spacecraft's spline velocity);
    ``gamma`` (N,)   its Lorentz factor;
    ``one_plus_zgrav`` (N,) ``1 / (1 + sum GM/(r c^2))`` over ``GRAVITY_BODIES``;
    ``shapiro_hat`` (B, N, 3) unit vectors from each body to the observer and
    ``shapiro_coef`` (B, N) ``-2 GM / (c^2 r)``, the parameter-independent
    halves of WE14 eq. 27 (the other half reads the fitted direction).
    """
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    from exozippy.ephemeris import get_observer_posvel, use_ephemeris

    t_tdb = np.atleast_1d(np.asarray(t_tdb, dtype=float))
    r_au, v = get_observer_posvel(t_tdb, observer_location, ephemeris)
    beta = v / C_MPS
    gamma = 1.0 / np.sqrt(1.0 - np.sum(beta**2, axis=1))
    r_m = r_au * AU_M
    phi = np.zeros(t_tdb.size)
    hats, coefs = [], []
    with use_ephemeris(ephemeris):
        t_obj = Time(t_tdb, format="jd", scale="tdb")
        for body, gm in GRAVITY_BODIES:
            rb = np.atleast_2d(
                get_body_barycentric(body, t_obj).xyz.to(u.m).value.T
            )
            sep = r_m - rb
            dist = np.linalg.norm(sep, axis=1)
            if not np.all(dist > 0.0):
                # Only an observer AT a body's centre gets here; raw mode
                # refuses 'earth'/'moon' at construction, so this is an
                # internal invariant, not a user input.
                raise ValueError(
                    f"observer {observer_location!r} sits at the centre of "
                    f"{body}; its potential is singular there."
                )
            phi += gm / dist
            hats.append(sep / dist[:, None])
            coefs.append(-2.0 * gm / (C_MPS**2 * dist))
    return {
        "r_au": r_au,
        "beta": beta,
        "gamma": gamma,
        "one_plus_zgrav": 1.0 / (1.0 + phi / C_MPS**2),
        "shapiro_hat": np.asarray(hats),
        "shapiro_coef": np.asarray(coefs),
    }


def star_frame(ra, dec, xp=np):
    """``(r0_hat, e_hat, n_hat)`` at (ra, dec) [rad], each a 3-tuple of
    scalars: the direction to the star and the spherical tangent basis
    (``skyframe.sky_basis``'s, the same e/n WE14 eqs. 14-17 build from the
    pole).  The orbit's sky frame maps onto them as X = North = n_hat,
    Y = East = e_hat, Z = away = r0_hat."""
    sa, ca = xp.sin(ra), xp.cos(ra)
    sd, cd = xp.sin(dec), xp.cos(dec)
    r0 = (cd * ca, cd * sa, sd)
    e = (-sa, ca, 0.0 * sa)
    n = (-sd * ca, -sd * sa, cd)
    return r0, e, n


def star_geometry(
    t_tdb, r_obs_au, ra, dec, pm_ra, pm_dec, parallax, rv, epoch, xp=np
):
    """The target half that does not involve the orbits.

    ``t_tdb`` (N,) observer coordinate time [JD_TDB]; ``r_obs_au`` (N, 3)
    its barycentric position; the star's astrometric parameters at
    ``epoch`` (BJD_TDB) in internal units (rad, mas/yr, mas, m/s).

    Returns a dict:

    ``frame``    (r0_hat, e_hat, n_hat), see ``star_frame``;
    ``rho_hat``  3-tuple of (N,): unit vector observer -> star;
    ``t_bary``   (N,) the barycentric arrival time BJD_TDB of each spectrum's
                 light, ``t + (|X| - |X - r_obs|)/c`` -- the Roemer delay
                 with its curvature (parallax) term, written in the
                 cancellation-free form ``(2 X.r - r.r)/(|X| + |X - r|)``;
                 the Shapiro DELAY (< 0.1 ms) is not modeled;
    ``beta_sys`` 3-tuple of scalars: the star's true barycentric space
                 velocity over c (module docstring for the ``1 + rv/c``).

    The star's position is taken at the reception instant ``t_tdb`` rather
    than at ``t_bary``: they differ by <= 8.3 min, which turns the
    direction by mu * 8 min (1e-12 rad for Barnard's star).
    """
    r0, e, n = star_frame(ra, dec, xp)
    d0 = PC_AU * 1000.0 / parallax  # AU
    mu = [(pm_ra * e[k] + pm_dec * n[k]) * MAS_RAD for k in range(3)]  # rad/yr
    T = (t_tdb - epoch) / JULIAN_YEAR_D  # yr
    rv_auyr = rv * MPS_TO_AU_PER_YR
    X = [d0 * r0[k] + (d0 * mu[k] + rv_auyr * r0[k]) * T for k in range(3)]
    rho = [X[k] - r_obs_au[:, k] for k in range(3)]
    rho_norm = xp.sqrt(rho[0] ** 2 + rho[1] ** 2 + rho[2] ** 2)
    rho_hat = tuple(rho[k] / rho_norm for k in range(3))
    X_norm = xp.sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)
    x_dot_r = sum(X[k] * r_obs_au[:, k] for k in range(3))
    r_dot_r = np.sum(r_obs_au**2, axis=1)
    t_bary = (
        t_tdb + (2.0 * x_dot_r - r_dot_r) / (X_norm + rho_norm) / C_AU_PER_D
    )
    # True velocity: the apparent transverse motion stretched by the
    # emission-time compression (module docstring), m/s.
    stretch = 1.0 + rv / C_MPS
    v_tan = d0 * AU_M / JULIAN_YEAR_S * stretch
    beta_sys = tuple((v_tan * mu[k] + rv * r0[k]) / C_MPS for k in range(3))
    return {
        "frame": (r0, e, n),
        "rho_hat": rho_hat,
        "t_bary": t_bary,
        "beta_sys": beta_sys,
    }


def doppler_ratio(geom, obs, v_reflex, xp=np):
    """``1 + z_meas`` without the instrument zero point (WE14 eq. 28 as a
    forward model; module docstring).

    ``geom`` -- ``star_geometry``'s dict; ``obs`` -- ``observer_terms``'
    dict (numpy constants); ``v_reflex`` -- ``(v_los, v_n, v_e)`` the star's
    reflex velocity in the orbit's sky frame (m/s; Z/away, X/North,
    Y/East), each (N,) or None for a component no model provides (no orbit,
    or a sky orientation the system does not model).
    """
    r0, e, n = geom["frame"]
    rho = geom["rho_hat"]
    v_los, v_n, v_e = v_reflex
    beta = []
    for k in range(3):
        b = geom["beta_sys"][k]
        if v_los is not None:
            b = b + v_los * r0[k] / C_MPS
        if v_n is not None:
            b = b + v_n * n[k] / C_MPS
        if v_e is not None:
            b = b + v_e * e[k] / C_MPS
        beta.append(b)
    beta2 = beta[0] ** 2 + beta[1] ** 2 + beta[2] ** 2
    gamma_star = 1.0 / xp.sqrt(1.0 - beta2)
    star = gamma_star * (1.0 + sum(rho[k] * beta[k] for k in range(3)))
    bo = obs["beta"]
    observer = obs["gamma"] * (1.0 + sum(rho[k] * bo[:, k] for k in range(3)))
    # WE14 eq. 27: sum_b -2GM/(c^2 r) [(rho - (R.rho)R) . beta_o] / (1 + R.rho)
    z_shapiro = 0.0
    for b in range(obs["shapiro_coef"].shape[0]):
        R = obs["shapiro_hat"][b]
        r_dot = sum(R[:, k] * rho[k] for k in range(3))
        perp_dot_beta = sum(
            (rho[k] - r_dot * R[:, k]) * bo[:, k] for k in range(3)
        )
        z_shapiro = z_shapiro + obs["shapiro_coef"][b] * perp_dot_beta / (
            1.0 + r_dot
        )
    return star / observer * obs["one_plus_zgrav"] * (1.0 + z_shapiro)


def predicted_cz(t_tdb, obs, star, epoch, z0=0.0, v_reflex=None, xp=np):
    """Convenience for tests and seeds: ``c z_meas`` (m/s) predicted for a
    star with parameters ``star = (ra, dec, pm_ra, pm_dec, parallax, rv)``
    (internal units) and an optional reflex ``v_reflex`` evaluated by the
    caller (it needs ``t_bary``; pass a callable ``f(t_bary) -> (v_los,
    v_n, v_e)``).  Returns ``(cz, t_bary)``."""
    geom = star_geometry(t_tdb, obs["r_au"], *star, epoch, xp=xp)
    if v_reflex is None:
        refl = (None, None, None)
    else:
        refl = v_reflex(geom["t_bary"])
    ratio = (1.0 + z0) * doppler_ratio(geom, obs, refl, xp=xp)
    return C_MPS * (ratio - 1.0), geom["t_bary"]


def barycentric_redshift(t_tdb, obs, star, epoch, xp=np):
    """WE14's ``z_B`` for a star with no companions, in the convention
    barycorrpy returns it: ``1 + z_true = (1 + z_B)(1 + z_meas)``, with
    ``1 + z_true = gamma_* (1 + r0_hat . beta_*)`` the star's own redshift
    along its reference direction.  The validation target, and the seed's
    correction (``RVInstrument`` raw mode's ``gamma`` start)."""
    geom = star_geometry(t_tdb, obs["r_au"], *star, epoch, xp=xp)
    ratio = doppler_ratio(geom, obs, (None, None, None), xp=xp)
    r0 = geom["frame"][0]
    b = geom["beta_sys"]
    beta2 = b[0] ** 2 + b[1] ** 2 + b[2] ** 2
    true = (1.0 + sum(r0[k] * b[k] for k in range(3))) / xp.sqrt(1.0 - beta2)
    return true / ratio - 1.0

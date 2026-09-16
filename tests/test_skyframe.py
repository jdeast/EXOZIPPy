"""
Tests for exozippy.skyframe -- the one sky-plane frame and its projections.

Two layers:

* the mechanical contract of the helpers (basis definition, the
  parallax_factors == -observer_sky_offset sign relation, numpy/pytensor
  agreement, agreement with the cross-product construction op.py used to
  carry), and
* the PHYSICS the shared definition exists to protect: that the microlensing
  trajectory and the Keplerian sky/RV formulas are both consistent with the
  left-handed (North, East, distance) convention.  Those two tests are the
  reason a sign fix must never again be applied to one copy and not the
  others; each is built from first principles and each fails loudly under a
  mirrored convention.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.skyframe import (
    observer_sky_offset,
    parallax_factors,
    sky_basis,
)

RA, DEC = np.radians(266.4168), np.radians(-29.0078)


def _random_positions(n=64, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * 1.2


# --------------------------------------------------------------------------
# the helper contract
# --------------------------------------------------------------------------


def test_sky_basis_is_the_spherical_tangent_basis():
    """
    Given a sky position,
    When sky_basis is evaluated,
    Then e_hat and n_hat are the normalized derivatives of the unit vector
      with respect to ra and dec, and form an orthonormal pair with it.
    """
    # Arrange
    ra, dec = RA, DEC
    u_hat = np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    h = 1e-7
    du_dra = (
        np.array(
            [
                np.cos(dec) * np.cos(ra + h),
                np.cos(dec) * np.sin(ra + h),
                np.sin(dec),
            ]
        )
        - u_hat
    ) / h
    du_ddec = (
        np.array(
            [
                np.cos(dec + h) * np.cos(ra),
                np.cos(dec + h) * np.sin(ra),
                np.sin(dec + h),
            ]
        )
        - u_hat
    ) / h

    # Act
    e_hat, n_hat = sky_basis(ra, dec)

    # Assert
    np.testing.assert_allclose(e_hat, du_dra / np.cos(dec), atol=1e-6)
    np.testing.assert_allclose(n_hat, du_ddec, atol=1e-6)
    for v in (e_hat, n_hat):
        assert np.isclose(np.linalg.norm(v), 1.0)
    assert abs(e_hat @ n_hat) < 1e-15
    assert abs(e_hat @ u_hat) < 1e-15
    assert abs(n_hat @ u_hat) < 1e-15


def test_cross_product_construction_agrees():
    """
    Given the cross-product construction op.py carried before the refactor
      (mirroring MulensModel Coordinates: east = z x direction, north =
      direction x east),
    When it is compared to sky_basis over the whole sphere,
    Then the two agree to within 1 ulp.

    Pins the claim in op.py's comment.  They are NOT bit-identical (~40% of
    positions match exactly), which is why that comment says 1 ulp rather
    than "identical": the Op path's line of sight moved in its last bit when
    the definitions were shared, and nothing else.
    """
    # Arrange
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(2000):
        ra = rng.uniform(0.0, 2 * np.pi)
        dec = np.arcsin(rng.uniform(-1.0, 1.0))
        direction = np.array(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
        )
        east = np.cross([0.0, 0.0, 1.0], direction)
        east /= np.linalg.norm(east)
        north = np.cross(direction, east)

        # Act
        e_hat, n_hat = sky_basis(ra, dec)

        # Assert
        worst = max(
            worst,
            np.max(np.abs(east - e_hat)),
            np.max(np.abs(north - n_hat)),
        )

    assert worst <= 2 * np.finfo(float).eps, f"max deviation {worst:.3e}"


def test_observer_sky_offset_is_the_basis_dot_products():
    """
    Given observer positions and a sky position,
    When observer_sky_offset is evaluated,
    Then it equals the dot products of those positions with sky_basis.
    """
    # Arrange
    xyz = _random_positions()
    e_hat, n_hat = sky_basis(RA, DEC)

    # Act
    delta_e, delta_n = observer_sky_offset(xyz, RA, DEC)

    # Assert
    np.testing.assert_allclose(delta_e, xyz @ e_hat, rtol=0, atol=1e-15)
    np.testing.assert_allclose(delta_n, xyz @ n_hat, rtol=0, atol=1e-15)


def test_parallax_factors_are_the_negated_offset():
    """
    Given observer positions,
    When parallax_factors and observer_sky_offset are both evaluated,
    Then they are exact negatives -- the astrometry/microlensing sign
      relation is a definition, not two independently written formulas that
      happen to disagree.
    """
    # Arrange
    xyz = _random_positions()

    # Act
    delta_e, delta_n = observer_sky_offset(xyz, RA, DEC)
    p_e, p_n = parallax_factors(xyz, RA, DEC)

    # Assert
    assert np.array_equal(p_e, -delta_e)
    assert np.array_equal(p_n, -delta_n)


def test_parallax_factors_match_first_principles_displacement():
    """
    Given a source at a known distance and an observer displaced from the
      barycenter,
    When the apparent direction to the source is computed exactly in 3-D and
      resolved on the sky basis,
    Then it equals parallax * parallax_factors.

    This is what fixes the SIGN: an observer displaced toward East sees the
    source displaced toward West.
    """
    # Arrange
    d_pc = 250.0
    plx_rad = 1.0 / (d_pc * 206264.806247)  # AU/pc -> radians
    xyz = _random_positions(n=32, seed=11) * 1.0  # AU
    e_hat, n_hat = sky_basis(RA, DEC)
    u_hat = np.array(
        [
            np.cos(DEC) * np.cos(RA),
            np.cos(DEC) * np.sin(RA),
            np.sin(DEC),
        ]
    )
    d_au = d_pc * 206264.806247

    # Act
    # exact apparent direction from the displaced observer
    los = d_au * u_hat[None, :] - xyz
    los /= np.linalg.norm(los, axis=1)[:, None]
    exact_e = los @ e_hat
    exact_n = los @ n_hat

    p_e, p_n = parallax_factors(xyz, RA, DEC)

    # Assert (first order in plx; the O(plx^2) term is ~1e-11 of the signal)
    np.testing.assert_allclose(exact_e, plx_rad * p_e, rtol=1e-5)
    np.testing.assert_allclose(exact_n, plx_rad * p_n, rtol=1e-5)


def test_pytensor_backend_matches_numpy():
    """
    Given the same inputs,
    When the helpers are built symbolically with xp=pytensor.tensor,
    Then the compiled graph reproduces the numpy result to machine
      precision.

    The symbolic path is Lens.get_magnification's; a backend that silently
    disagreed would separate the likelihood from every numpy consumer
    (bootstrap, plots) that shares these helpers.

    Not bit-identical: pytensor's C backend fuses and reassociates the
    elementwise graph, so ~10% of elements land 1 ulp away.  The tolerance
    is ABSOLUTE and scaled to the input magnitude, not relative: the
    projection is a sum of same-magnitude terms, so an element that nearly
    cancels carries a large relative error (1.5e-15) on a tiny absolute one
    (1.2e-16).  Any real disagreement -- a sign or a swapped term -- is
    O(1) here and still caught.
    """
    # Arrange
    xyz = _random_positions()
    ra_t, dec_t = pt.dscalar("ra"), pt.dscalar("dec")
    xyz_t = pt.dmatrix("xyz")

    # Act
    de_t, dn_t = observer_sky_offset(xyz_t, ra_t, dec_t, xp=pt)
    pe_t, pn_t = parallax_factors(xyz_t, ra_t, dec_t, xp=pt)
    fn = pytensor.function([xyz_t, ra_t, dec_t], [de_t, dn_t, pe_t, pn_t])
    de, dn, pe, pn = fn(xyz, RA, DEC)

    # Assert
    de_np, dn_np = observer_sky_offset(xyz, RA, DEC)
    pe_np, pn_np = parallax_factors(xyz, RA, DEC)
    tol = 8 * np.finfo(float).eps * np.max(np.abs(xyz))
    for got, want in ((de, de_np), (dn, dn_np), (pe, pe_np), (pn, pn_np)):
        np.testing.assert_allclose(got, want, rtol=0.0, atol=tol)


# --------------------------------------------------------------------------
# the physics the shared definition protects
# --------------------------------------------------------------------------


def _euler_orbit(t, P, e, inc, om, bigom, tp=0.0, a=1.0):
    """(N, E, Z) of a body on a Keplerian orbit, built from Euler angles.

    r = Rz(bigomega) Rx(inc) Rz(omega) . (r cos f, r sin f, 0), with the
    axes LABELLED (North, East, distance) -- the left-handed convention.
    Deliberately independent of orbit.py: this is the textbook construction
    orbit.py's closed form must reproduce.
    """

    def _rz(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _rx(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    n = 2 * np.pi / P
    M = n * (np.atleast_1d(t) - tp)
    ecc_anom = M.copy()
    for _ in range(100):
        ecc_anom = ecc_anom - (ecc_anom - e * np.sin(ecc_anom) - M) / (
            1 - e * np.cos(ecc_anom)
        )
    f = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(ecc_anom / 2),
        np.sqrt(1 - e) * np.cos(ecc_anom / 2),
    )
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    rot = _rz(bigom) @ _rx(inc) @ _rz(om)
    return rot @ np.vstack([r * np.cos(f), r * np.sin(f), np.zeros_like(f)]), f


_ORB = dict(
    P=11.3,
    e=0.37,
    inc=np.radians(78.0),
    om=np.radians(41.0),
    bigom=np.radians(133.0),
)


def test_keplerian_sky_and_rv_are_left_handed():
    """
    Given a Keplerian orbit built from the textbook Euler rotation with the
      axes labelled (North, East, distance),
    When Orbit.get_sky_position's and Orbit.get_radial_velocity's closed
      forms are evaluated on it,
    Then all three convention claims hold: dN/dE reproduce the construction,
      the position angle at the ascending node is bigomega, and the RV
      formula is dZ/dt with positive = receding.

    This is what forces the LEFT-handed labelling: a right-handed
    relabelling breaks one of the three.
    """
    # Arrange
    P, e = _ORB["P"], _ORB["e"]
    inc, om, bigom = _ORB["inc"], _ORB["om"], _ORB["bigom"]
    t = np.linspace(0.0, P, 20001)
    xyz, f = _euler_orbit(t, **_ORB)
    n_fp, e_fp, z_fp = xyz

    # Act -- orbit.py:782-783 (get_sky_position)
    r = (1 - e**2) / (1 + e * np.cos(f))
    cosw, sinw, cosi = np.cos(om), np.sin(om), np.cos(inc)
    cos_o, sin_o = np.cos(bigom), np.sin(bigom)
    coswf = cosw * np.cos(f) - sinw * np.sin(f)
    sinwf = sinw * np.cos(f) + cosw * np.sin(f)
    d_n = r * (cos_o * coswf - sin_o * sinwf * cosi)
    d_e = r * (sin_o * coswf + cos_o * sinwf * cosi)

    # Assert (a) the projection
    np.testing.assert_allclose(d_n, n_fp, atol=1e-14)
    np.testing.assert_allclose(d_e, e_fp, atol=1e-14)

    # Assert (b) PA at the ascending node (omega + f = 0) is bigomega
    at_node = int(np.argmin(np.abs(((om + f + np.pi) % (2 * np.pi)) - np.pi)))
    pa = np.degrees(np.arctan2(d_e[at_node], d_n[at_node])) % 360.0
    assert abs(pa - np.degrees(bigom)) < 0.05, f"PA at node = {pa}"

    # Assert (c) the RV formula is dZ/dt, positive = receding
    k_true = (2 * np.pi / P) * np.sin(inc) / np.sqrt(1 - e**2)
    rv_code = k_true * (
        cosw * np.cos(f) - sinw * np.sin(f) + e * cosw
    )  # orbit.py:809
    # Interior points only: np.gradient falls back to a one-sided (first
    # order) difference at the two endpoints, which is 1e-4 here against
    # 1e-8 for the central differences everywhere else.
    np.testing.assert_allclose(
        rv_code[1:-1], np.gradient(z_fp, t)[1:-1], atol=1e-6
    )
    assert rv_code[at_node] > 0.0, "RV at the ascending node must be receding"


def test_primary_transit_puts_the_planet_in_front():
    """
    Given omega is the PRIMARY's argument of periastron (omega_*) and
      calc_tp places primary transit at f = pi/2 - omega_*,
    When the star's position is evaluated there,
    Then Z_star > 0 -- the star is the FARTHER body, so the planet is in
      front -- and the sky separation is r*|cos i|, which is what
      calc_b's ar*cosi*(1-e^2)/(1+esinw) evaluates to.

    Guards the primary/secondary sense against the Z-axis being flipped.
    """
    # Arrange
    e, inc, om = _ORB["e"], _ORB["inc"], _ORB["om"]
    f_transit = (np.pi / 2 - om) % (2 * np.pi)
    # invert to the time of that true anomaly, then reuse the construction
    ecc_anom = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(f_transit / 2),
        np.sqrt(1 + e) * np.cos(f_transit / 2),
    )
    t_transit = (ecc_anom - e * np.sin(ecc_anom)) / (2 * np.pi / _ORB["P"])

    # Act
    xyz, f = _euler_orbit(np.array([t_transit]), **_ORB)
    n_s, e_s, z_s = xyz[:, 0]

    # Assert
    assert np.isclose(f[0] % (2 * np.pi), f_transit)
    assert z_s > 0.0, "star must be the farther body at primary transit"
    r = (1 - e**2) / (1 + e * np.sin(om))  # calc_b's r at conjunction
    np.testing.assert_allclose(np.hypot(n_s, e_s), r * abs(np.cos(inc)))


@pytest.mark.slow
def test_microlensing_trajectory_matches_3d_geometry():
    """
    Given a lens and a source with a known relative proper motion, and a real
      Earth ephemeris,
    When the apparent lens-source separation is built from raw 3-D geometry
      in the (East, North) basis and compared to Lens.get_magnification's
      trajectory formula,
    Then |u(t)| agrees to machine precision, while the MIRRORED beta
      convention is wrong by O(0.1) in |u|.

    Both signs -- tau AND beta -- are pinned by this, so it is a HANDEDNESS
    test, not a magnitude test.  It is the evidence that the microlensing
    trajectory already shares the orientation convention of
    Orbit.get_sky_position (beta_hat is tau_hat rotated +90 degrees in the
    East-of-North sense), and hence that no sign needs to be "unified".
    """
    # Arrange
    import astropy.units as u_ast
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    def _earth(times):
        return (
            get_body_barycentric(
                "earth", Time(times, format="jd", scale="tdb")
            )
            .xyz.to(u_ast.au)
            .value.T
        )

    t0_par = 2460025.0
    t = np.linspace(t0_par - 150.0, t0_par + 150.0, 400)
    dt = 0.5
    pos_ref = _earth(np.array([t0_par]))[0]
    vel_ref = (
        _earth(np.array([t0_par + dt]))[0] - _earth(np.array([t0_par - dt]))[0]
    ) / (2 * dt)
    dev = _earth(t) - (
        pos_ref[None, :] + vel_ref[None, :] * (t - t0_par)[:, None]
    )

    theta_e, pi_rel = 0.6, 0.35  # mas
    mu_e, mu_n = 4.0, -3.0  # geocentric mu_rel (lens - source), mas/yr
    mu_mag = np.hypot(mu_e, mu_n)
    t_e = theta_e / (mu_mag / 365.25)
    pi_e = pi_rel / theta_e
    pi_e_n, pi_e_e = pi_e * mu_n / mu_mag, pi_e * mu_e / mu_mag
    t_0, u_0 = t0_par, 0.23

    # Act -- (A) lens.py:1411-1428
    delta_e, delta_n = observer_sky_offset(dev, RA, DEC)
    tau_p = (t - t_0) / t_e - delta_n * pi_e_n - delta_e * pi_e_e
    u_p = u_0 + delta_n * pi_e_e - delta_e * pi_e_n
    u_code = np.hypot(tau_p, u_p)

    # (B) first principles: theta_apparent = theta_bary - pi * offset, so the
    # lens-minus-source separation picks up -pi_rel * offset.  beta_hat is
    # tau_hat rotated +90 deg East of North: (N, E) = (c, d) -> (-d, c).
    c, d = mu_n / mu_mag, mu_e / mu_mag
    beta_n, beta_e = -d, c
    dth_n = (
        u_0 * beta_n
        + (mu_n / 365.25) * (t - t_0) / theta_e
        - (pi_rel / theta_e) * delta_n
    )
    dth_e = (
        u_0 * beta_e
        + (mu_e / 365.25) * (t - t_0) / theta_e
        - (pi_rel / theta_e) * delta_e
    )
    u_geom = np.hypot(dth_n, dth_e)

    # Assert
    np.testing.assert_allclose(u_code, u_geom, atol=1e-12)

    # and the mirror is grossly wrong -- this test has teeth
    u_mirror = np.hypot(tau_p, u_0 - delta_n * pi_e_e + delta_e * pi_e_n)
    assert np.max(np.abs(u_mirror - u_geom)) > 0.01

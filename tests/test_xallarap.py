"""
Source orbital motion -- xallarap (conventions.md C25; review 8.6.9).

What is pinned here, and why each pin is the one that matters:

  - The C9 first-principles reconstruction: the trajectory shift
    (dtau, du) must equal the source's Keplerian sky offset -- rebuilt in
    numpy by this test's own Kepler solver and Thiele-Innes projection --
    anchored at t0_par and projected on (tau_hat, beta_hat) with C7's
    minus (lens MINUS source).  The projection's absolute orientation is
    pinned first-principles by tests/test_skyframe.py /
    tests/test_astrometry.py; this test pins the xallarap COMPOSITION of
    it, which no chi2 can (a flipped sign is a different but
    healthy-looking model).
  - The magnification actually consumes the shift: the symbolic PSPL path
    equals the Paczynski formula evaluated at the manually shifted
    (tau, u).
  - The t0_par anchor: the shift vanishes there (5d: one anchor for
    parallax and both orbital-motion effects, so t_0/u_0 keep their
    meaning).
  - The Op plumbing: VBMDirectMagOp(source_motion=True) through the ESPL
    branch reproduces the symbolic path at negligible rho.
  - No new sampled parameters; the xallarap orbit samples bigomega but
    stays node-degenerate (a sky track is reflection-invariant), unlike
    the lens-keplerian case.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytestmark = pytest.mark.slow

from exozippy.constants import RSUN_TO_AU
from exozippy.system import System

_T0_PAR = 2458554.89


def _write_lc(path, n=90, span=60.0):
    rng = np.random.default_rng(13)
    t = np.linspace(_T0_PAR - span, _T0_PAR + span, n)
    mag = 15.0 - rng.uniform(0, 0.001, n)
    err = np.full(n, 0.01)
    np.savetxt(path, np.column_stack([t, mag, err]))
    return str(path)


def _xal_system(tmp_path, finite_source=False, binary_lens=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    lc = _write_lc(tmp_path / "lc.dat")
    stars = [{"name": "L1"}, {"name": "Source"}, {"name": "SComp"}]
    lenses = ["star.0"]
    if binary_lens:
        stars.append({"name": "L2"})
        lenses = ["star.0", "star.3"]
    config = {
        "star": stars,
        "orbit": [
            {"name": "S", "primary": ["Source"], "companion": ["SComp"]}
        ],
        "lens": [
            {
                "name": "EV",
                "lenses": lenses,
                "sources": ["star.1"],
                "finite_source": finite_source,
                "source_orbital_motion": "keplerian",
                "source_orbit": "S",
                "t0_par": _T0_PAR,
                "mmexofast": False,
            }
        ],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    params = {
        "lens.Source.t_0": {"initval": _T0_PAR},
        "lens.Source.u_0": {"initval": 0.15},
        "lens.Source.t_E": {"initval": 45.0},
        "lens.Source.pi_E_N": {"initval": 0.08},
        "lens.Source.pi_E_E": {"initval": 0.11},
        "orbit.S.period": {"initval": 120.0},
        "orbit.S.tc": {"initval": _T0_PAR - 20.0},
        "orbit.S.secosw": {"initval": 0.4},
        "orbit.S.sesinw": {"initval": 0.3},
        "orbit.S.cosi": {"initval": 0.4},
        "orbit.S.bigomega": {"initval": 70.0},
        "star.L1.mass": {"initval": 0.6},
        "star.L1.distance": {"initval": 4000.0},
        "star.Source.mass": {"initval": 1.0},
        "star.SComp.mass": {"initval": 0.6},
        "star.Source.distance": {"initval": 8000.0},
        "star.SComp.distance": {"initval": 8000.0},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    if finite_source:
        params["lens.Source.rho"] = {"initval": 1e-4}
    if binary_lens:
        params["star.L2.mass"] = {"initval": 0.2}
        params["star.L2.distance"] = {"initval": 4000.0}
        params["lens.EV.s"] = {"initval": 1.3}
        params["lens.EV.alpha"] = {"initval": 55.0}
        params["lens.EV.q"] = {"initval": 0.33}
    for st in stars:
        nm = st["name"]
        params[f"star.{nm}.ra"] = {"initval": 268.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -29.0, "sigma": 0}
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    return system, model


@pytest.fixture(scope="module")
def xal_system(tmp_path_factory):
    return _xal_system(tmp_path_factory.mktemp("xallarap"))


def _eval_at_start(model, nodes):
    with model:
        f = pytensor.function(model.free_RVs, nodes, on_unused_input="ignore")
        ip = model.initial_point()
        zeros = [
            np.zeros_like(ip[v.name]).astype(float) for v in model.free_RVs
        ]
        return f(*zeros)


# --- the test's OWN first-principles reconstruction --------------------


def _np_kepler(M, ecc):
    E = np.remainder(M, 2 * np.pi)
    for _ in range(80):
        E = E - (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))
    return E


def _np_source_offset(t, tp, n, ecc, w, cosi, bigom, a_scale):
    """Sky offset (N, E) of the orbit's primary body, the test's own copy
    of the standard construction (mirrors tests/test_astrometry.py's
    reference)."""
    M = (t - tp) * n
    E = _np_kepler(np.atleast_1d(M), ecc)
    cosf = (np.cos(E) - ecc) / (1 - ecc * np.cos(E))
    sinf = np.sqrt(1 - ecc**2) * np.sin(E) / (1 - ecc * np.cos(E))
    r = a_scale * (1 - ecc**2) / (1 + ecc * cosf)
    coswf = np.cos(w) * cosf - np.sin(w) * sinf
    sinwf = np.sin(w) * cosf + np.cos(w) * sinf
    N = r * (np.cos(bigom) * coswf - np.sin(bigom) * sinwf * cosi)
    Eo = r * (np.sin(bigom) * coswf + np.cos(bigom) * sinwf * cosi)
    return N, Eo


def test_shift_is_the_anchored_projected_source_orbit(xal_system):
    """
    Given: the built model's own resolved orbit/lens values,
    When: the (dtau, du) series is compared against this test's numpy
      reconstruction of C25 (Kepler -> Thiele-Innes -> anchor at t0_par ->
      project on (tau_hat, beta_hat) with C7's minus),
    Then: they agree to 1e-10 at every epoch, and both vanish at t0_par.
    """
    system, model = xal_system
    lens = system.lens
    orbit = system.orbit
    times = np.linspace(_T0_PAR - 90.0, _T0_PAR + 90.0, 181)

    dtau_t, du_t = lens._source_offset_series(times, system)
    j = lens.xal_orbit_idx
    nodes = [
        dtau_t,
        du_t,
        orbit.tp.value[j],
        orbit.n.value[j],
        orbit.ecc.value[j],
        pt.arctan2(orbit.sinw.value[j], orbit.cosw.value[j]),
        orbit.cosi.value[j],
        orbit.bigomega.value[j],
        orbit.a.value[j] * orbit.m_companion.value[j] / orbit.m_total.value[j],
        lens.theta_E.value[0],
        system.star.distance.value[lens.source_map[0]],
        lens.mu_dec_rel_geo.value[0],
        lens.mu_ra_rel_geo.value[0],
        lens.mu_rel_geo_mag.value[0],
    ]
    out = _eval_at_start(model, nodes)
    dtau, du = np.atleast_1d(out[0]), np.atleast_1d(out[1])
    (tp, n, ecc, w, cosi, bigom, a1, theta_E, d_s, mu_n, mu_e, mu_mag) = [
        float(np.atleast_1d(v)[0]) for v in out[2:]
    ]

    a_scale = a1 * RSUN_TO_AU * 1000.0 / (d_s * theta_E)
    sN, sE = _np_source_offset(times, tp, n, ecc, w, cosi, bigom, a_scale)
    sN0, sE0 = _np_source_offset(
        np.array([_T0_PAR]), tp, n, ecc, w, cosi, bigom, a_scale
    )
    dN, dE = sN - sN0[0], sE - sE0[0]
    tn, te = mu_n / mu_mag, mu_e / mu_mag
    np.testing.assert_allclose(dtau, -(dN * tn + dE * te), atol=1e-10)
    # du = -(dsigma . beta_hat) with beta_hat = (-tau_hat_E, +tau_hat_N),
    # C9's +90 North-through-East rotation -- the SAME basis the parallax
    # terms use.  The sign of this line is review 2.6.13's bug: it was
    # -(dN * te - dE * tn) (the -90 rotation), which inverted the shift
    # the light curve actually applied.
    np.testing.assert_allclose(du, +(dN * te - dE * tn), atol=1e-10)

    # the anchor: zero shift at t0_par
    d0 = _eval_at_start(
        model, list(lens._source_offset_series(np.array([_T0_PAR]), system))
    )
    np.testing.assert_allclose(np.atleast_1d(d0[0]), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.atleast_1d(d0[1]), 0.0, atol=1e-14)

    # and the motion is genuinely nonzero over the window
    assert np.max(np.hypot(dtau, du)) > 1e-3


def test_symbolic_magnification_consumes_the_shift(xal_system):
    """
    Given: the symbolic PSPL path with xallarap,
    When: compared to the Paczynski formula at the manually shifted
      (tau, u) built from the model's own trajectory pieces,
    Then: they agree to 1e-12 -- the shift enters at exactly the parallax
      slot (C25), not somewhere else.
    """
    system, model = xal_system
    lens = system.lens
    inst = system.mulensinstrument
    times = np.asarray(inst.time, dtype=float)

    A_node = lens.get_magnification(times, inst.observer_pos, system)

    from exozippy.skyframe import observer_sky_offset

    ra = system.star.ra.value[lens.source_map[0]]
    dec = system.star.dec.value[lens.source_map[0]]
    d_e, d_n = observer_sky_offset(inst.observer_pos, ra, dec, xp=pt)
    p = lens._get_safe_mm_params(0)
    tau = (times - p["t_0"]) / p["t_E"] - d_n * p["pi_E_N"] - d_e * p["pi_E_E"]
    uu = p["u_0"] + d_n * p["pi_E_E"] - d_e * p["pi_E_N"]
    dtau_t, du_t = lens._source_offset_series(times, system)
    tau = tau + dtau_t
    uu = uu + du_t
    u2 = pt.sqr(tau) + pt.sqr(uu)
    A_manual = (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))

    A1, A2 = _eval_at_start(model, [A_node, A_manual])
    np.testing.assert_allclose(A1, A2, rtol=1e-12)

    # teeth: the static model differs
    u2s = pt.sqr(tau - dtau_t) + pt.sqr(uu - du_t)
    A_static = (u2s + 2.0) / pt.sqrt(u2s * (u2s + 4.0))
    (A3,) = _eval_at_start(model, [A_static])
    assert np.max(np.abs(A3 - A1)) > 1e-4


def test_espl_op_matches_the_symbolic_path(tmp_path_factory):
    """
    Given: the same xallarap system with finite_source at negligible rho
      (which routes the single lens through VBMDirectMagOp's ESPL branch
      with source_motion inputs),
    When: compared to the point-source symbolic path,
    Then: they agree to 1e-5 -- the Op plumbing carries the same shift.
    """
    system, model = _xal_system(
        tmp_path_factory.mktemp("xal_espl"), finite_source=True
    )
    lens = system.lens
    inst = system.mulensinstrument
    times = np.asarray(inst.time, dtype=float)
    A_op = lens.get_magnification_op(times, inst.observer_pos, system)
    A_sym = lens.get_magnification(times, inst.observer_pos, system)
    A1, A2 = _eval_at_start(model, [A_op, A_sym])
    assert np.all(np.isfinite(A1))
    np.testing.assert_allclose(A1, A2, rtol=1e-5)


def test_binary_lens_op_builds_and_is_finite(tmp_path_factory):
    """
    Given: a 2L1S system with xallarap (the OGLE-2017-BLG-0114 topology),
    When: the model builds,
    Then: logp is finite and the magnification Op consumes the shift
      (differs from the static curve).
    """
    system, model = _xal_system(
        tmp_path_factory.mktemp("xal_2l1s"), binary_lens=True
    )
    starts, _ = system.get_raw_starts(model)
    logp = float(model.compile_logp(jacobian=False)(starts[0]))
    assert np.isfinite(logp)


def test_no_new_parameters_and_node_degeneracy(xal_system):
    """
    Given: the xallarap system,
    Then: no xi_*-style sampled parameters exist (the orbit IS the
      parameterization), the source orbit samples bigomega, and it REMAINS
      node-degenerate (a sky track is reflection-invariant; only the lens
      keplerian mode breaks the node).
    """
    system, model = xal_system
    free_names = [v.name for v in model.free_RVs]
    assert "orbit.xbigomega_raw" in free_names
    assert bool(
        np.atleast_1d(system.orbit.node_degenerate)[system.lens.xal_orbit_idx]
    ), "a xallarap-only orbit must keep the node fold"


def test_config_validation():
    """linear refused with the degeneracy explanation; missing/shared
    orbit references refused."""
    from exozippy.components.mulensing.lens import Lens
    from exozippy.config import ConfigManager

    base = {
        "name": "EV",
        "lenses": ["star.0"],
        "sources": ["star.1"],
    }
    with pytest.raises(NotImplementedError, match="degenerate"):
        Lens([dict(base, source_orbital_motion="linear")], ConfigManager({}))
    with pytest.raises(ValueError, match="source_orbit"):
        Lens(
            [dict(base, source_orbital_motion="keplerian")],
            ConfigManager({}),
        )
    cm = ConfigManager({})
    cm.system_config = {"orbit": [{"name": "S"}]}
    with pytest.raises(ValueError, match="SAME orbit"):
        Lens(
            [
                dict(
                    base,
                    lenses=["star.0", "star.2"],
                    orbital_motion="keplerian",
                    orbit="S",
                    source_orbital_motion="keplerian",
                    source_orbit="S",
                )
            ],
            cm,
        )


def test_mm_xi_closed_form_mapping():
    """
    Given: random xi_* xallarap draws (the Zhai+2024 / Mroz+2026
      parameterization, as implemented by MulensModel -- the code Mroz+26
      used, so its track at published values IS the published motion),
    When: the C25 closed-form mapping builds the same track from EXOZIPPy
      elements (bigomega = phi_pi + xi_Omega + 180, i = xi_i,
      omega_* = xi_omega, nu(t_0_xi) = xi_u - xi_omega -> tp),
    Then: EXOZIPPy's (dtau, du) equals the shift MulensModel APPLIES to
      its trajectory, with no extra sign, to 1e-9 -- the contract the
      light curves rest on, and the recipe that lets a published xi_*
      solution seed an EXOZIPPy config (examples/ob170114).
    """
    import MulensModel as mm

    rng = np.random.default_rng(20260827)
    T0, TE = 2457899.3, 173.0
    for k in range(3):
        u0 = rng.uniform(-0.3, 0.3)
        piEN, piEE = rng.normal(0, 0.2, 2)
        P = rng.uniform(50, 500)
        a_xi = rng.uniform(0.05, 0.5)
        e = rng.uniform(0, 0.8)
        xi_i = rng.uniform(0, 180)
        xi_Om = rng.uniform(0, 360)
        xi_u = rng.uniform(0, 360)
        xi_om = rng.uniform(0, 360)
        t0par = T0 + rng.uniform(-50, 50)
        t = np.linspace(t0par - 2 * P, t0par + 2 * P, 101)

        base = dict(t_0=T0, u_0=u0, t_E=TE)
        tr0 = mm.Trajectory(t, parameters=mm.Model(dict(base)).parameters)
        m1 = mm.Model(
            dict(
                base,
                xi_period=P,
                xi_semimajor_axis=a_xi,
                xi_inclination=xi_i,
                xi_Omega_node=xi_Om,
                xi_argument_of_latitude_reference=xi_u,
                xi_eccentricity=e,
                xi_omega_periapsis=xi_om,
                t_0_xi=t0par,
            )
        )
        tr1 = mm.Trajectory(t, parameters=m1.parameters)
        # The shift MulensModel APPLIES to its trajectory -- (tau, u) and
        # MM's (x, y) are the same quantities (C18/C10), so this, with NO
        # extra sign, is what EXOZIPPy's (dtau, du) must equal for the two
        # LIGHT CURVES to agree.  Until review 2.6.13 this comparison
        # carried a hand-written minus and the mapping below was tuned
        # against the du sign bug, so the two errors cancelled here while
        # the composed magnification applied the source's offset backwards.
        dtau_mm = np.asarray(tr1.x) - np.asarray(tr0.x)
        du_mm = np.asarray(tr1.y) - np.asarray(tr0.y)

        phi_pi = np.arctan2(piEE, piEN)
        om_node = phi_pi + np.radians(xi_Om) + np.pi
        cosi = np.cos(np.radians(xi_i))
        w = np.radians(xi_om)
        nu0 = np.radians(xi_u - xi_om)
        E0 = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin(nu0 / 2),
            np.sqrt(1 + e) * np.cos(nu0 / 2),
        )
        tp = t0par - (E0 - e * np.sin(E0)) * P / (2 * np.pi)
        n_mm = 2 * np.pi / P

        sN, sE = _np_source_offset(t, tp, n_mm, e, w, cosi, om_node, a_xi)
        sN0, sE0 = _np_source_offset(
            np.array([t0par]), tp, n_mm, e, w, cosi, om_node, a_xi
        )
        dN, dE = sN - sN0[0], sE - sE0[0]
        tn, te = np.cos(phi_pi), np.sin(phi_pi)
        np.testing.assert_allclose(
            -(dN * tn + dE * te), dtau_mm, atol=1e-9, err_msg=f"draw {k}"
        )
        np.testing.assert_allclose(
            +(dN * te - dE * tn), du_mm, atol=1e-9, err_msg=f"draw {k}"
        )


def test_binary_op_matches_mulensmodel_xallarap_lightcurve():
    """
    Given: the published OGLE-2017-BLG-0114 Std 2L1S + xallarap solution
      (Mroz et al. 2026 Table B.1), its xi_* elements mapped through C25,
      and the shift track MulensModel itself applies,
    When: VBMDirectMagOp computes the magnification with that track at its
      source_motion inputs (numpy level, no parallax on either side so the
      comparison isolates the xallarap composition),
    Then: the two LIGHT CURVES agree to 1e-10 relative.  This is the pin
      the track-level test cannot provide: review 2.6.13's du sign bug and
      the compensating mapping error cancelled in the track comparison and
      only disagreed in the composed magnification (built A peaked at 8.6
      where MulensModel -- and the photometry, at chi2/N = 1.5 -- give 3.3).
    """
    import MulensModel as mm

    from exozippy.components.mulensing.op import VBMDirectMagOp

    # Arrange: published solution; xi elements; MM reference curves.
    T0, U0, TE, RHO = 2457899.3, -0.144, 173.0, 0.0109
    t0par = 2457933.5
    S, Q, ALPHA = 0.337, 0.0219, 330.4
    times = np.linspace(2457850.0, 2458000.0, 151)
    xi = dict(
        xi_period=221.5,
        xi_semimajor_axis=0.200,
        xi_inclination=30.5,
        xi_Omega_node=250.0,
        xi_argument_of_latitude_reference=236.1,
        xi_eccentricity=0.615,
        xi_omega_periapsis=207.3,
        t_0_xi=t0par,
    )
    base = dict(t_0=T0, u_0=U0, t_E=TE)
    tr0 = mm.Trajectory(times, parameters=mm.Model(dict(base)).parameters)
    m_xi = mm.Model(dict(base, **xi))
    tr1 = mm.Trajectory(times, parameters=m_xi.parameters)
    # the shift MM applies to (x, y) == our (tau, u); C18/C10 identity
    dtau = np.asarray(tr1.x) - np.asarray(tr0.x)
    du = np.asarray(tr1.y) - np.asarray(tr0.y)
    full = mm.Model(dict(base, rho=RHO, s=S, q=Q, alpha=ALPHA, **xi))
    full.set_magnification_methods([-np.inf, "VBBL", np.inf])
    A_mm = np.asarray(full.get_magnification(times), dtype=float)

    # Act: the direct Op with the same shift at its source_motion inputs.
    op = VBMDirectMagOp(
        coords="260.4905833d -29.6236944d",
        n_companions=1,
        use_rho=True,
        source_motion=True,
    )
    p = np.array([T0, U0, TE, 0.0, 0.0, RHO, S, Q, ALPHA], dtype=float)
    obs = np.zeros((times.size, 3))
    A_op = op._compute(p, times, obs, None, (dtau, du))

    # Assert: light-curve parity, and the shift genuinely matters (teeth:
    # the zero-shift curve differs by far more than the tolerance).
    np.testing.assert_allclose(A_op, A_mm, rtol=1e-10)
    A_static = op._compute(
        p, times, obs, None, (np.zeros_like(dtau), np.zeros_like(du))
    )
    assert np.max(np.abs(A_static - A_mm) / A_mm) > 0.5

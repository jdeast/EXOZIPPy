"""Raw-redshift RVs (rvinstrument ``data_format: raw_redshift``) in
multi-orbit systems, checked against an INDEPENDENT numpy forward model.

The likelihood composes the star's reflex from every orbit it is a body of
(``RVInstrument._rv_model``, ``_reflex_sky``) and retards the emission by
the star's own line-of-sight displacement summed over those orbits
(``RVInstrument._emission_time``, through ``components/ltt.py``).  Here the
same star velocity is composed BY HAND from each orbit's elements at the
model's start -- its own Kepler solve, its own barycentric shares and
signs, its own fixed-point solve of ``t_emit = t_B - Z_star(t_emit)/c`` --
and fed to the shared Doppler relation (``redshift.predicted_cz``, itself
validated against barycorrpy in tests/test_raw_redshift.py).  So what is
tested is exactly the orbit composition and the retardation:

* two planets around one star, the outer one on a Jacobi orbit whose
  primary group is [A, b] (the star is in BOTH orbits' primary groups);
* a binary's SECONDARY: the observed star on the companion side;
* the size of the retardation (rvinstrument.md, "The star's own light
  travel time").
"""

import numpy as np
import pytest

from exozippy.components.rvinstrument.redshift import (
    observer_terms,
    predicted_cz,
)
from exozippy.constants import C_MPS

pytestmark = pytest.mark.slow

SITE = "-104.0147,30.6814,2026"
EPOCH = 2457389.0
RA, DEC = 209.5063, 31.6955
SOLRAD_M = 6.957e8
GM_SUN = 1.3271244e20


def _times(n=40, span=(2459950.0, 2460700.0), seed=3):
    return np.sort(np.random.default_rng(seed).uniform(*span, n))


def _kepler(t, tp, P, e, w):
    """(r/a, sin(w+f), cos(w+f), vz_phase) of a Keplerian at times t."""
    M = 2 * np.pi * (t - tp) / P
    E = M.copy()
    for _ in range(80):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    f = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2)
    )
    return (
        1 - e * np.cos(E),
        np.sin(w + f),
        np.cos(w + f),
        np.cos(w + f) + e * np.cos(w),
    )


def _star_motion(t, orbits):
    """The star's line-of-sight position (m, AWAY from the observer) and
    velocity (m/s) at target-frame times t, composed from ``orbits``:
    dicts with tp, P [d], e, w [rad], a [m, relative orbit], sini, share (signed:
    +M_c/M on an orbit's primary side, -M_p/M on its companion side)."""
    Z = np.zeros_like(t)
    V = np.zeros_like(t)
    for o in orbits:
        r_a, s_wf, _, vphase = _kepler(t, o["tp"], o["P"], o["e"], o["w"])
        n = 2 * np.pi / (o["P"] * 86400.0)
        Z = Z + o["share"] * o["a"] * r_a * s_wf * o["sini"]
        V = (
            V
            + o["share"]
            * n
            * o["a"]
            * o["sini"]
            / np.sqrt(1 - o["e"] ** 2)
            * vphase
        )
    return Z, V


def _independent_cz(jd_utc, orbits, star, gamma_ms, rv, retard=True):
    """c z_meas predicted for the star at ``star`` (internal units) with
    reflex composed by ``_star_motion`` at the RETARDED time."""
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    lon, lat, h = (float(x) for x in SITE.split(","))
    loc = EarthLocation.from_geodetic(lon, lat, h)
    t_tdb = Time(jd_utc, format="jd", scale="utc", location=loc).tdb.jd
    obs = observer_terms(t_tdb, SITE, "builtin")

    def emit(t_bary):
        t_emit = np.array(t_bary, dtype=float)
        if retard:
            for _ in range(8):
                Z, _ = _star_motion(t_emit, orbits)
                t_emit = t_bary - Z / C_MPS / 86400.0
        return t_emit

    def reflex(t_bary):
        _, V = _star_motion(emit(t_bary), orbits)
        return V, None, None

    def potential(t_bary):
        # The companions' potential at the star: G M_other / (r c^2).
        t_emit = emit(t_bary)
        phi = np.zeros_like(t_emit)
        for o in orbits:
            r_a, _, _, _ = _kepler(t_emit, o["tp"], o["P"], o["e"], o["w"])
            phi = phi + o["gm_other"] / (o["a_rel"] * r_a * C_MPS**2)
        return phi

    z0 = (1 + gamma_ms / C_MPS) / (1 + rv / C_MPS) - 1
    cz, _ = predicted_cz(
        t_tdb, obs, star, EPOCH, z0=z0, v_reflex=reflex, phi_star=potential
    )
    return cz


def _write(path, jd, cz):
    np.savetxt(path, np.column_stack([jd, cz, np.full(jd.size, 5.0)]))
    return str(path)


def _star_params(names):
    p = {}
    for nm, m in names.items():
        p.update(
            {
                f"star.{nm}.mass": {"initval": m, "sigma": 0.02},
                f"star.{nm}.radius": {"initval": max(m, 0.1), "sigma": 0.02},
                f"star.{nm}.ra": {"initval": RA},
                f"star.{nm}.dec": {"initval": DEC},
            }
        )
    first = next(iter(names))
    p.update(
        {
            f"star.{first}.pm_ra": {"initval": -75.53},
            f"star.{first}.pm_dec": {"initval": 17.97},
            f"star.{first}.distance": {"initval": 73.37},
            f"star.{first}.rv": {
                "initval": -2.0e4,
                "mu": -2.0e4,
                "sigma": 500.0,
            },
        }
    )
    return p


def _raw_entry(path, star=None):
    e = {
        "name": "HPF",
        "file": path,
        "data_format": "raw_redshift",
        "observer_location": SITE,
        "time_frame": "jd",
        "time_scale": "utc",
        "epoch": EPOCH,
    }
    if star is not None:
        e["star_ndx"] = star
    return e


def _build(cfg, params):
    from exozippy.system import System

    system = System(cfg, user_params=params)
    system.prepare()
    model = system.build_model()
    return system, model, system.get_raw_start(model)


def _at(model, point, nodes):
    import pytensor

    fn = pytensor.function(model.free_RVs, nodes, on_unused_input="ignore")
    return [
        np.asarray(x) for x in fn(*[point[v.name] for v in model.free_RVs])
    ]


def _elements(system, model, start, shares):
    """The orbits' start elements, as numpy, for the independent model."""
    o = system.orbit
    names = ["tp_target", "period", "ecc", "omega", "a", "sini"]
    tp, P, e, w, a, si = _at(
        model, start, [getattr(o, k).value for k in names]
    )
    mp, mc, mt = _at(
        model, start, [o.m_primary.value, o.m_companion.value, o.m_total.value]
    )
    out = []
    for k, side in shares.items():
        share = mc[k] / mt[k] if side == "primary" else -mp[k] / mt[k]
        m_other = mc[k] if side == "primary" else mp[k]
        out.append(
            {
                "tp": tp[k],
                "P": P[k],
                "e": e[k],
                "w": w[k],
                "a": a[k] * SOLRAD_M,
                "sini": si[k],
                "share": share,
                # the RELATIVE orbit (the star's share above is of it)
                "a_rel": a[k] * SOLRAD_M,
                "gm_other": GM_SUN * m_other,
            }
        )
    return out


def _model_cz(system, model, start):
    rvi = system.rvinstrument
    (mod,) = _at(model, start, [rvi._rv_model_data_node])
    return mod * rvi._rv_factor()


def _star_start(system, model, start, s):
    st = system.star
    vals = _at(
        model,
        start,
        [
            st.ra.value,
            st.dec.value,
            st.pm_ra.value,
            st.pm_dec.value,
            st.parallax.value,
            st.rv.value,
        ],
    )
    return tuple(float(v[s]) for v in vals)


@pytest.fixture(scope="module")
def two_planet_jacobi(tmp_path_factory):
    """Star A with planet b (12 d) and planet c on a Jacobi orbit whose
    primary group is [A, b] (400 d)."""
    d = tmp_path_factory.mktemp("rawz_jacobi")
    jd = _times()
    path = _write(d / "A.rv", jd, np.full(jd.size, -2.0e4))
    cfg = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}, {"name": "c"}],
        "orbit": [
            {"name": "b", "primary": ["A"], "companion": ["b"]},
            {"name": "c", "primary": ["A", "b"], "companion": ["c"]},
        ],
        "rvinstrument": [_raw_entry(path)],
    }
    p = _star_params({"A": 1.0})
    p.update(
        {
            "orbit.b.period": {"initval": 12.0},
            "orbit.b.tc": {"initval": 2460300.0},
            "orbit.b.ecc": {"initval": 0.2},
            "orbit.b.omega": {"initval": 70.0},
            "orbit.c.period": {"initval": 400.0},
            "orbit.c.tc": {"initval": 2460200.0},
            "orbit.c.ecc": {"initval": 0.4},
            "orbit.c.omega": {"initval": 200.0},
            "planet.b.K": {"initval": 300.0},
            "planet.c.K": {"initval": 150.0},
            "rvinstrument.HPF.gamma": {"initval": -19990.0},
        }
    )
    return _build(cfg, p), jd


def test_two_planets_and_a_jacobi_orbit_match_the_independent_model(
    two_planet_jacobi,
):
    """
    Given a star in BOTH orbits' primary groups (b around A; c around
      [A, b]),
    When the raw-redshift model is evaluated at the start,
    Then it equals the independent numpy composition -- each orbit's
      reflex at the star's share M_c/M, Jacobi masses for c, both summed
      into one velocity at the retarded time -- to < 0.1 mm/s.
    """
    (system, model, start), jd = two_planet_jacobi
    assert [r for _, r in system.orbit.star_membership(0)] == [
        "primary",
        "primary",
    ]
    orbits = _elements(system, model, start, {0: "primary", 1: "primary"})
    (gamma,) = _at(model, start, [system.rvinstrument.gamma.value])
    star = _star_start(system, model, start, 0)
    gamma_ms = float(gamma[0]) * system.rvinstrument._rv_factor()
    ind = _independent_cz(jd, orbits, star, gamma_ms, star[5])
    mod = _model_cz(system, model, start)
    assert np.max(np.abs(mod - ind)) < 1e-4
    # and the composition is not trivially one orbit: dropping c moves it
    # by its full ~K.
    alone = _independent_cz(jd, orbits[:1], star, gamma_ms, star[5])
    assert np.max(np.abs(alone - ind)) > 50.0


@pytest.fixture(scope="module")
def binary_secondary(tmp_path_factory):
    """A close binary (A 1.0, B 0.3 Msun, P = 3 d) observed in B's light:
    the observed star on the COMPANION side."""
    d = tmp_path_factory.mktemp("rawz_binary")
    jd = _times(span=(2460000.0, 2460030.0))
    path = _write(d / "B.rv", jd, np.full(jd.size, -2.0e4))
    cfg = {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "orbit": [{"name": "AB", "primary": ["A"], "companion": ["B"]}],
        "rvinstrument": [_raw_entry(path, star="B")],
    }
    p = _star_params({"A": 1.0, "B": 0.3})
    p.update(
        {
            "orbit.AB.period": {"initval": 3.0},
            "orbit.AB.tc": {"initval": 2460010.0},
            "orbit.AB.ecc": {"initval": 0.1},
            "orbit.AB.omega": {"initval": 40.0},
            "orbit.AB.cosi": {"initval": 0.2},
            "rvinstrument.HPF.gamma": {"initval": -19990.0},
        }
    )
    return _build(cfg, p), jd


def test_a_binary_secondary_matches_the_independent_model(binary_secondary):
    """
    Given the observed star on the companion side of a 3-day binary,
    When the raw-redshift model is evaluated at the start,
    Then it equals the independent composition with share -M_A/M (the
      secondary moves opposite to the primary, scaled by the mass ratio)
      and the secondary's own retardation, to < 1 mm/s of ~100 km/s; and
      the retardation it carries is the ~0.1-1 m/s a 3-day binary's
      light-travel time is worth.
    """
    (system, model, start), jd = binary_secondary
    assert system.orbit.star_membership(1) == [(0, "companion")]
    orbits = _elements(system, model, start, {0: "companion"})
    (gamma,) = _at(model, start, [system.rvinstrument.gamma.value])
    star = _star_start(system, model, start, 1)
    gamma_ms = float(gamma[0]) * system.rvinstrument._rv_factor()
    ind = _independent_cz(jd, orbits, star, gamma_ms, star[5])
    unretarded = _independent_cz(
        jd, orbits, star, gamma_ms, star[5], retard=False
    )
    mod = _model_cz(system, model, start)
    assert np.max(np.abs(mod - ind)) < 1e-3
    effect = np.max(np.abs(ind - unretarded))
    print(f"binary secondary: retardation moves c*z by up to {effect:.3f} m/s")
    assert effect > 0.05

"""Taylor orbits: `orbit: type: linear | quadratic` (review 8.8.14).

An orbit too long for the data to resolve is written as the low-order time
derivatives of what each consumer measures, about a reference epoch.  It is
an Orbit -- the same block, the same body groups -- whose Keplerian elements
are INACTIVE (it must self-declare that it has no period, mass or K) and
whose Taylor coefficients are active exactly where a consumer reads them:
`gammadot` (and `gammaddot`) for the RVs of a star in its primary group,
`ds_dt`/`dalpha_dt` for a lens companion (tests/test_lens_orbital_motion.py).

Pinned here: the RV model the trend adds, the epoch rule, the
self-declaration, the refusals of every consumer that cannot read one, and
the data seed.
"""

import numpy as np
import pytest

from exozippy.components.parameterization import restrict_active
from exozippy.system import System

_EPOCH_SPAN = (2459600.0, 2459760.0)
_K_FREE_TRUTH = {"gamma": 310.0, "gammadot": -1.0, "gammaddot": 0.004}


def _write_rv(path, gammadot=0.0, gammaddot=0.0, n=25, seed=3):
    """A trend-only RV file (m/s) spanning _EPOCH_SPAN."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(*_EPOCH_SPAN, n))
    t[0], t[-1] = _EPOCH_SPAN
    mid = 0.5 * (t[0] + t[-1])
    rv = (
        _K_FREE_TRUTH["gamma"]
        + gammadot * (t - mid)
        + 0.5 * gammaddot * (t - mid) ** 2
    )
    np.savetxt(path, np.column_stack([t, rv, np.full(n, 5.0)]))
    return str(path)


def _config(rv_file, orbits, planets=None, rv_extra=None):
    cfg = {
        "star": [{"name": "A", "mist": False}],
        "orbit": orbits,
        "rvinstrument": [
            {"name": "TRES", "file": rv_file, **(rv_extra or {})}
        ],
    }
    if planets is not None:
        cfg["planet"] = planets
    return cfg


_STAR = {
    "star.A.mass": {"initval": 1.2, "sigma": 0.05},
    "star.A.radius": {"initval": 1.3, "sigma": 0.05},
}


def _system(cfg, params=None, build=True):
    p = dict(_STAR)
    p.update(params or {})
    system = System(cfg, user_params=p)
    system.prepare()
    model = system.build_model() if build else None
    return system, model


def _at_start(system, model, node):
    """``node`` evaluated at the fit's start (the raw start fed in for the
    free RVs -- Parameter.value would otherwise draw from the prior)."""
    import pytensor

    start = system.get_raw_start(model)
    fn = pytensor.function(model.free_RVs, node, on_unused_input="ignore")
    return np.asarray(fn(*[start[v.name] for v in model.free_RVs]))


# ---------------------------------------------------------------------------
# 1. The RV model a Taylor orbit adds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("otype", ["linear", "quadratic"])
def test_the_trend_is_the_taylor_series_about_the_epoch(tmp_path, otype):
    """
    Given a star whose only orbit is a Taylor orbit, with seeded coefficients,
    When the RV model is evaluated at the start,
    Then it is exactly gamma + gammadot (t - epoch) [+ gammaddot (t -
      epoch)**2 / 2] -- a DERIVATIVE, not EXOFASTv2's QUAD coefficient --
      with no constant term of its own (gamma carries it).
    """
    rv = _write_rv(tmp_path / "t.rv")
    orbit = {"name": "trend", "type": otype, "primary": ["A"]}
    params = {
        "orbit.trend.gammadot": {"initval": -1.25},
        "rvinstrument.TRES.gamma": {"initval": 300.0},
    }
    if otype == "quadratic":
        params["orbit.trend.gammaddot"] = {"initval": 0.006}
    system, model = _system(_config(rv, [orbit]), params)
    rvi = system.rvinstrument

    model_ms = (
        _at_start(system, model, rvi._rv_model_data_node) * rvi._rv_factor()
    )
    epoch = system.orbit.taylor_epoch[0]
    dt = rvi.time - epoch
    expected = 300.0 - 1.25 * dt
    if otype == "quadratic":
        expected = expected + 0.5 * 0.006 * dt**2
    # atol: the start round-trips through the logit raw coordinate (~1e-8).
    np.testing.assert_allclose(model_ms, expected, rtol=1e-10, atol=1e-6)
    # One column per orbit, the Taylor orbit's included, so the phased
    # panels of any Keplerian orbit subtract it as an "other orbit".
    assert list(rvi._plot_orbit_map) == [0]


def test_a_slope_reports_its_companion_mass_over_r2_bound(tmp_path):
    """
    Given an RV-read linear Taylor orbit with a seeded slope,
    When the model is built,
    Then orbit.<trend>.mc_over_r2_min is |gammadot| / G in M_J/AU^2 --
      1 M_J at 1 AU pulls a star at 0.489 m/s/day -- reported as a derived
      bound and entering no potential.
    """
    rv = _write_rv(tmp_path / "t.rv")
    orbit = {"name": "trend", "type": "linear", "primary": ["A"]}
    system, model = _system(
        _config(rv, [orbit]), {"orbit.trend.gammadot": {"initval": -0.8}}
    )
    orb = system.orbit
    got = orb.mc_over_r2_min.from_internal(
        _at_start(system, model, orb.mc_over_r2_min.value)
    )
    np.testing.assert_allclose(
        np.atleast_1d(got)[0], 0.8 / 0.48909515328448067, rtol=1e-6
    )
    assert not any("mc_over_r2_min" in p.name for p in model.potentials), (
        "a bound is reported, never a potential"
    )


def test_a_trend_beside_a_planet_adds_to_the_keplerian(tmp_path):
    """
    Given a planet's Keplerian orbit and a linear Taylor orbit on one star,
    When the RV model is built,
    Then the per-orbit matrix has the Keplerian column first and the trend
      column second, summing to the model minus gamma, and the trend column
      is gammadot (t - epoch).
    """
    rv = _write_rv(tmp_path / "t.rv")
    cfg = _config(
        rv,
        [
            {"name": "b", "primary": ["A"], "companion": ["b"]},
            {"name": "trend", "type": "linear", "primary": ["A"]},
        ],
        planets=[{"name": "b"}],
    )
    params = {
        "orbit.b.period": {"initval": 7.43},
        "orbit.b.tc": {"initval": 2459650.0},
        "planet.b.mass": {"initval": 3.0},
        "orbit.trend.gammadot": {"initval": 0.8},
    }
    system, model = _system(cfg, params)
    rvi = system.rvinstrument
    f = rvi._rv_factor()
    matrix = _at_start(system, model, rvi._rv_matrix_data_node) * f
    total = _at_start(system, model, rvi._rv_model_data_node) * f
    gamma = _at_start(system, model, rvi.gamma.value) * f
    assert list(rvi._plot_orbit_map) == [0, 1]
    np.testing.assert_allclose(matrix.sum(axis=1), total - gamma[0], atol=1e-8)
    dt = rvi.time - system.orbit.taylor_epoch[1]
    np.testing.assert_allclose(matrix[:, 1], 0.8 * dt, rtol=1e-10, atol=1e-6)
    assert np.ptp(matrix[:, 0]) > 1.0  # the planet is really there


# ---------------------------------------------------------------------------
# 2. The epoch
# ---------------------------------------------------------------------------


def test_the_default_epoch_is_the_midpoint_of_the_rvs(tmp_path):
    """
    Given no `epoch:`,
    Then the reference epoch is (min + max)/2 of the primary star's RVs --
      EXOFASTv2's RVEPOCH default (mkss.pro).
    """
    rv = _write_rv(tmp_path / "t.rv")
    orbit = {"name": "trend", "type": "linear", "primary": ["A"]}
    system, _ = _system(_config(rv, [orbit]), build=False)
    assert system.orbit.taylor_epoch[0] == pytest.approx(
        0.5 * sum(_EPOCH_SPAN), abs=1e-9
    )


def test_a_user_epoch_is_honored_and_shown_in_the_table_note(tmp_path):
    """
    Given `epoch:` on the orbit block,
    Then it is the expansion's epoch, and gammadot's table note states it.
    """
    rv = _write_rv(tmp_path / "t.rv")
    orbit = {
        "name": "trend",
        "type": "linear",
        "primary": ["A"],
        "epoch": 2459700.25,
    }
    system, _ = _system(_config(rv, [orbit]), build=False)
    assert system.orbit.taylor_epoch[0] == 2459700.25
    note = system.orbit.manifest["gammadot"]["table_note"]
    assert "2459700.250000" in note


# ---------------------------------------------------------------------------
# 3. A Taylor orbit has no Keplerian elements
# ---------------------------------------------------------------------------


def test_the_keplerian_elements_are_inactive_on_a_taylor_orbit(tmp_path):
    """
    Given a planet orbit and a Taylor orbit,
    When the model is built,
    Then every sampled orbit coordinate has ONE element (the planet's), the
      Taylor orbit samples only gammadot, and none of its Keplerian
      elements -- period, ecc, K, masses -- is active (so none is reported).
    """
    rv = _write_rv(tmp_path / "t.rv")
    cfg = _config(
        rv,
        [
            {"name": "b", "primary": ["A"], "companion": ["b"]},
            {"name": "trend", "type": "linear", "primary": ["A"]},
        ],
        planets=[{"name": "b"}],
    )
    params = {
        "orbit.b.period": {"initval": 7.43},
        "orbit.b.tc": {"initval": 2459650.0},
    }
    system, model = _system(cfg, params)
    ip = model.initial_point()
    for name, value in ip.items():
        if name.startswith("orbit.") and name != "orbit.gammadot_raw":
            assert np.shape(value) == (1,), name
    assert np.shape(ip["orbit.gammadot_raw"]) == (1,)
    orbit = system.orbit
    for name in ("period", "ecc", "K", "m_primary", "m_companion", "a", "tc"):
        par = getattr(orbit, name)
        assert par.element_is_active(0), name
        assert not par.element_is_active(1), name
    assert not orbit.gammadot.element_is_active(0)
    assert orbit.gammadot.element_is_active(1)
    # The slope's bound lives where the slope does.
    assert not orbit.mc_over_r2_min.element_is_active(0)
    assert orbit.mc_over_r2_min.element_is_active(1)


def test_restrict_active_leaves_an_all_active_entry_untouched():
    """
    Given every element active,
    Then restrict_active hands back the very same entry -- which is what
      keeps a system without a Taylor orbit building exactly the graph it
      always did.
    """
    entry = {"expr_key": "default", "force_node": True}
    assert restrict_active(entry, np.ones(3, bool), 3) is entry


def test_restrict_active_narrows_every_selector():
    """
    Given a mode-split entry and a kind mask,
    Then the mask is ANDed in and each expression selector is narrowed to
      the surviving elements (an element both inactive and derived is a
      contradiction the interpreter refuses), a bare expr_key included.
    """
    keep = np.array([True, True, False])
    out = restrict_active("default", keep, 3)
    assert out["mask"].tolist() == [True, True, False]
    assert out["expr_key"]["default"].tolist() == [True, True, False]
    split = {
        "expr_key": {"from_vcve": np.array([False, False, True])},
        "output_expr_key": {"from_ecc": np.array([True, False, True])},
        "mask": np.array([True, False, True]),
    }
    out = restrict_active(split, keep, 3)
    assert out["mask"].tolist() == [True, False, False]
    assert "expr_key" not in out  # its only element was the removed one
    assert out["output_expr_key"]["from_ecc"].tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# 4. The refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block, exc, match",
    [
        ({"type": "Linear"}, ValueError, "case-sensitive"),
        ({"type": "nbody"}, NotImplementedError, "nbody"),
        ({"type": "linear", "fitvcve": True}, ValueError, "Keplerian"),
        ({"epoch": 2459700.0}, ValueError, "epoch"),
    ],
)
def test_invalid_orbit_blocks_raise(tmp_path, block, exc, match):
    """
    Given a bad `type:`, a reserved one, a Keplerian-only key on a Taylor
      orbit, or a Taylor-only key on a Keplerian one,
    Then construction raises -- a silently inert key is a config the user
      believes is in effect.
    """
    rv = _write_rv(tmp_path / "t.rv")
    orbit = {"name": "o", "primary": ["A"], **block}
    with pytest.raises(exc, match=match):
        System(_config(rv, [orbit]), user_params=dict(_STAR))


def test_a_taylor_orbit_needs_an_explicit_primary(tmp_path):
    """
    Given a Taylor orbit with no `primary:`,
    Then it raises: the implicit planet pairing would invent a companion.
    """
    rv = _write_rv(tmp_path / "t.rv")
    with pytest.raises(ValueError, match="primary"):
        System(
            _config(rv, [{"name": "trend", "type": "linear"}]),
            user_params=dict(_STAR),
        )


def test_an_unread_taylor_orbit_raises(tmp_path):
    """
    Given a Taylor orbit whose primary is not the RV star,
    Then stage 3 raises: its coefficients would be dimensions no likelihood
      term constrains.
    """
    rv = _write_rv(tmp_path / "t.rv")
    cfg = _config(
        rv,
        [
            {"name": "trend", "type": "linear", "primary": ["B"]},
            {"name": "AB", "primary": ["A"], "companion": ["B"]},
        ],
    )
    cfg["star"].append({"name": "B", "mist": False})
    with pytest.raises(ValueError, match="nothing in the model reads"):
        _system(cfg, {"star.B.mass": {"initval": 0.5}}, build=False)


def test_the_rv_star_on_the_companion_side_is_refused(tmp_path):
    """
    Given the RV star in a Taylor orbit's COMPANION group,
    Then stage 3 raises: its trend is the primary's scaled by a mass ratio
      a Taylor orbit does not have.
    """
    rv = _write_rv(tmp_path / "t.rv")
    cfg = _config(
        rv,
        [
            {
                "name": "trend",
                "type": "linear",
                "primary": ["B"],
                "companion": ["A"],
            }
        ],
    )
    cfg["star"].append({"name": "B", "mist": False})
    with pytest.raises(NotImplementedError, match="COMPANION"):
        _system(cfg, {"star.B.mass": {"initval": 0.5}}, build=False)


def test_a_planet_on_a_taylor_orbit_is_refused(tmp_path):
    """
    Given a planet whose orbit_ndx points at a Taylor orbit,
    Then stage 3 raises: its transit and derived geometry need a Keplerian.
    """
    rv = _write_rv(tmp_path / "t.rv")
    cfg = _config(
        rv,
        [
            {
                "name": "trend",
                "type": "linear",
                "primary": ["A"],
                "companion": ["b"],
            }
        ],
        planets=[{"name": "b", "orbit_ndx": 0}],
    )
    with pytest.raises(ValueError, match="Keplerian orbit"):
        _system(cfg, build=False)


# ---------------------------------------------------------------------------
# 5. The data seed
# ---------------------------------------------------------------------------


def test_the_coefficients_are_seeded_from_the_rvs(tmp_path):
    """
    Given trend-only RVs with a known slope and curvature and no params-file
      seed,
    When the system is prepared,
    Then gammadot and gammaddot start at the polynomial fit -- EXOFASTv2's
      start -- with the curvature as a second DERIVATIVE (twice polyfit's
      coefficient).
    """
    rv = _write_rv(tmp_path / "t.rv", gammadot=-1.0, gammaddot=0.004)
    orbit = {"name": "trend", "type": "quadratic", "primary": ["A"]}
    system, _ = _system(_config(rv, [orbit]))
    orbit_c = system.orbit
    gd = orbit_c.gammadot.from_internal(orbit_c.gammadot.initval)
    gdd = orbit_c.gammaddot.from_internal(orbit_c.gammaddot.initval)
    assert float(np.atleast_1d(gd)[0]) == pytest.approx(-1.0, rel=1e-6)
    assert float(np.atleast_1d(gdd)[0]) == pytest.approx(0.004, rel=1e-6)

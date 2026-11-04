"""rvinstrument ``data_format: raw_redshift``: the barycentric correction in
the model (``src/exozippy/components/rvinstrument/rvinstrument.md``, "Raw
redshifts"; ``redshift.py``).

Pinned here:

* the correction itself against barycorrpy (Kanodia & Wright 2018, the
  reference implementation of Wright & Eastman 2014) on four sites x four
  stars x 40 epochs, Barnard's star 25 yr from its epoch among them, from a
  committed fixture (``tests/fixtures/raw_redshift/``; barycorrpy is not a
  dependency);
* the forward model's own consistency: barycorrpy, given the redshifts this
  model predicts for a companion-free star, recovers a CONSTANT velocity;
* the size of what the mode exists to model, on a Gaia-4-like system;
* the spelling and its refusals;
* on the simulated example (``examples/raw_redshift_sim``): the likelihood's
  graph equals the numpy forward model at the truth, its gradients are
  finite on both backends, the plots draw reflex + residual, and a MAP fit
  recovers the injected planet and zero points.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from exozippy.components.rvinstrument.redshift import (
    barycentric_redshift,
    observer_terms,
    predicted_cz,
)
from exozippy.constants import C_MPS

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "raw_redshift_sim"
FIXTURE = ROOT / "tests" / "fixtures" / "raw_redshift" / "barycorrpy_zb.txt"


def _load_make_data():
    spec = importlib.util.spec_from_file_location(
        "raw_redshift_make_data", EXAMPLE / "make_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MD = _load_make_data()


def _read_fixture():
    stars, sites, rows = [], [], []
    for line in FIXTURE.read_text().splitlines():
        if line.startswith("# STAR ") and "name" not in line:
            f = line.split()[2:]
            stars.append((f[0], *map(float, f[1:])))
        elif line.startswith("# SITE "):
            f = line.split()[2:]
            sites.append((f[0], *map(float, f[1:])))
        elif not line.startswith("#"):
            a, b, t, v = line.split()
            rows.append((int(a), int(b), float(t), float(v)))
    return stars, sites, np.array(rows)


def _observer(site, jd_utc, ephemeris="builtin"):
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    _, lon, lat, alt = site
    loc = EarthLocation.from_geodetic(lon, lat, alt)
    t_tdb = Time(jd_utc, format="jd", scale="utc", location=loc).tdb.jd
    return t_tdb, observer_terms(t_tdb, f"{lon},{lat},{alt}", ephemeris)


def _internal(star):
    _, ra, dec, pmra, pmdec, px, rv, ep = star
    return (np.radians(ra), np.radians(dec), pmra, pmdec, px, rv), ep


# ---------------------------------------------------------------------------
# 1. The correction against barycorrpy
# ---------------------------------------------------------------------------


def test_barycentric_redshift_matches_barycorrpy():
    """
    Given barycorrpy's c z_B (DE430) for four sites, four stars (Barnard's
      star at J2016 and at J2000, tau Ceti, a Gaia-4-like star) and 40
      epochs over 2015-2026,
    When redshift.barycentric_redshift computes the same with the builtin
      (erfa) ephemeris -- no download,
    Then every epoch agrees to < 1 cm/s (measured: 0.27 cm/s worst, nearly
      all of it the builtin ephemeris -- with DE430 on both sides it is
      0.12 mm/s above the horizon; the remainder is three refinements this
      model has and barycorrpy lacks, the X(t_B) one barycorrpy's error per
      PINT; rvinstrument.md, "Validation").
    """
    stars, sites, rows = _read_fixture()
    worst = 0.0
    for a, site in enumerate(sites):
        sel = rows[:, 0] == a
        jd = np.unique(rows[sel, 2])
        t_tdb, obs = _observer(site, jd)
        for b, star in enumerate(stars):
            s = sel & (rows[:, 1] == b)
            assert np.array_equal(rows[s, 2], jd)
            st, ep = _internal(star)
            mine = C_MPS * barycentric_redshift(t_tdb, obs, st, ep)
            worst = max(worst, float(np.max(np.abs(mine - rows[s, 3]))))
    assert len(stars) == 4 and len(sites) == 4
    assert worst < 0.01, f"worst |c z_B - barycorrpy| = {worst * 100:.3f} cm/s"


def test_barycorrpy_recovers_a_constant_velocity_from_the_forward_model():
    """
    Given the c z_meas this model PREDICTS for a companion-free star at each
      fixture epoch (the forward direction the likelihood uses),
    When barycorrpy's own correction is applied to it,
      v = c[(1 + z_B)(1 + z_meas) - 1],
    Then v is the star's constant velocity along its reference direction,
      c[gamma_* (1 + rv/c) - 1], to < 1 cm/s at every epoch: the forward
      model and WE14's inverse are one relation, perspective acceleration,
      cross terms and all.
    """
    stars, sites, rows = _read_fixture()
    worst = 0.0
    for a, site in enumerate(sites):
        sel = rows[:, 0] == a
        jd = np.unique(rows[sel, 2])
        t_tdb, obs = _observer(site, jd)
        for b, star in enumerate(stars):
            s = sel & (rows[:, 1] == b)
            st, ep = _internal(star)
            cz, _ = predicted_cz(t_tdb, obs, st, ep)
            v = C_MPS * ((1.0 + rows[s, 3] / C_MPS) * (1.0 + cz / C_MPS) - 1.0)
            worst = max(worst, float(np.ptp(v)))
    assert worst < 0.01, (
        f"peak-to-peak of the recovered v: {worst * 100:.3f} cm/s"
    )


def test_a_spacecraft_velocity_is_its_ephemeris_derivative():
    """
    Given the shipped Gaia ephemeris (positions only, 1-day grid),
    When get_observer_posvel reads a velocity off it,
    Then it is the spline's derivative: it matches a central difference of
      the positions to < 1 mm/s, and is the ~30 km/s of an L2 orbiter.
    """
    from exozippy.ephemeris import get_observer_position, get_observer_posvel

    t = np.linspace(2457000.0, 2459000.0, 7)
    _, v = get_observer_posvel(t, "gaia")
    h = 0.01
    dx = get_observer_position(t + h, "gaia") - get_observer_position(
        t - h, "gaia"
    )
    v_fd = dx / (2 * h) * (149597870700.0 / 86400.0)
    assert np.max(np.abs(v - v_fd)) < 1e-3
    assert np.all(np.abs(np.linalg.norm(v, axis=1) - 3.0e4) < 2.0e3)


# ---------------------------------------------------------------------------
# 2. What the mode models that a corrected-RV fit does not: Gaia-4-like
# ---------------------------------------------------------------------------


def _gaia4_like_terms():
    """The example's epochs (three sites, +/-23 km/s of barycentric
    velocity), its truth, and three versions of the 'corrected' RVs."""
    T = MD.TRUTH
    star = (
        np.radians(T["ra"]),
        np.radians(T["dec"]),
        T["pm_ra"],
        T["pm_dec"],
        T["parallax"],
        T["rv"],
    )
    frozen = star[:2] + (0.0, 0.0) + star[4:]
    out = {k: [] for k in ("inst", "tb", "cz", "cz0", "zb", "zb_frozen")}
    for k, name in enumerate(MD.SITES):
        jd = np.loadtxt(EXAMPLE / f"Gaia4like.{name}.rv")[:, 0]
        cz, tb, t_tdb, obs = MD.predict(name, jd)
        cz0, _, _, _ = MD.predict(name, jd, reflex=False)
        out["inst"].append(np.full(jd.size, k))
        out["tb"].append(tb)
        out["cz"].append(cz)
        out["cz0"].append(cz0)
        out["zb"].append(barycentric_redshift(t_tdb, obs, star, T["epoch"]))
        out["zb_frozen"].append(
            barycentric_redshift(t_tdb, obs, frozen, T["epoch"])
        )
    return {k: np.concatenate(v) for k, v in out.items()}


def _postfit(R, d):
    """RMS of ``d`` left after fitting what a corrected-RV fit would
    absorb anyway: one offset per instrument, a rescaled Keplerian and a
    linear trend."""
    shape = MD.keplerian_rv(R["tb"])
    cols = [(R["inst"] == k).astype(float) for k in range(3)]
    A = np.column_stack(cols + [shape, R["tb"] - R["tb"].mean()])
    x, *_ = np.linalg.lstsq(A, d, rcond=None)
    return float(np.sqrt(np.mean((d - A @ x) ** 2)))


def test_gaia4_like_cross_term_and_pm_consistency_magnitudes():
    """
    Given a Gaia-4-like system (K = 330 m/s, three ground sites, barycentric
      velocities of +/-23 km/s) and the example's epochs,
    When the conventional corrections are compared with the full model,
    Then (rvinstrument.md, "What a corrected-RV fit misses"):
      * applying the correction ADDITIVELY (v = c z_meas + c z_B) instead of
        multiplicatively drops the planet cross term K v_B / c: 1.1 cm/s
        RMS, 1.7 cm/s peak, and essentially none of it is absorbed by a
        refit;
      * computing the correction toward the star's catalog position frozen
        at its epoch (proper motion not applied) is 14 cm/s RMS, 2.9 cm/s
        RMS after a refit absorbs its offset and slope;
      * even the EXACT multiplicative correction with the same, known star
        followed by a Keplerian leaves 2.6 mm/s RMS after the refit: the
        companion's gravitational redshift at the star (1.7-3.5 cm/s, mostly
        absorbed into gamma/K) and the star's own light travel (0.4 mm/s),
        which no corrected-RV fit models.
    """
    R = _gaia4_like_terms()

    def v_true(cz, zb):
        return C_MPS * ((1.0 + zb) * (1.0 + cz / C_MPS) - 1.0)

    berv = C_MPS * R["zb"]
    assert berv.min() < -20e3 and berv.max() > 20e3
    mult = v_true(R["cz"], R["zb"])
    add = R["cz"] + berv
    planet = (mult - add) - (v_true(R["cz0"], R["zb"]) - (R["cz0"] + berv))
    rms = float(np.sqrt(np.mean(planet**2)))
    assert 0.008 < rms < 0.015, rms
    assert 0.012 < np.max(np.abs(planet)) < 0.025
    assert _postfit(R, planet) > 0.8 * rms  # the refit cannot absorb it

    frozen = v_true(R["cz"], R["zb_frozen"]) - mult
    assert 0.10 < np.sqrt(np.mean(frozen**2)) < 0.18
    assert 0.02 < _postfit(R, frozen) < 0.04

    exact = mult - MD.keplerian_rv(R["tb"])
    assert 1e-3 < _postfit(R, exact) < 5e-3


# ---------------------------------------------------------------------------
# 3. The spelling
# ---------------------------------------------------------------------------


def _raw_entry(**over):
    e = {
        "name": "HPF",
        "file": str(EXAMPLE / "Gaia4like.HPF.rv"),
        "data_format": "raw_redshift",
        "observer_location": MD.SITES["HPF"],
        "time_frame": "jd",
        "time_scale": "utc",
        "epoch": MD.TRUTH["epoch"],
    }
    e.update(over)
    return {k: v for k, v in e.items() if v is not None}


def _cfg(entry):
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [entry],
    }


@pytest.mark.parametrize(
    "over, match",
    [
        ({"data_format": "rawz"}, "not one of"),
        ({"observer_location": None}, "needs observer_location"),
        ({"observer_location": "earth"}, "not an observatory"),
        ({"time_location": "-104.0,30.7"}, "second spelling"),
        ({"time_frame": None}, "time_frame: said explicitly"),
        ({"epoch": "J2016"}, "reference epoch"),
        ({"data_format": None}, "belongs to data_format: raw_redshift"),
    ],
)
def test_the_raw_keys_are_validated_at_construction(over, match):
    """
    Given an rvinstrument entry with one raw-redshift key wrong (or a raw
      key on a corrected file),
    When the System is constructed,
    Then it raises naming the key, before any data are read.
    """
    from exozippy.system import System

    with pytest.raises(ValueError, match=match):
        System(_cfg(_raw_entry(**over)), user_params={})


def _prepare(entry, params):
    from exozippy.system import System

    p = {
        "star.A.mass": {"initval": 0.64, "sigma": 0.03},
        "star.A.radius": {"initval": 0.62, "sigma": 0.03},
        "star.A.ra": {"initval": MD.TRUTH["ra"]},
        "star.A.dec": {"initval": MD.TRUTH["dec"]},
        "orbit.b.period": {"initval": MD.TRUTH["period"]},
        "orbit.b.tc": {"initval": MD.TRUTH["tc"]},
    }
    p.update(params)
    system = System(_cfg(entry), user_params=p)
    system.prepare()
    return system


def test_star_rv_needs_a_prior():
    """
    Given a raw-redshift file and no prior on star.rv,
    When the system is prepared,
    Then it raises: the redshifts constrain only (1 + z0)(1 + rv/c) per
      instrument, and the message names star.rv and a Gaia DR3 prior.
    """
    with pytest.raises(ValueError, match=r"needs a prior on\s+star.0.rv"):
        _prepare(_raw_entry(), {})


def test_the_reference_epoch_is_required_without_astrometry():
    """
    Given a raw-redshift file without epoch: and no absolute astrometry,
    When the system is prepared,
    Then it raises naming epoch: -- star.ra/dec/pm at an unstated epoch
      would be propagated from the wrong instant.
    """
    with pytest.raises(ValueError, match="needs epoch:"):
        _prepare(
            _raw_entry(epoch=None),
            {"star.A.rv": {"mu": -2.0e4, "sigma": 500.0}},
        )


def test_a_raw_file_declares_the_star_kinematics_and_seeds_from_corrected_rvs():
    """
    Given a prepared system with one raw-redshift file,
    When its manifest and seeds are read,
    Then the star carries ra, dec, pm_ra, pm_dec, distance (parallax) and
      rv; the plotted/seed times are BJD_TDB (minutes from the UTC input);
      the seed velocities are the start-corrected ones (within the planet's
      +/-K of the systemic velocity, not the +/-23 km/s raw spread), and
      gamma's hint is their mean.
    """
    s = _prepare(_raw_entry(), {"star.A.rv": {"mu": -2.0e4, "sigma": 500.0}})
    for p in ("ra", "dec", "pm_ra", "pm_dec", "distance", "parallax", "rv"):
        assert p in s.star.manifest
    rvi = s.rvinstrument
    raw_t = np.loadtxt(EXAMPLE / "Gaia4like.HPF.rv")[:, 0]
    assert np.all(np.abs(rvi.time - raw_t) * 86400.0 < 600.0)
    seed = rvi.rv_seed * rvi._rv_factor()
    assert np.ptp(rvi.rv * rvi._rv_factor()) > 10e3
    assert np.all(np.abs(seed - MD.TRUTH["rv"]) < 1.5 * MD.TRUTH["K"] + 500)
    assert rvi.gamma_init[0] == pytest.approx(np.mean(seed))


def test_a_corrected_file_is_untouched():
    """
    Given the same file read as today's corrected RVs (no data_format),
    When the system is prepared,
    Then the seeds read the data array itself (rv_seed IS rv) and the star
      declares none of the raw-mode kinematics -- the byte-identity of
      every shipped RV fit rests on this (rvinstrument.md).
    """
    entry = {"name": "HPF", "file": str(EXAMPLE / "Gaia4like.HPF.rv")}
    s = _prepare(entry, {})
    assert s.rvinstrument.rv_seed is s.rvinstrument.rv
    assert not any(s.rvinstrument.raw)
    for p in ("ra", "pm_ra", "rv", "parallax"):
        assert p not in s.star.manifest


# ---------------------------------------------------------------------------
# 4. The likelihood, on the simulated example (slow: one build)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example():
    """The example system with every start at the injected truth."""
    import os

    import yaml

    from exozippy.system import System

    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        cfg = yaml.safe_load(open("raw_redshift_sim.yaml"))
        params = yaml.safe_load(open(cfg["parameter_file"]))
        for name in MD.SITES:
            params[f"rvinstrument.{name}.gamma"] = {
                "initval": MD.gamma_truth(name)
            }
        system = System(cfg, user_params=params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    start = system.get_raw_start(model)
    return system, model, start


def _at(system, model, point, node):
    import pytensor

    fn = pytensor.function(model.free_RVs, node, on_unused_input="ignore")
    return np.asarray(fn(*[point[v.name] for v in model.free_RVs]))


@pytest.mark.slow
def test_the_likelihood_is_the_forward_model_at_the_truth(example):
    """
    Given the example built with its starts at the injected truth,
    When the likelihood's model node is evaluated there,
    Then it equals the numpy forward model that simulated the data (before
      noise) to < 1 mm/s -- the residual is the start's logit round trip.
      The numpy model is handed the graph's TARGET-frame conjunction: a raw
      file retards the orbit, so the fitted `tc` is the OBSERVED (virtual
      transit) time, tc_target plus the planet's light travel
      (Orbit._ltt_reporting_mask), hundreds of seconds here.
    """
    system, model, start = example
    rvi = system.rvinstrument
    mod = _at(system, model, start, rvi._rv_model_data_node) * rvi._rv_factor()
    tc_target = float(
        _at(system, model, start, system.orbit.tc_target.value)[0]
    )
    assert abs(tc_target - MD.TRUTH["tc"]) * 86400.0 > 100.0
    truth = np.concatenate(
        [
            MD.predict(
                name,
                np.loadtxt(EXAMPLE / f"Gaia4like.{name}.rv")[:, 0],
                tc=tc_target,
            )[0]
            for name in MD.SITES
        ]
    )
    assert np.max(np.abs(mod - truth)) < 1e-3


@pytest.mark.slow
def test_gradients_are_finite_on_both_backends(example):
    """
    Given the example at its start,
    When dlogp is evaluated by the C backend and by JAX,
    Then both are finite in every coordinate -- ra/dec/pm/distance/rv and
      the orbit included -- and agree.
    """
    import pytensor
    from pymc.sampling.jax import get_jaxified_graph

    system, model, _ = example
    ip = model.initial_point()
    grads = pytensor.grad(model.logp(), model.value_vars)
    vals = [ip[v.name] for v in model.value_vars]
    g_c = pytensor.function(model.value_vars, grads)(*vals)
    g_j = get_jaxified_graph(inputs=model.value_vars, outputs=grads)(*vals)
    names = [v.name for v in model.value_vars]
    for want in (
        "star.ra",
        "star.pm_ra",
        "star.distance",
        "star.rv",
        "orbit.tc",
    ):
        assert any(n.startswith(want) for n in names), want
    scale = max(float(np.max(np.abs(g))) for g in g_c)
    for name, gc, gj in zip(names, g_c, g_j):
        gj = np.asarray(gj)
        assert np.all(np.isfinite(gc)) and np.all(np.isfinite(gj)), name
        np.testing.assert_allclose(
            gj, gc, rtol=1e-6, atol=1e-9 * scale, err_msg=name
        )


@pytest.mark.slow
def test_the_plots_draw_reflex_plus_residual(example):
    """
    Given the example at the truth,
    When the plotted observations are formed,
    Then they are gamma + the Keplerian + the likelihood residual: within
      the noise (10 m/s) of gamma + reflex, where the raw data sit +/-23
      km/s away.
    """
    system, model, start = example
    rvi = system.rvinstrument
    with model:
        point = system.get_internal_point(model, start)
    plotted = rvi._plotted_rv(system, point) * rvi._rv_factor()
    gammas = np.array([MD.gamma_truth(n) for n in MD.SITES])[rvi.inst_map]
    expected = gammas + MD.keplerian_rv(rvi.time)
    assert np.std(plotted - expected) < 15.0
    assert np.ptp(rvi.rv * rvi._rv_factor() - expected) > 10e3


@pytest.mark.slow
def test_a_map_fit_recovers_the_injected_planet_and_zero_points(example):
    """
    Given the simulated raw redshifts (K = 330 m/s, 75 spectra at 10 m/s)
      and a start moved off the truth in K and tc,
    When the posterior is maximized,
    Then K and every instrument's gamma come back within ~3 sigma of the
      injection, P within its (loose, 1.3-orbit) constraint, and the star's rv stays on its prior (the data cannot move
      it: rvinstrument.md, "What the data constrain").
    """
    from scipy.optimize import minimize

    system, model, start = example
    rvs = model.free_RVs
    names = [v.name for v in model.value_vars]
    sizes = [np.size(start[v.name]) for v in rvs]
    logp = model.compile_logp()
    dlogp = model.compile_dlogp()

    def unpack(x):
        out, k = {}, 0
        for v, n in zip(rvs, sizes):
            out[v.name] = np.reshape(x[k : k + n], np.shape(start[v.name]))
            k += n
        return out

    x0 = np.concatenate([np.ravel(start[v.name]) for v in rvs]).astype(float)
    k_msini = names.index("planet.msini_raw")
    k_tc = names.index("orbit.tc_raw")
    off = np.cumsum([0] + sizes)
    x0[off[k_msini]] += 0.05
    x0[off[k_tc]] += 0.05
    res = minimize(
        lambda x: -logp(unpack(x)),
        x0,
        jac=lambda x: -np.asarray(dlogp(unpack(x)), dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 5000},
    )
    best = unpack(res.x)
    rvi = system.rvinstrument
    K = (
        float(_at(system, model, best, system.orbit.K.value)[0])
        * rvi._rv_factor()
    )
    P = float(_at(system, model, best, system.orbit.period.value)[0])
    g = _at(system, model, best, rvi.gamma.value) * rvi._rv_factor()
    rv = float(_at(system, model, best, system.star.rv.value)[0])
    print(
        f"MAP: K = {K:.2f} (330), P = {P:.2f} (571), rv = {rv:.0f} "
        f"(-20000), gamma - truth = "
        f"{[round(float(gi - MD.gamma_truth(n)), 2) for n, gi in zip(MD.SITES, g)]}"
    )
    assert abs(K - MD.TRUTH["K"]) < 6.0, K
    # 75 spectra over 750 d cover 1.3 orbits: P is the loosest element (the
    # likelihood is nearly flat over 571-579 d on this noise realization,
    # logp 293.9 at 574.7 d and 295.5 at 578.8 d), K and gamma are not.
    assert abs(P - MD.TRUTH["period"]) < 15.0, P
    for name, gi in zip(MD.SITES, g):
        assert abs(gi - MD.gamma_truth(name)) < 10.0, (name, gi)
    assert abs(rv - MD.TRUTH["rv"]) < 1500.0
